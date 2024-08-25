import json
import os
from pathlib import Path
import re
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union
from uuid import UUID, uuid4

from sqlalchemy import text
import typer
from src.product_chains.product_combined_info import (
    ProductCombinedInformation,
    describe_image,
    encode_image,
    summarise_product_info,
)

# Vector Store for Document Embeddings
from langchain_chroma import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_openai import OpenAIEmbeddings

from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_core.documents import Document

from chromadb.errors import InvalidDimensionException

from src.product_chains.schemas import RAWProduct
from src.pipelines.description_pipeline import generate_concise_description
from src.database import RawDBSessionLocal, SessionLocal
from src.models.models import ProductModel, ReviewModel
from src.schemas import (
    CategoryConfigCode,
    Product,
    ProductList,
    ProductSubCategory,
    Review,
    ReviewList,
    ProductUpdate,
    ReviewUpdate,
    TextLLM,
    VisionLLM,
)
from bs4 import BeautifulSoup
from loguru import logger
from src.utils import StorageOption, filter_review_date
from langchain_core.vectorstores import VectorStore

from langchain.storage.exceptions import InvalidKeyException

store_db_session = SessionLocal()

MAXIMUM_GPT_ITERATIONS = 20

from langchain_core.stores import BaseStore

from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field, ValidationError
from datasets import Dataset
from langchain_community.chat_models import ChatOllama


class ProductAnalysisItem(BaseModel):
    name: str = Field(description="Name of the product")
    product_asin: str = Field(description="Identification number")
    total_customers_that_rated: str = Field(
        description="The number of people that rater this product"
    )
    overall_ratings: str = Field(description="The ratings for this product")
    brand: str = Field(description="The brand name for this item")
    available_written_reviews: str = Field(
        description="The number of people that wrote review for this ietm"
    )
    weather: str = Field(description="Best weather this item can be used in.")
    usage: str = Field("In what situation can this item be used for")
    type: str = Field(
        description="What is the best category this item can be placed in"
    )
    material: str = Field(description="What is the product made of?")
    color: str = Field(description="What is the colour of the product?")
    target_audience: str = Field(
        description="Who is the best audience this item is targetted to"
    )
    price: str = Field(description="The price tag for this product.")
    currency: str = Field(description="The currency for the product's price")
    ciam_category: str = Field(description="The CIAM category for this item")


def generate_analysis_data(
    raw_data_file_path_str: str,
    psi_file_path_str: str,
    pid_file_path_str: str,
    analysis_output_path: str,
    llm_choice: TextLLM = TextLLM.LLAMA_3_1_INSTRUCT,
    number_of_products: int = -1,
    name: str = "analysis",
):
    raw_data_file_path = Path(raw_data_file_path_str)
    psi_file_path = Path(psi_file_path_str)
    pid_file_path = Path(pid_file_path_str)

    assert raw_data_file_path.exists()
    assert pid_file_path.exists()
    assert psi_file_path.exists()

    output_file_path = Path(analysis_output_path + f"/{name}-{llm_choice.lower()}.json")

    product_summaries = json.loads(psi_file_path.read_text())
    product_image_descriptions = json.loads(pid_file_path.read_text())
    raw_products = json.loads(raw_data_file_path.read_bytes())

    products_combined_infos = {}
    for rp in raw_products:
        if rp["product_asin"] in product_summaries:
            products_combined_infos[rp["product_asin"]] = ProductCombinedInformation(
                product=RAWProduct.model_validate(rp)
            )

    # Everything must be in equal
    # TODO: Check that same item is in the exact position in all dict
    assert (
        len(product_image_descriptions)
        == len(product_summaries)
        == len(products_combined_infos)
    )

    product_image_descriptions = sorted(
        product_image_descriptions.items(), key=lambda x: x[0]
    )
    product_summaries = sorted(product_summaries.items(), key=lambda x: x[0])
    products_combined_infos = sorted(
        products_combined_infos.items(), key=lambda x: x[0]
    )

    actual_number_of_products = number_of_products
    if number_of_products != -1:
        logger.info(f"Clipping number of products to {number_of_products}")
        products_combined_infos = products_combined_infos[0:number_of_products]
        product_summaries = product_summaries[0:number_of_products]
        product_image_descriptions = product_image_descriptions[0:number_of_products]
    else:
        actual_number_of_products = len(product_summaries)
        number_of_products = len(product_summaries)
        logger.info(f"Generating analysis data for {actual_number_of_products} data.")

    if number_of_products < -1:
        logger.error("Invalid number of products to process, please use >= -1")
        raise typer.Exit()

    if llm_choice == TextLLM.GPT_4O and (
        number_of_products > 2 or number_of_products == -1
    ):
        logger.error("Nah!!!! Too costly")
        raise typer.Exit(1)

    # Load these data into Document
    ## Product Info Documents
    product_info_docs = [
        Document(
            page_content=product_info.info(),
            metadata={
                "id": product_info.product.id,
                "name": product_info.product.name,
                "product_asin": product_info.product.product_asin,
            },
        )
        for product_asin, product_info in products_combined_infos
    ]
    # summary_docs = [
    #     Document(
    #         page_content=summary_text,
    #         metadata={
    #             "product_asin": product_asin,
    #         },
    #     )
    #     for product_asin, summary_text in product_summaries
    # ]
    # image_descriptions_docs = [
    #     Document(
    #         page_content=description_text,
    #         metadata={
    #             "product_asin": product_asin,
    #         },
    #     )
    #     for product_asin, description_text in product_image_descriptions
    # ]

    questions = {
        "brand": "What is the brand or maker of this product? Return back only the name of the brand/maker. If you do not know it, return back UNKNOWN.",
        "weather": "Which of this best suit the condition this product can be used in Summer, Winter, Spring, autumn. If all return back 'Transitional' to indicate it could be used in between summer and winter. If multiple seperate them with a forward slash '/'",
        "usage": "Is this item best for Casual, Sport, Athleisure, Format, or Outdoor. If the answer is not in here, list out the most appropriate answer. If multiple, seperate them with a '/' no space.",
        "type": "What is the common name for this item? Return only one word, pluralised.",
        "material": "What material is used to make this product.Return back only the name of the material and If you can't find the answer return back UNKNOWN.",
        "color": "What is the primary colour of this product? Check the image description before you check the product details. If you can't find the answer return back UNKNOWN. If multiple separate them with a '/",
        "target_audience": "Who are the target aduience for this item Kids, Teens, Adults, Any, Kids and Teens, Kids and Adults or Teens and Adults?",
    }

    prompt = """Answer the following questions about this product and return back your answers with the questions' key while replacing the value with your answer for each question.

    Questions:
    {questions}

    Product Details:
    {product_details}

    Product Image Description:
    {image_description}

    Only respond with a correct JSON, no comment no explanation. The output must be a well formatted JSON output.
    Follow this pattern:
    {{
        "brand": "",
        "weather": "",
        "usage": "",
        "type": "",
        "material": "",
        "color": "",
        "target_audience": ""
    }}
    """

    model_name = TextLLM.LLAMA_3_1_INSTRUCT
    model = ChatOllama(model=model_name)
    chain_prompt = ChatPromptTemplate.from_template(prompt)

    output = {}
    analysis_items = []
    max_retries = 5

    for index, product_doc in enumerate(product_info_docs[0:actual_number_of_products]):
        # reset
        try_times = 1

        product_combo_instance = products_combined_infos[index][1].product
        logger.success(
            f"{index + 1}/{actual_number_of_products} -> {product_combo_instance.product_asin}\n"
        )

        if output_file_path.exists():
            analysis_bag = json.loads(output_file_path.read_text())
            assert isinstance(analysis_bag, dict)
        else:
            analysis_bag = {}

        if product_combo_instance.product_asin not in analysis_bag:
            while try_times < max_retries:
                try:
                    image_description = product_image_descriptions[index]
                    chain = chain_prompt | model | JsonOutputParser()
                    output = chain.invoke(
                        {
                            "questions": questions,
                            "product_details": product_doc.page_content,
                            "image_description": image_description,
                        }
                    )

                    output["price"] = str(product_combo_instance.price)
                    output["product_asin"] = product_combo_instance.product_asin
                    output["name"] = product_combo_instance.name
                    output["overall_ratings"] = str(
                        product_combo_instance.overall_ratings
                    )
                    output["available_written_reviews"] = str(
                        len(product_combo_instance.reviews)
                    )
                    output["total_customers_that_rated"] = str(
                        product_combo_instance.total_customers_that_rated
                    )
                    output["currency"] = product_combo_instance.currency
                    output["ciam_category"] = product_combo_instance.category
                    _analysis_item = ProductAnalysisItem.model_validate(output)
                    analysis_items.append(_analysis_item.model_dump())

                    analysis_bag[product_combo_instance.product_asin] = (
                        _analysis_item.model_dump()
                    )
                    output_file_path.write_text(json.dumps(analysis_bag))
                    break
                except (ValidationError, OutputParserException) as e:
                    try_times += 1

                    if try_times == max_retries:
                        logger.error(
                            f"\n\nSkipping after {max_retries} retries. {product_combo_instance.product_asin} due to ane error {e}\n"
                        )
                        break

        index += 1


class CIAMDocumentFileStore(BaseStore[str, Document]):

    def __init__(
        self,
        root_path: Union[str, Path],
        *,
        chmod_file: Optional[int] = None,
        chmod_dir: Optional[int] = None,
        update_atime: bool = False,
    ) -> None:
        """Implement the BaseStore interface for the local file system.

        Args:
            root_path (Union[str, Path]): The root path of the file store. All keys are
                interpreted as paths relative to this root.
            chmod_file: (optional, defaults to `None`) If specified, sets permissions
                for newly created files, overriding the current `umask` if needed.
            chmod_dir: (optional, defaults to `None`) If specified, sets permissions
                for newly created dirs, overriding the current `umask` if needed.
            update_atime: (optional, defaults to `False`) If `True`, updates the
                filesystem access time (but not the modified time) when a file is read.
                This allows MRU/LRU cache policies to be implemented for filesystems
                where access time updates are disabled.
        """
        self.root_path = Path(root_path).absolute()
        self.chmod_file = chmod_file
        self.chmod_dir = chmod_dir
        self.update_atime = update_atime

        """Initialize an empty store."""

        self.store: Dict[str, Document] = {}

        root_path_str = self.root_path
        if type(self.root_path) == Path:
            root_path_str = self.root_path.absolute()

        for _path in os.listdir(root_path_str):
            path = Path(_path)
            if path.exists():
                value = path.read_text()
                content = json.loads(value)

                document = Document(
                    page_content=content["page_content"],
                    id=content["id"],
                    metadata=content["metadata"],
                )
                self.store[_path] = document

    def _get_full_path(self, key: str) -> Path:
        """Get the full path for a given key relative to the root path.

        Args:
            key (str): The key relative to the root path.

        Returns:
            Path: The full path for the given key.
        """
        if not re.match(r"^[a-zA-Z0-9_.\-/]+$", key):
            raise InvalidKeyException(f"Invalid characters in key: {key}")
        full_path = os.path.abspath(self.root_path / key)
        common_path = os.path.commonpath([str(self.root_path), full_path])
        if common_path != str(self.root_path):
            raise InvalidKeyException(
                f"Invalid key: {key}. Key should be relative to the full path."
                f"{self.root_path} vs. {common_path} and full path of {full_path}"
            )

        return Path(full_path)

    def mset(self, key_value_pairs: Sequence[Tuple[str, Document]]) -> None:
        """Set the values for the given keys.

        Args:
            key_value_pairs: A sequence of key-value pairs.

        Returns:
            None
        """
        for key, value in key_value_pairs:
            self.store[key] = value

            full_path = self._get_full_path(key)
            self._mkdir_for_store(full_path.parent)
            content = {
                "metadata": value.metadata,
                "content": value.page_content,
                "id": value.id,
            }
            full_path.write_text(json.dumps(content))
            if self.chmod_file is not None:
                os.chmod(full_path, self.chmod_file)

    def mget(self, keys: Sequence[str]) -> List[Optional[Document]]:
        """Get the values associated with the given keys.

        Args:
            keys: A sequence of keys.

        Returns:
            A sequence of optional values associated with the keys.
            If a key is not found, the corresponding value will be None.
        """

        values: List[Optional[Document]] = []
        for key in keys:
            values.append(self.store[key])

            # full_path = self._get_full_path(key)
            # if full_path.exists():
            #     value = full_path.read_text()
            #     content = json.loads(value)

            #     values.append(
            #         Document(
            #             page_content=content["page_content"],
            #             id=content["id"],
            #             metadata=content["metadata"],
            #         )
            #     )
            #     if self.update_atime:
            #         # update access time only; preserve modified time
            #         os.utime(full_path, (time.time(), os.stat(full_path).st_mtime))
            # else:
            #     values.append(Document(page_content="Empty Document", id=None))

        return values

    def _mkdir_for_store(self, dir: Path) -> None:
        """Makes a store directory path (including parents) with specified permissions

        This is needed because `Path.mkdir()` is restricted by the current `umask`,
        whereas the explicit `os.chmod()` used here is not.

        Args:
            dir: (Path) The store directory to make

        Returns:
            None
        """
        if not dir.exists():
            self._mkdir_for_store(dir.parent)
            dir.mkdir(exist_ok=True)
            if self.chmod_dir is not None:
                os.chmod(dir, self.chmod_dir)

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys and their associated values.

        Args:
            keys (Sequence[str]): A sequence of keys to delete.
        """
        for key in keys:
            if key in self.store:
                del self.store[key]

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:  # type: ignore
        """Get an iterator over keys that match the given prefix.

        Args:
            prefix (str, optional): The prefix to match. Defaults to None.

        Yields:
            Iterator[str]: An iterator over keys that match the given prefix.
        """
        if prefix is None:
            yield from self.store.keys()
        else:
            for key in self.store.keys():
                if key.startswith(prefix):
                    yield key


# Helper function to add documents into the vector and the doument store
def add_documents_no_split(
    id_key: str,
    summary_to_embeds: List[str],
    combined_product_infos: List[ProductCombinedInformation],
    docstore: CIAMDocumentFileStore,
    vectorstore: VectorStore,
):
    product_ids = [
        str(product_info_obj.product.id) for product_info_obj in combined_product_infos
    ]
    data = list(zip(product_ids, summary_to_embeds, combined_product_infos))

    docs = []
    parent_docs_contents = [
        Document(
            page_content=product_info.info(),
            metadata={
                id_key: product_info.product.id,
                "id": product_info.product.id,
                "name": product_info.product.name,
                "product_asin": product_info.product.product_asin,
            },
        )
        for product_info in combined_product_infos
    ]

    for single_item in data:
        product_id, content_to_embed, product_info = single_item
        docs.append(
            Document(
                page_content=content_to_embed,
                metadata={
                    id_key: product_id,
                    "id": product_id,
                    "name": product_info.product.name,
                    "product_asin": product_info.product.product_asin,
                },
            )
        )

    assert len(docs) == len(parent_docs_contents)

    vectorstore.add_documents(docs, ids=product_ids)
    docstore.mset(list(zip(product_ids, parent_docs_contents)))


def generate_embeddings(
    raw_data_file_path_str: str,
    psi_file_path_str: str,
    pid_file_path_str: str,
    embedding_cache_folder: str,
    document_store_cache_folder: str,
):
    raw_data_file_path = Path(raw_data_file_path_str)
    psi_file_path = Path(psi_file_path_str)
    pid_file_path = Path(pid_file_path_str)

    assert raw_data_file_path.exists()
    assert pid_file_path.exists()
    assert psi_file_path.exists()

    product_summaries = json.loads(psi_file_path.read_text())
    product_image_descriptions = json.loads(pid_file_path.read_text())
    raw_products = json.loads(raw_data_file_path.read_bytes())

    products_combined_infos = {}
    for rp in raw_products:
        if rp["product_asin"] in product_summaries:
            products_combined_infos[rp["product_asin"]] = ProductCombinedInformation(
                product=RAWProduct.model_validate(rp)
            )

    # Everything must be in equal
    # TODO: Check that same item is in the exact position in all dict
    assert (
        len(product_image_descriptions)
        == len(product_summaries)
        == len(products_combined_infos)
    )

    product_image_descriptions = sorted(
        product_image_descriptions.items(), key=lambda x: x[0]
    )
    product_summaries = sorted(product_summaries.items(), key=lambda x: x[0])
    products_combined_infos = sorted(
        products_combined_infos.items(), key=lambda x: x[0]
    )

    text_embedding_model_name = "nomic-embed-text"
    underlying_embedding = OpenAIEmbeddings()

    # for nomic-embed, embed_instruction is `search_document` to embed documents for RAG and `search_query` to embed the question
    underlying_embedding = OllamaEmbeddings(
        model=text_embedding_model_name,
        embed_instruction="search_document",
        query_instruction="search_query",
    )

    embedding_store = LocalFileStore(embedding_cache_folder)
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=underlying_embedding,
        document_embedding_cache=embedding_store,
        namespace=underlying_embedding.model,
    )

    collection_name = f"fashion_store_mrag_v_{underlying_embedding.model}"
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=cached_embedder,
        # https://docs.trychroma.com/guides#changing-the-distance-function
        # Cosine, 1 means most similar, 0 means orthogonal, -1 means opposite
        collection_metadata={"hnsw:space": "cosine"},  # l2 is the default
        # embedding_function=OpenCLIPEmbeddings(model=None, preprocess=None, tokenizer=None, model_name=model_name, checkpoint=checkpoint)
    )

    # Setup the document store
    document_store = CIAMDocumentFileStore(document_store_cache_folder)
    add_documents_no_split(
        id_key="product_id",
        summary_to_embeds=[psi[1] for psi in product_summaries],
        combined_product_infos=[pci[1] for pci in products_combined_infos],
        docstore=document_store,
        vectorstore=vectorstore,
    )

    add_documents_no_split(
        id_key="product_id",
        summary_to_embeds=[pid[1] for pid in product_image_descriptions],
        combined_product_infos=[pci[1] for pci in products_combined_infos],
        docstore=document_store,
        vectorstore=vectorstore,
    )


def generate_pid(
    pci_json_file_path_str: str,
    images_directory: str,
    pid_json_output_folder_path_str: str,
    k: int = 10,
    llm_choice: VisionLLM = VisionLLM.LLAVA_VICUNA_Q4_0,
):
    """Generates the product image description for the given PCI JSON file.

    Args:
        pci_json_file_path_str (str): The file containing the product combined information
        pid_json_output_folder_path_str (str): The folder to save the product summary information.
        k: (int): The total number of products to generate summary for.
        llm_choice (TextLLM, optional): The model to use to for summary generation. Defaults to TextLLM.LLAMA_3_1.

    Raises:
        FileNotFoundError: If `pci_json_file_path_str` does not exist.
        ValueError: If any other error occurs.
    """

    pci_file_path = Path(pci_json_file_path_str)
    if not pci_file_path.exists():
        raise FileNotFoundError(f"{pci_json_file_path_str} does not exist.")

    product_combined_infos = json.loads(pci_file_path.read_text())
    assert isinstance(product_combined_infos, dict), "pcis must be dict"

    if llm_choice == TextLLM.GPT_4O and k > MAXIMUM_GPT_ITERATIONS:
        raise ValueError(
            f"{MAXIMUM_GPT_ITERATIONS} iterations! This is expensive! Please change LLM from {llm_choice}"
        )

    output_file_path = Path(
        pid_json_output_folder_path_str + f"/pid-{llm_choice.lower()}.json"
    )
    if output_file_path.exists():
        pids = json.loads(output_file_path.read_text())
        assert isinstance(pids, dict)
    else:
        pids = {}

    if k <= 0:
        k = len(product_combined_infos)

    for index, product_info_data in enumerate(product_combined_infos.items()):
        # We did this to ensure we save the latest summary should in case any error
        # occurs, we can continue from where we stopped.
        if output_file_path.exists():
            pids = json.loads(output_file_path.read_text())
            assert isinstance(pids, dict)
        else:
            pids = {}

        logger.info(f"{index + 1}/{k}\n")

        product_asin, product_info_str = product_info_data
        if product_asin not in pids:
            image_path = Path(f"{images_directory}/{product_asin}.png")
            if image_path.exists():
                pids[product_asin] = describe_image(
                    img_base64=encode_image(str(image_path.absolute())),
                    model_name=llm_choice,
                )
            else:
                logger.error(f"Image description not available: {product_asin}")
                pids[product_asin] = "Image description not available"

        Path(output_file_path).write_text(json.dumps(pids))

        if (index + 1) == k:
            break

    print(output_file_path.absolute())


def generate_psi(
    pci_json_file_path_str: str,
    psi_json_output_folder_path_str: str,
    k: int = 10,
    llm_choice: TextLLM = TextLLM.LLAMA_3_1,
):
    """Generates the product summary information for the given PCI JSON file.

    Args:
        pci_json_file_path_str (str): The file containing the product combined information
        psi_json_output_folder_path_str (str): The folder to save the product summary information.
        k: (int): The total number of products to generate summary for.
        llm_choice (TextLLM, optional): The model to use to for summary generation. Defaults to TextLLM.LLAMA_3_1.

    Raises:
        FileNotFoundError: If `pci_json_file_path_str` does not exist.
        ValueError: If any other error occurs.
    """

    pci_file_path = Path(pci_json_file_path_str)
    if not pci_file_path.exists():
        raise FileNotFoundError(f"{pci_json_file_path_str} does not exist.")

    product_combined_infos = json.loads(pci_file_path.read_text())
    assert isinstance(product_combined_infos, dict), "pcis must be dict"

    if llm_choice == TextLLM.GPT_4O and k > MAXIMUM_GPT_ITERATIONS:
        raise ValueError(
            f"{MAXIMUM_GPT_ITERATIONS} iterations! This is expensive! Please change LLM from {llm_choice}"
        )

    output_file_path = Path(
        psi_json_output_folder_path_str + f"/psi-{llm_choice.lower()}.json"
    )
    if output_file_path.exists():
        psis = json.loads(output_file_path.read_text())
        assert isinstance(psis, dict)
    else:
        psis = {}

    if k <= 0:
        k = len(product_combined_infos)

    for index, product_info_data in enumerate(product_combined_infos.items()):
        # We did this to ensure we save the latest summary should in case any error
        # occurs, we can continue from where we stopped.
        if output_file_path.exists():
            psis = json.loads(output_file_path.read_text())
            assert isinstance(psis, dict)
        else:
            psis = {}

        logger.info(f"{index + 1}/{k}\n")

        product_asin, product_info_str = product_info_data
        if product_asin not in psis:
            psis[product_asin] = summarise_product_info(
                product_info=product_info_str, model_name=llm_choice
            )

        Path(output_file_path).write_text(json.dumps(psis))

        if (index + 1) == k:
            break

    print(output_file_path.absolute())


def generate_pci(
    folder_path_str: str,
    k: int = 10,
    sub_category: CategoryConfigCode = CategoryConfigCode.FASHION_MEN,
):
    """Generates the combined product information for each product in the store's database.

    Args:
        folder_path_str (str): The directory to save the `pci.json` file into.
        k (int, optional): The number of products to generate for. -1 means all. Defaults to 10.
        sub_category (CategoryConfigCode, optional): The category to fetch the products from. Defaults to CategoryConfigCode.FASHION_MEN.

    Raises:
        ValueError: If any error occurs.
    """

    query = store_db_session.query(ProductModel).filter(
        ProductModel.sub_category == sub_category
    )
    if k is not None and k > 0:
        query = query.limit(k)

    products = query.all()

    output_directory_path = Path(folder_path_str)
    output_file_path = Path(f"{output_directory_path.absolute()}/pci.json")
    if not output_directory_path.exists():
        os.makedirs(folder_path_str, exist_ok=True)

    pcis = {}
    if output_file_path.exists():
        pcis = json.loads(output_file_path.read_text())
        assert isinstance(pcis, dict), "existing PCIs must be in dict form"

    for index, product in enumerate(products):
        logger.info(f"{index + 1}/{len(products)}")

        product_info = ProductCombinedInformation(
            product=RAWProduct.model_validate(product, from_attributes=True)
        )
        pcis[product.product_asin] = product_info.info_for_summary()

    output_file_path.write_text(json.dumps(pcis))


def revise_product_descriptions(
    k: int = 10,
    sub_category: ProductSubCategory = ProductSubCategory.FASHION_MEN,
):
    products = (
        store_db_session.query(ProductModel)
        .filter(
            ProductModel.sub_category == sub_category,
            ProductModel.revised_description == None,
        )
        .limit(k)
    )

    for product in products:
        logger.info(f"Generating for product: {product.name}")
        product.revised_description = generate_concise_description(product=product)
        store_db_session.add(product)
        store_db_session.commit()
        store_db_session.refresh(product)


def update_store_product(product_data: ProductUpdate):
    product_db = get_product(product_data.id)
    for key, value in product_data.model_dump(exclude_none=True).items():
        setattr(product_db, key, value)

    store_db_session.add(product_db)
    store_db_session.commit()
    store_db_session.refresh(product_db)
    return product_db


def update_store_product_review(review_data: ReviewUpdate):
    review_db = get_product_review(review_data.id)
    for key, value in review_data.model_dump(exclude_none=True).items():
        setattr(review_db, key, value)

    store_db_session.add(review_db)
    store_db_session.commit()
    store_db_session.refresh(review_db)
    return review_db


def get_product_review(product_id: UUID):
    existing_review = (
        store_db_session.query(ReviewModel).filter(ReviewModel.id == product_id).first()
    )
    if not existing_review:
        raise Exception("Review does not exist.")

    return existing_review


def get_product(product_id: UUID):
    existing_product = (
        store_db_session.query(ProductModel)
        .filter(ProductModel.id == product_id)
        .first()
    )
    if not existing_product:
        raise Exception("Product does not exist.")

    return existing_product


def get_products():
    return store_db_session.query(ProductModel).all()


def insert_store_product(product: Product):
    existing_product = (
        store_db_session.query(ProductModel)
        .filter(ProductModel.id == product.id)
        .first()
    )
    if not existing_product:
        logger.info("Inserting new product into the store's db")
        product_model = ProductModel(**product.model_dump())
        store_db_session.add(product_model)
        store_db_session.commit()
        store_db_session.refresh(product_model)
        return product_model
    else:
        logger.warning("Duplicate product in the store's db")

    return existing_product


def insert_store_review(review: Review):
    existing_review = (
        store_db_session.query(ReviewModel)
        .filter(ReviewModel.review_ref == review.review_ref)
        .first()
    )
    if not existing_review:
        review_model = ReviewModel(**review.model_dump())
        store_db_session.add(review_model)
        store_db_session.commit()
        store_db_session.refresh(review_model)
        return review_model
    else:
        logger.warning(f"Duplicate review {review.review_ref} in the store's db")

    return existing_review


def fetch_raw_data(
    k: int | None = None,
    randomise: bool = False,
    only_products_with_reviews: bool = True,
    config_category_ref_code: CategoryConfigCode | None = None,
    storage_option: StorageOption = StorageOption.DATABASE,
    folder_name: str | None = None,
):

    products_with_reviews = []
    Path(f"./{folder_name}/images").mkdir(parents=True, exist_ok=True)

    if storage_option not in [StorageOption.DATABASE, StorageOption.JSON]:
        raise ValueError("Invalid Storage Option: Supported are DATABASE and JSON")

    if (
        config_category_ref_code == None
        or config_category_ref_code == CategoryConfigCode.ALL
    ):
        categories_refs = [
            CategoryConfigCode.FASHION_MEN.value,
            CategoryConfigCode.FASHION_WOMEN.value,
            CategoryConfigCode.BEAUTY_SKIN_CARE.value,
        ]
    else:
        categories_refs = [config_category_ref_code.value]

    for category_ref in categories_refs:
        logger.info(f"Fetching {category_ref}\n\n")
        distinct_data = "products.config_category_ref_code"
        group_by_data = "products.data_asin, products.config_category_ref_code"
        if randomise:
            distinct_data = "products.config_category_ref_code, RANDOM()"

        raw_db_session = RawDBSessionLocal()
        sql = f"""
            SELECT products.* 
            FROM products
        """

        # if only_products_with_reviews:
        #     sql += "\nJOIN reviews ON products.data_asin = reviews.product_asin"

        where_sql = ""

        where_sql += f"products.config_category_ref_code in ('{category_ref}')"

        if len(where_sql) > 0:
            sql += "\nWHERE " + where_sql

        sql += f"\n GROUP BY {group_by_data}"

        sql += f"\n ORDER BY {distinct_data}"

        if k is not None and k > 0:
            sql += f" \n LIMIT {k} \n"

        print(sql)
        result = raw_db_session.execute(text(sql))

        products = []
        for row in result.all():
            row_item = {}

            for key, value in enumerate(row._fields):

                content = row[key]
                if value != "image_data":
                    if value == "description":
                        soup = BeautifulSoup(row[key], features="html.parser")
                        content = " ".join(soup.text.split("\n"))

                    row_item[value] = content
                elif value == "image_data":
                    Path(
                        f"./{folder_name}/images/{row_item['data_asin']}.png"
                    ).write_bytes(
                        content  # type: ignore
                    )

            products.append(Product.model_validate(row_item, from_attributes=True))

        processed_products = ProductList.model_validate({"products": products})

        if storage_option == StorageOption.DATABASE:
            product_models = list(
                map(insert_store_product, processed_products.products)
            )
        elif storage_option == StorageOption.JSON:
            if not only_products_with_reviews:
                if Path(f"./{folder_name}/raw_products.json").exists():
                    os.remove(f"./{folder_name}/raw_products.json")

                Path(f"./{folder_name}/raw_products.json").write_text(
                    processed_products.model_dump_json()
                )
            logger.success("Product raw data stored into raw_products.json")
        else:
            raise ValueError("Invalid Storage Option")

        if only_products_with_reviews:
            for product in list(processed_products.products):
                sql = f"SELECT * FROM reviews WHERE product_asin = '{product.product_asin}'"
                db_result = raw_db_session.execute(text(sql))
                reviews = []
                for row in db_result.all():
                    row_item = {"product_id": product.id, "id": uuid4()}

                    for key, value in enumerate(row._fields):
                        content = row[key]
                        if value == "review_date":
                            content = filter_review_date(content)

                        row_item[value] = content

                    reviews.append(
                        Review.model_validate(row_item, from_attributes=True)
                    )

                product_reviews = ReviewList.model_validate({"reviews": reviews})
                product_json = {
                    **product.model_dump(mode="json"),
                    **product_reviews.model_dump(mode="json"),
                }
                products_with_reviews.append(product_json)

                logger.success(
                    f"Product: {product.id} - Reviews: {len(product_reviews.reviews)}"
                )
                if storage_option == StorageOption.DATABASE:
                    db_reviews_result = list(
                        map(insert_store_review, product_reviews.reviews)
                    )
                    raw_db_session.close()
                    logger.success("Product raw data stored into the store's database")
                elif storage_option == StorageOption.JSON:
                    if Path(f"./{folder_name}//raw_products.json").exists():
                        os.remove(f"./{folder_name}//raw_products.json")
                    Path(f"./{folder_name}//raw_products.json").write_text(
                        json.dumps(products_with_reviews)
                    )
                    logger.success(
                        "Product with reviews raw data stored into raw_products.json"
                    )
                else:
                    raise ValueError("Invalid Storage Option")
