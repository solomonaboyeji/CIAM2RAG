import json
import os
from pathlib import Path
import pprint
from typing import List
from uuid import UUID, uuid4

from sqlalchemy import text
from src.product_chains.product_combined_info import (
    ProductCombinedInformation,
    describe_image,
    encode_image,
    summarise_product_info,
)
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


from langchain_openai import ChatOpenAI

store_db_session = SessionLocal()

MAXIMUM_GPT_ITERATIONS = 20


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

        if only_products_with_reviews:
            sql += "\nJOIN reviews ON products.data_asin = reviews.product_asin"

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
            if Path(f"./{folder_name}/raw_products.json").exists():
                os.remove(f"./{folder_name}/raw_products.json")

            Path(f"./{folder_name}/raw_products.json").write_text(
                processed_products.model_dump_json()
            )
            logger.success("Product raw data stored into raw_products.json")
        else:
            raise ValueError("Invalid Storage Option")

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

                reviews.append(Review.model_validate(row_item, from_attributes=True))

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
                logger.success("Product raw data stored into raw_products.json")
            else:
                raise ValueError("Invalid Storage Option")


# product = processed_products.products[1]
# prompt_template = f"""
#     You are an expert in writing descriptions for product in a store. You are given the title of a product and a rough information about the product.
#     Your task is to use this information to write a concise description. You should not use any content outside the information provided.
#     Only return back an answer in the pattern provided below: Include product details, and other meta data about the product.

#     Product Title: {product.name}
#     Product Information: {product.description}

#     Generate an answer in this format

#     {{
#         "description": ""
#     }}

# """
