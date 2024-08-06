from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough


from src.schemas import TextLLM, VisionLLM
from src.product_chains.schemas import RAWProduct


class ProductCombinedInformation:

    product: RAWProduct

    def __init__(self, product: RAWProduct) -> None:
        self.product = product

    def get_document(self):
        return Document(
            page_content=self.info(),
            id=str(self.product.id),
            metadata={
                "product_id": self.product.id,
                "product_asin": self.product.product_asin,
                "image_url": self.product.image_url,
                "name": self.product.name,
            },
        )

    def info(self):
        return (
            f"Product Name: {str(self.product.name).strip()} \n"
            f"Product Description: {str(self.product.description).strip()} \n"
            f"Product ID: {str(self.product.id).strip()} \n"
            f"Product Asin: {self.product.product_asin} \n"
            f"Overall Ratings {self.product.overall_ratings} \n"
            f"Total Customers that rated: {self.product.total_customers_that_rated} \n"
            f"Price: {self.product.currency}{self.product.price} \n"
        )

    def info_for_summary(self):
        return (
            f"Product Name: {str(self.product.name).strip()} \n"
            f"Product Description: {str(self.product.description).strip()} \n"
        )

    def image_path(self):
        raise ValueError("Not yet implemented!")
        return f"{product_images_path}/{self.product.product_asin}.png"


def summarise_product_info(product_info: str, model_name: str):

    if model_name.lower() not in [
        TextLLM.GPT_4O.lower(),
        TextLLM.LLAMA_3_1.lower(),
        TextLLM.LLAMA_3_1_INSTRUCT.lower(),
        TextLLM.MISTRAL.lower(),
        VisionLLM.GPT_4O.lower(),
        VisionLLM.LLAVA.lower(),
        VisionLLM.LLAVA_VICUNA_Q4_0.lower(),
    ]:
        raise ValueError(
            f"Please provide a supported model. {model_name} not supported"
        )

    prompt_template = (
        "Your response should be plain string not markdown or json and start with the product name."
        "You are an assistant tasked with summarizing "
        "text for retrieval. These summaries will be embedded and used "
        "to retrieve the raw text. Give a concise summary of the product. "
        "Ensure you include important details such as what is made of and it can be used for, these summary should be"
        "well optimized for retrieval.\n"
        f"Product: {product_info}\n"
    )

    prompt = ChatPromptTemplate.from_template(prompt_template)

    if model_name.lower() in [TextLLM.GPT_4O, VisionLLM.GPT_4O]:
        model = ChatOpenAI(model=model_name, max_tokens=1024, temperature=0)
    else:
        model = ChatOllama(model=model_name, temperature=0, num_ctx=4000)

    # { "product_info": RunnablePassthrough() } is same as lambda x: x
    # basically do not modify what the user entered, pass into the key product_info
    summarise_chain = (
        {"product_info": RunnablePassthrough()} | prompt | model | StrOutputParser()
    )
    summary = summarise_chain.invoke(product_info)
    return summary
