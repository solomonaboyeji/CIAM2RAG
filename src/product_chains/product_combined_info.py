from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import base64


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


def encode_image(image_path: str):
    """Returns the base64 string for the image"""

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def describe_image(img_base64, model_name: str, image_format: str = "png"):
    """
    Describes the image provided. The prompt is used to guide the model on what
    to do with the description and what it should do.

    Raises:
        ValueError: If the wrong model name is provided.
        Exception: If any error occur

    Returns:
        str: The description generated for the image
    """

    prompt = f"""
    Describe and summarise the characteristics of the product you are looking at. Start your response with: `The image is a product ... `. 
    In addition, give a short summary of what the product can be used for. Return back plain text no markdown.
    """

    if model_name not in [
        VisionLLM.GPT_4O,
        VisionLLM.LLAVA_VICUNA_Q4_0,
        VisionLLM.LLAVA,
    ]:
        raise ValueError(f"Please provide either {' '.join(VisionLLM)}")

    image_url = f"data:image/{image_format};base64,{img_base64}"

    if model_name.lower() == VisionLLM.GPT_4O.lower():
        chat = ChatOpenAI(model=model_name, max_tokens=1024, temperature=0)
        msg = chat.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ]
                )
            ]
        )
    elif model_name.lower() in [
        VisionLLM.LLAVA.lower(),
        VisionLLM.LLAVA_VICUNA_Q4_0.lower(),
    ]:
        chat = ChatOllama(model=model_name, num_ctx=1024)
        msg = chat.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ]
                )
            ]
        )
    else:
        raise Exception(f"Unsupported model: {model_name}.")

    return msg.content
