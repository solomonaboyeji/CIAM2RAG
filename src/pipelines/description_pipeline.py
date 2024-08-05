import json
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import (
    OllamaGenerator,
)

from src.models.models import ProductModel


def generate_concise_description(
    product: ProductModel, model_name: str = "zephyr", temperature: str = "0.9"
):

    template = """

    You are an expert marketing genius. Your task is to review the description of a product.
    You will remove unncessary characters in the description, while you return the updated description.
    You should not add any information from any other source, all your content should be within the provided 
    information.

    When you have an answer, review it to ensure it is readable by humans.


    Here is the product information:
    Product Name: {{product_name}}
    Product Data
    {{product_information}}

    Your revised description:

    """

    prompt_builder = PromptBuilder(template=template)
    llm = OllamaGenerator(model=model_name)

    pipeline = Pipeline()
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("llm", llm)

    pipeline.connect("prompt_builder", "llm")

    result = pipeline.run(
        {
            "prompt_builder": {
                "product_name": product.name,
                "product_information": product.description,
            }
        }
    )

    output = result["llm"]["replies"][0]

    return output
