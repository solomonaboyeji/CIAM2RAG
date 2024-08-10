from typing import final
from dotenv import load_dotenv

load_dotenv()

from numpy import vectorize
import typer
from src.helper import (
    fetch_raw_data,
    generate_embeddings,
    generate_pci,
    generate_pid,
    generate_psi,
    revise_product_descriptions,
)
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from src.schemas import (
    CategoryConfigCode,
    EmbeddingModel,
    ProductSubCategory,
    TextLLM,
    VisionLLM,
)
from src.utils import StorageOption

app = typer.Typer()


@app.command()
def generate_multivector_data(
    pid_input_path: str = typer.Option(help="pid.json file"),
    psi_input_path: str = typer.Option(help="psi.json file"),
    raw_data_input_path: str = typer.Option(help="File to the JSON of the raw data."),
    embedding_model: EmbeddingModel = typer.Option(
        EmbeddingModel.NOMIC, help="The embedding model to use."
    ),
    document_store_input_path: str = typer.Option(
        help="The file to store the document's store data"
    ),
    embedding_cache_input_path: str = typer.Option(
        help="The file to store cache the embeddings of the texts"
    ),
):
    """
    Generate PID for each product found the `pid.json` file.
    """

    typer.echo(f"Generating multi vector data with {embedding_model}.")
    try:
        generate_embeddings(
            pid_file_path_str=pid_input_path,
            psi_file_path_str=psi_input_path,
            raw_data_file_path_str=raw_data_input_path,
            document_store_cache_folder=document_store_input_path,
            embedding_cache_folder=embedding_cache_input_path,
        )
        typer.echo(f"PID Generated successfully with {embedding_model}")
    finally:
        wait_for_all_tracers()


@app.command()
def generate_product_image_description(
    k: int = typer.Option(2, help="Number of products to fetch"),
    llm: VisionLLM = typer.Option(VisionLLM.LLAVA_VICUNA_Q4_0, help="The LLM to use."),
    images_directory: str = typer.Option(help="Directory to fetch images from."),
    pci_input_path: str = typer.Option(help="File to read the pci from."),
    pid_output_folder_path: str = typer.Option(
        help="Folder to store the pid.json file"
    ),
):
    """
    Generate PID for each product found the `pid.json` file.
    """

    typer.echo(f"Generating PID with {llm}.")
    try:
        generate_pid(
            k=k,
            pci_json_file_path_str=pci_input_path,
            pid_json_output_folder_path_str=pid_output_folder_path,
            llm_choice=llm,
            images_directory=images_directory,
        )
        typer.echo(f"PID Generated successfully with {llm}")
    finally:
        wait_for_all_tracers()


@app.command()
def generate_product_summary_information(
    k: int = typer.Option(2, help="Number of products to fetch"),
    llm: TextLLM = typer.Option(TextLLM.LLAMA_3_1, help="The LLM to use."),
    pci_input_path: str = typer.Option(help="File to read the pci from."),
    psi_output_folder_path: str = typer.Option(
        help="Folder to store the psi.json file"
    ),
):
    """
    Generate PSI for each product found the `pci.json` file.
    """

    try:
        typer.echo(f"Generating PSI with {llm}.")
        generate_psi(
            k=k,
            pci_json_file_path_str=pci_input_path,
            psi_json_output_folder_path_str=psi_output_folder_path,
            llm_choice=llm,
        )
        typer.echo(f"PSI Generated successfully with {llm}!")
    finally:
        wait_for_all_tracers()


@app.command()
def generate_product_combined_information(
    k: int = typer.Option(2, help="Number of products to fetch"),
    category: CategoryConfigCode = typer.Option(
        CategoryConfigCode.FASHION_MEN, help="Sub Category"
    ),
    folder_path: str = typer.Option(help="Folder to store the pci.json file"),
):
    """
    Generate PCI for each product found in this category.
    """

    try:
        typer.echo(f"Fetching {k} items from category {category}")
        generate_pci(k=k, sub_category=category, folder_path_str=folder_path)
        typer.echo("PCI Generated successfully!")
    finally:
        wait_for_all_tracers()


@app.command()
def fetch_data(
    k: int = typer.Option(20, help="Number of items to fetch"),
    randomise: bool = typer.Option(
        False,
        help="Randomise the k fetched.",
    ),
    with_reviews: bool = typer.Option(
        False,
        help="Only fetch products with reviews",
    ),
    category: CategoryConfigCode = typer.Option(
        CategoryConfigCode.FASHION_MEN, help="Category config reference code"
    ),
    storage_option: StorageOption = typer.Option(
        StorageOption.JSON, help="Where to store the feteched data"
    ),
    folder_name: str = typer.Option("products", help="Folder to store the JSON files"),
):
    """
    Fetch raw data based on the specified parameters.
    """

    typer.echo(f"Fetching {k} items from category {category}")
    fetch_raw_data(
        k=k,
        config_category_ref_code=category,
        storage_option=storage_option,
        randomise=randomise,
        folder_name=folder_name,
        only_products_with_reviews=with_reviews,
    )
    typer.echo("Data fetched successfully!")


@app.command()
def revise_descriptions(
    k: int = typer.Option(2, help="Number of products to fetch"),
    category: ProductSubCategory = typer.Option(
        ProductSubCategory.FASHION_MEN, help="Sub Category"
    ),
):
    """
    Fetch raw data based on the specified parameters.
    """

    typer.echo(f"Fetching {k} items from category {category}")
    revise_product_descriptions(k=k, sub_category=category)
    typer.echo("Descriptions Revised successfully!")


if __name__ == "__main__":
    app()
