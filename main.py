from dotenv import load_dotenv

load_dotenv()

import typer
from src.helper import (
    fetch_raw_data,
    generate_pci,
    generate_psi,
    revise_product_descriptions,
)
from src.schemas import CategoryConfigCode, ProductSubCategory, TextLLM
from src.utils import StorageOption

app = typer.Typer()


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

    typer.echo(f"Generating PSI.")
    generate_psi(
        k=k,
        pci_json_file_path_str=pci_input_path,
        psi_json_output_folder_path_str=psi_output_folder_path,
        llm_choice=llm,
    )
    typer.echo("PSI Generated successfully!")


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

    typer.echo(f"Fetching {k} items from category {category}")
    generate_pci(k=k, sub_category=category, folder_path_str=folder_path)
    typer.echo("PCI Generated successfully!")


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
