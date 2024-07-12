from typing import List, Tuple
import typer
from src.helper import fetch_raw_data, revise_product_descriptions
from src.schemas import CategoryConfigCode, ProductSubCategory
from src.utils import StorageOption

app = typer.Typer()


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
