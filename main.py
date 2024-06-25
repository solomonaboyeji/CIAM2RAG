import typer
from src.helper import fetch_raw_data, revise_product_descriptions
from src.schemas import CategoryConfigCode, ProductSubCategory

app = typer.Typer()


@app.command()
def fetch_data(
    k: int = typer.Option(20, help="Number of items to fetch"),
    category: CategoryConfigCode = typer.Option(
        CategoryConfigCode.FASHION_MEN, help="Category config reference code"
    ),
):
    """
    Fetch raw data based on the specified parameters.
    """
    config_category_ref_code = getattr(CategoryConfigCode, category.value)
    if category == CategoryConfigCode.ALL.value:
        config_category_ref_code = None

    typer.echo(f"Fetching {k} items from category {category}")
    fetch_raw_data(k=k, config_category_ref_code=config_category_ref_code)
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
