from uuid import UUID, uuid4

from sqlalchemy import text
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
)
from bs4 import BeautifulSoup
from loguru import logger
from src.utils import filter_review_date

store_db_session = SessionLocal()


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
    k: int | None = None, config_category_ref_code: CategoryConfigCode | None = None
):

    raw_db_session = RawDBSessionLocal()
    sql = """
        SELECT * FROM products
    """
    where_sql = ""

    if config_category_ref_code == None:
        categories_ref = [
            CategoryConfigCode.FASHION_MEN.value,
            CategoryConfigCode.FASHION_WOMEN.value,
            CategoryConfigCode.BEAUTY_SKIN_CARE.value,
        ]
    else:
        categories_ref = [config_category_ref_code]

    where_sql += f"products.config_category_ref_code in {tuple(categories_ref)}"

    if len(where_sql) > 0:
        sql += "WHERE " + where_sql

    sql += "\n ORDER BY products.config_category_ref_code"

    if k is not None:
        sql += f" \n LIMIT {k} \n"

    result = raw_db_session.execute(text(sql))

    products = []
    for row in result.all():
        row_item = {}

        for key, value in enumerate(row._fields):

            if value != "image_data":
                content = row[key]
                if value == "description":
                    soup = BeautifulSoup(row[key], features="html.parser")
                    content = " ".join(soup.text.split("\n"))

                row_item[value] = content

        products.append(Product.model_validate(row_item, from_attributes=True))

    processed_products = ProductList.model_validate({"products": products})

    result = map(insert_store_product, processed_products.products)

    for product in list(result):
        sql = f"SELECT * FROM reviews WHERE product_asin = '{product.product_asin}'"
        result = raw_db_session.execute(text(sql))
        reviews = []
        for row in result.all():
            row_item = {"product_id": product.id, "id": uuid4()}

            for key, value in enumerate(row._fields):
                content = row[key]
                if value == "review_date":
                    content = filter_review_date(content)

                row_item[value] = content

            reviews.append(Review.model_validate(row_item, from_attributes=True))

        product_reviews = ReviewList.model_validate({"reviews": reviews})
        logger.success(
            f"Product: {product.id} - Reviews: {len(product_reviews.reviews)}"
        )
        result = list(map(insert_store_review, product_reviews.reviews))

    raw_db_session.close()


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
