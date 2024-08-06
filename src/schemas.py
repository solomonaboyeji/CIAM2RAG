from datetime import date, datetime
import enum
from typing import List, Optional
from uuid import UUID
from pydantic import AfterValidator, BaseModel, BeforeValidator, Field, validator
from typing_extensions import Annotated

from src.utils import filter_helpful_vote, filter_review_location, filter_review_rating


class TextLLM(enum.StrEnum):
    GPT_4O = "gpt-4o"
    LLAMA_3_1 = "llama3.1"
    LLAMA_3_1_INSTRUCT = "llama3.1:8b-instruct-q5_K_M"
    MISTRAL = "mistral"


class VisionLLM(enum.StrEnum):
    GPT_4O = "gpt-4o"
    LLAVA = "llava"
    LLAVA_VICUNA_Q4_0 = "llava:13b-v1.6-vicuna-q4_0".lower()


class ProductSubCategory(enum.StrEnum):
    FASHION_MEN = "Men's Fashion"
    FASHION_WOMEN = "Women's Fashion"
    BEAUTY_SKINCARE = "Skin Care"


class ProductCategory(enum.StrEnum):
    FASHION = "Fashion"
    BEAUTY = "Beauty"


class CategoryConfigCode(enum.StrEnum):
    BEAUTY_SKIN_CARE = "BEAUTY_SKIN_CARE"
    FASHION_MEN = "FASHION_MEN"
    FASHION_WOMEN = "FASHION_WOMEN"
    ALL = "ALL"


def parse_date(value):
    if isinstance(value, str):
        try:
            return datetime.strptime(value, "%d %B %Y")
        except ValueError:
            raise ValueError('Invalid date format. Use "DD Month YYYY"')
    return value


class Review(BaseModel):
    id: UUID
    review_ref: str = Field(alias="review_id")
    product_id: UUID
    review_content: str
    review_title: Optional[str] = ""
    date_written: Annotated[date, BeforeValidator(parse_date)] = Field(
        alias="review_date"
    )
    product_asin: str
    country: Annotated[str, BeforeValidator(filter_review_location)] = Field(
        alias="review_location"
    )
    helpful_count: Annotated[
        int, BeforeValidator(lambda x: 0 if not x else filter_helpful_vote(x))
    ] = Field(0, alias="helpful_vote")
    rating_given: Annotated[float, BeforeValidator(filter_review_rating)] = Field(
        alias="review_rating"
    )
    review_page_url: str = Field(alias="page_url")
    sentiment: Optional[str] = None
    product_aspects: Optional[str] = None


class ReviewUpdate(BaseModel):
    id: UUID
    review_content: Optional[str]
    sentiment: Optional[str]
    product_aspects: Optional[str]


class ProductUpdate(BaseModel):
    id: UUID
    revised_description: Optional[str]
    image_description: Optional[str]


class Product(BaseModel):
    id: UUID = Field(alias="data_uuid")
    name: str
    description: str
    revised_description: Optional[str] = None
    product_asin: str = Field(alias="data_asin")

    overall_ratings: Annotated[
        float, BeforeValidator(lambda x: "0.0" if not x else x)
    ] = Field(alias="ratings")

    total_customers_that_rated: Annotated[
        int, BeforeValidator(lambda x: 0 if not x else x)
    ] = Field(alias="total_customer_that_rated")

    price: Annotated[float, BeforeValidator(lambda x: "0.0" if not x else x)]

    currency: Annotated[str, BeforeValidator(lambda x: "0" if not x else x)]

    category: ProductCategory
    sub_category: ProductSubCategory
    product_page_url: str
    image_url: str = Field(alias="img_url")


class ReviewList(BaseModel):
    reviews: List[Review]


class ProductList(BaseModel):
    products: List[Product]
