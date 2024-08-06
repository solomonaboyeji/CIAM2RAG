from uuid import UUID
from typing import List, Optional
from datetime import date
from pydantic import BaseModel


class RawReview(BaseModel):
    id: UUID
    review_ref: str
    product_id: UUID
    review_content: str
    review_title: str
    date_written: date
    product_asin: str
    helpful_count: int
    rating_given: int
    review_page_url: str


class RAWProduct(BaseModel):
    id: UUID
    name: str
    description: str
    product_asin: str
    overall_ratings: float
    total_customers_that_rated: int
    price: float
    currency: str
    category: str
    sub_category: str
    product_page_url: str
    image_url: str
    reviews: Optional[List[RawReview]] = []
