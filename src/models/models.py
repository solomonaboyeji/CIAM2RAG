from typing import Optional
import uuid
from datetime import datetime

from datetime import UTC
from sqlalchemy import Boolean, DateTime, ForeignKey, Enum
from sqlalchemy.orm import mapped_column, Mapped
from sqlalchemy.dialects.postgresql import UUID


from src.schemas import ProductCategory, ProductSubCategory
from src.utils import Base


class ProductModel(Base):
    __tablename__ = "products"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, index=True, default=uuid.uuid4
    )

    name: Mapped[str]
    description: Mapped[str]
    revised_description: Mapped[Optional[str]]
    product_asin: Mapped[str]
    currency: Mapped[str]
    overall_ratings: Mapped[float]
    total_customers_that_rated: Mapped[int]
    price: Mapped[float]

    category: Mapped[ProductCategory] = mapped_column(Enum(ProductCategory))
    sub_category: Mapped[ProductSubCategory] = mapped_column(Enum(ProductSubCategory))

    product_page_url: Mapped[str]
    image_url: Mapped[str]

    image_description: Mapped[Optional[str]]

    # GENERAL ATTRIBUTES
    date_created: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=datetime.now(UTC)
    )
    date_modified: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=True, onupdate=datetime.now(UTC)
    )


class ReviewModel(Base):
    __tablename__ = "reviews"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, index=True, default=uuid.uuid4
    )

    # review_id from raw data
    review_ref: Mapped[str]
    product_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("products.id", ondelete="CASCADE")
    )
    review_content: Mapped[str]
    product_asin: Mapped[str]
    country: Mapped[str]
    review_title: Mapped[str]
    rating_given: Mapped[float]
    helpful_count: Mapped[int]
    date_written: Mapped[str]
    review_page_url: Mapped[str]

    sentiment: Mapped[Optional[str]]
    product_aspects: Mapped[Optional[str]]

    # GENERAL ATTRIBUTES
    date_created: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=datetime.now(UTC)
    )
    date_modified: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=True, onupdate=datetime.now(UTC)
    )
