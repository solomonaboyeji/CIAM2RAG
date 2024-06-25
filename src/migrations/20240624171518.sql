-- Create enum type "productcategory"
CREATE TYPE "productcategory" AS ENUM ('FASHION', 'SKIN_CARE');
-- Create enum type "productsubcategory"
CREATE TYPE "productsubcategory" AS ENUM ('FASHION_MEN', 'FASHION_WOMEN', 'BEAUTY_SKINCARE');
-- Create "products" table
CREATE TABLE "products" (
  "id" uuid NOT NULL,
  "name" character varying NOT NULL,
  "description" character varying NOT NULL,
  "product_asin" character varying NOT NULL,
  "currency" character varying NOT NULL,
  "overall_ratings" integer NOT NULL,
  "total_customers_that_rated" integer NOT NULL,
  "price" double precision NOT NULL,
  "category" "productcategory" NOT NULL,
  "sub_category" "productsubcategory" NOT NULL,
  "product_page_url" character varying NOT NULL,
  "image_url" character varying NOT NULL,
  "date_created" timestamp NOT NULL,
  "date_modified" timestamp NULL,
  PRIMARY KEY ("id")
);
-- Create index "ix_products_id" to table: "products"
CREATE INDEX "ix_products_id" ON "products" ("id");
-- Create "reviews" table
CREATE TABLE "reviews" (
  "id" uuid NOT NULL,
  "review_ref" character varying NOT NULL,
  "product_id" uuid NOT NULL,
  "review_content" character varying NOT NULL,
  "product_asin" character varying NOT NULL,
  "country" character varying NOT NULL,
  "review_title" character varying NOT NULL,
  "rating_given" integer NOT NULL,
  "helpful_count" integer NOT NULL,
  "date_written" character varying NOT NULL,
  "review_page_url" character varying NOT NULL,
  "date_created" timestamp NOT NULL,
  "date_modified" timestamp NULL,
  PRIMARY KEY ("id"),
  CONSTRAINT "reviews_product_id_fkey" FOREIGN KEY ("product_id") REFERENCES "products" ("id") ON UPDATE NO ACTION ON DELETE CASCADE
);
-- Create index "ix_reviews_id" to table: "reviews"
CREATE INDEX "ix_reviews_id" ON "reviews" ("id");
