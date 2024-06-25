-- Modify "products" table
ALTER TABLE "products" ALTER COLUMN "overall_ratings" TYPE double precision, ADD COLUMN "product_aspects" character varying NULL;
-- Modify "reviews" table
ALTER TABLE "reviews" ALTER COLUMN "rating_given" TYPE double precision;
