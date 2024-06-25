-- Modify "products" table
ALTER TABLE "products" ADD COLUMN "image_description" character varying NULL;
-- Modify "reviews" table
ALTER TABLE "reviews" ADD COLUMN "sentiment" character varying NULL;
