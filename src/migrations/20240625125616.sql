-- Modify "products" table
ALTER TABLE "products" DROP COLUMN "product_aspects";
-- Modify "reviews" table
ALTER TABLE "reviews" ADD COLUMN "product_aspects" character varying NULL;
