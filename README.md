# CIAM

## Setup

```sh
make run-raw-db
```

Ensure you have the scraped_data.sql in a folder before the repository

```sh
make load-raw-data
psql -U amzon -d CIAM2RAG-DB < /tmp/scraped_data.sql
```

Apply migrations to the store-db

```sh
make apply-migrations
```

Download all the products in a particular category

```sh
python -m main fetch-data --storage-option JSON --with-reviews --folder-name V1_DATA --k -1
```
