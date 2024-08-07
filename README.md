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

Generate Combined Product Information

```sh
python -m main generate-product-combined-information --k -1 --category "FASHION_MEN"  --folder-path V1_DATA
```

Generate Product Summary Information

```sh
 python -m main generate-product-summary-information --llm llama3.1 --pci-input-path V1_DATA/pci.json --psi-output-folder-path V1_DATA --k 20

 python -m main generate-product-summary-information  --pci-input-path V1_DATA/pci.json --psi-output-folder-path V1_DATA --k 20 --llm llama3.1:8b-instruct-q5_K_M

 This started from 2024-08-06 13:54:11.247 and ended 2024-08-06 15:39:36.439 as shown in `./research_data/generating-psi`


```
