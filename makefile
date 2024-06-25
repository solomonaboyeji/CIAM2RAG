
# This should be ran outside the finco-api container
run-dev-migrations:
	export PYTHONPATH=. && atlas migrate diff --env run-migrations \
	--var database_password=CHANGE_ME \
	--var database_host=localhost \
	--var database_name=CHANGE_ME \
	--var database_username=CHANGE_ME \
	--var database_port=5432 \
	--var models_path=src/models

# This should be ran outside the finco-api container
run-test-migrations:
	export PYTHONPATH=. && atlas migrate diff --env run-migrations \
	--var database_password=CIAM2RAG-SECRET-PASSWORD \
	--var database_host=localhost \
	--var database_name=CIAM2RAG-DB \
	--var database_username=CIAM2RAG-USER \
	--var database_port=5432 \
	--var models_path=src/models

apply-migrations:
	export PYTHONPATH=. && atlas migrate apply --env local \
	--var database_password=CIAM2RAG-SECRET-PASSWORD \
	--var database_host=localhost \
	--var database_name=CIAM2RAG-DB \
	--var database_username=CIAM2RAG-USER \
	--var database_port=5432 \
	--var models_path=src/models


debug-migration-error:
	export PYTHONPATH=. && atlas-provider-sqlalchemy --path src/models  --dialect postgresql

run-test-database:
	-docker stop ciam2rag-store-db
	-docker rm ciam2rag-store-db -f

	docker run \
		--rm --name ciam2rag-store-db \
		--network ciam2rag-network \
		-p 5432:5432 \
		-e POSTGRES_USER=CIAM2RAG-USER \
		-e POSTGRES_PASSWORD=CIAM2RAG-SECRET-PASSWORD \
		-e POSTGRES_DB=CIAM2RAG-DB \
		-d postgres:16.2
	
	chmod +x scripts/wait_for_test_db.sh

	sh scripts/wait_for_test_db.sh

	make debug-migration-error
	make run-test-migrations
	make apply-migrations


# spin up raw database
run-raw-db:
	-mkdir ciam2rag_postgres_data
	docker compose -f docker/docker-compose-raw.yml up -d
	
log-raw-db:
	docker compose -f docker/docker-compose-raw.yml logs -f

down-raw-db:
	docker compose -f docker/docker-compose-raw.yml down

# load the raw data into the raw database
load-raw-data:
	docker cp ../scraped_data.sql ciam2rag-raw-data:/tmp/scraped_data.sql
	docker exec -it ciam2rag-raw-data bash
	psql -U amzon -d CIAM2RAG-DB < /tmp/scraped_data.sql


# -----------------
# Store Database
run-store-db:
	-mkdir ciam2rag_store_postgres_data
	docker compose -f docker/docker-compose.yml up -d
# -----------------

create-network:
	docker network create ciam2rag-network


# -----------------
# Ollama
start-llama3:
	ollama run llama3

start-zephyr:
	ollama run zephyr

start-llava:
	ollama run llava

start-mistral:
	ollama run mistral

start-phil:
	ollama run phi3:medium