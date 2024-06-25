import os
import dotenv

dotenv.load_dotenv()

MIN_DB_POOL_SIZE = 1
MAX_DB_POOL_SIZE = int(os.getenv("MAX_DB_POOL_SIZE", "10"))

# Connection to a PostgreSQL database
DATABASE_USERNAME = os.getenv("DATABASE_USERNAME")
DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD")
DATABASE_NAME = os.getenv("DATABASE_NAME")
DATABASE_SERVER = os.getenv("DATABASE_SERVER")
DATABASE_PORT = os.getenv("DATABASE_PORT", "1433")


RAW_DATA_DATABASE_USERNAME = os.getenv("RAW_DATA_DATABASE_USERNAME")
RAW_DATA_DATABASE_PASSWORD = os.getenv("RAW_DATA_DATABASE_PASSWORD")
RAW_DATA_DATABASE_SERVER = os.getenv("RAW_DATA_DATABASE_SERVER")
RAW_DATA_DATABASE_PORT = os.getenv("RAW_DATA_DATABASE_PORT")
RAW_DATA_DATABASE_NAME = os.getenv("RAW_DATA_DATABASE_NAME")
