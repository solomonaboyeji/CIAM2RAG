from src.config import (
    DATABASE_NAME,
    DATABASE_PASSWORD,
    DATABASE_PORT,
    DATABASE_SERVER,
    DATABASE_USERNAME,
    MAX_DB_POOL_SIZE,
    MIN_DB_POOL_SIZE,
    RAW_DATA_DATABASE_USERNAME,
    RAW_DATA_DATABASE_PASSWORD,
    RAW_DATA_DATABASE_SERVER,
    RAW_DATA_DATABASE_PORT,
    RAW_DATA_DATABASE_NAME,
)


# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker


# engine = create_engine(DB_CONN_STRING)
engine = create_engine(
    f"postgresql://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_SERVER}:{DATABASE_PORT}/{DATABASE_NAME}",
    pool_size=MIN_DB_POOL_SIZE,
    max_overflow=MAX_DB_POOL_SIZE,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Dependency to get the database session
def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# RAW DATA database
# engine = create_engine(DB_CONN_STRING)
raw_db_engine = create_engine(
    f"postgresql://{RAW_DATA_DATABASE_USERNAME}:{RAW_DATA_DATABASE_PASSWORD}@{RAW_DATA_DATABASE_SERVER}:{RAW_DATA_DATABASE_PORT}/{RAW_DATA_DATABASE_NAME}",
    pool_size=MIN_DB_POOL_SIZE,
    max_overflow=MAX_DB_POOL_SIZE,
)
RawDBSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=raw_db_engine)


# Dependency to get the database session
def get_raw_data_db_session():
    db = RawDBSessionLocal()
    try:
        yield db
    finally:
        db.close()
