import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from dotenv import load_dotenv
import os

load_dotenv() # Load environment variables from .env file

# Credentials
USER = "postgres"
PASSWORD = os.getenv("DB_PASSWORD")   
HOST = os.getenv("DB_HOST")
PORT = 5432
DBNAME = "postgres"

TABLE_NAME = "people"
CSV_FILE_PATH = "stabiliser_data.csv"

# URL encode password, in case it has special characters
ENCODED_PASSWORD = quote_plus(PASSWORD)

# Create engine
engine = create_engine(
    f"postgresql+psycopg2://{USER}:{ENCODED_PASSWORD}@{HOST}:{PORT}/{DBNAME}"
)