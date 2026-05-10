import streamlit as st
import pandas as pd
import io
from PIL import Image
import google.generativeai as genai
from DB_Engine import engine
from dotenv import load_dotenv
import os
from sqlalchemy import inspect
from sqlalchemy import text
import numpy as np
import chromadb #type: ignore

load_dotenv() # Load environment variables from .env file

# Helper function to map pandas dtypes to SQL types
def map_pandas_dtype_to_sql(dtype):
    dtype_str = str(dtype)
    if 'int' in dtype_str:
        return 'INTEGER'
    elif 'float' in dtype_str:
        return 'FLOAT'
    elif 'datetime' in dtype_str:
        return 'TIMESTAMP'
    elif 'bool' in dtype_str:
        return 'BOOLEAN'
    else:
        return 'TEXT' # Default fallback for strings/objects

# RAG Setup for Schema Metadata Storage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
client = chromadb.PersistentClient(path=CHROMA_PATH) # Stores data in the disk
print("Chroma path:", CHROMA_PATH)

collection = client.get_or_create_collection(
    name="table_schema_metadata"
)
def generate_and_store_schema_metadata(table_name, df, model): # Used only for new tables.

    columns_info = ["id: SERIAL PRIMARY KEY (auto-incrementing unique identifier)"]

    for col_name, dtype in df.dtypes.items():
        columns_info.append(
            f"{col_name}: {map_pandas_dtype_to_sql(dtype)}"
        )

    columns_info = "\n".join(columns_info)

    sample_df = df.head(5)

    prompt = f"""
    You are a database documentation assistant.

    Generate semantic metadata for this table.

    Table Name:
    {table_name}

    Columns:
    {columns_info}

    Sample Data:
    {sample_df.to_dict(orient='records')}

    Output Format:

    Table Name:
    ...

    Table Description:
    ...

    Columns:
    - column_name:
      type:
      description:
      sample_values:
    """

    response = model.generate_content(prompt)

    schema_metadata = response.text
    # For now using default embedding model
    collection.add(
        documents=[schema_metadata],
        metadatas=[{
            "table_name": table_name
        }],
        ids=[table_name]
    )

    return schema_metadata

api_key =  os.getenv("google_api_key")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('models/gemini-2.5-flash') # Initialize the Gemini 2.5 Flash model

# --- UI LAYOUT ---
st.title("📝 Handwritten Table to CSV Converter")

uploaded_file = st.file_uploader(
    "Upload Handwritten Table Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is None:
    st.info("Please upload an image to continue.")
    st.stop()

# 1. Display the image
image = Image.open(uploaded_file)
st.image(image, width="stretch", caption="Uploaded Image") 

# 2. Process Button
submit = st.button("Extract the Data")

# If the button is clicked, fetch data and store in session_state
if submit:
    prompt = """
    Analyze this image of a handwritten table. 
    Extract the data into a clean CSV format.
    Rules:
    1. Output ONLY the raw CSV text. Do not include markdown formatting.
    """
    
    # API CALL
    response = model.generate_content([prompt, image])   # Returns response in json format
    csv_data = response.text.strip() # String containing CSV-formatted data

    # Clean up markdown if present
    if csv_data.startswith("```"):
        csv_data = csv_data.replace("```csv", "").replace("```", "").strip()
    
    # STORE IN SESSION STATE, this ensures the data persists when you click other buttons
    st.session_state['csv_data'] = csv_data

upload_clicked = False

# CHECK IF DATA EXISTS IN SESSION STATE
if 'csv_data' in st.session_state:
    csv_data = st.session_state['csv_data'] # Retrieve stored data
    
    st.success("Data Extraction Complete!")
    st.subheader("Preview Data")
    # read_csv expects a file-like object, so we use StringIO to convert the string to a file-like object
    df = pd.read_csv(io.StringIO(csv_data)) # Dataframe is created here
    df.columns = (df.columns.str.strip().str.lower().str.replace(" ", "_")) # Preprocessing Col names
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="converted_batch_data.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        # Capture the click event in a variable
        upload_clicked = st.button("Upload to Database", use_container_width=True)

if upload_clicked:
    st.session_state["show_table_input"] = True

if st.session_state.get("show_table_input", False):
    st.subheader("Database Upload")

    table_name = st.text_input(
        "Enter table name",
        placeholder="e.g. people"
    )

    submit_upload = st.button("Submit & Upload")

if st.session_state.get("show_table_input") and submit_upload:
    try:
        if not table_name:
            st.warning("Please enter a table name.")
            st.stop()

        inspector = inspect(engine)

        if not inspector.has_table(table_name):
            # 1. Generate column definitions dynamically
            column_definitions = ['id SERIAL PRIMARY KEY']

            # 2. Generate remaining column definitions dynamically from DF
            for col_name, dtype in df.dtypes.items():
                sql_type = map_pandas_dtype_to_sql(dtype)
                column_definitions.append(f'{col_name} {sql_type}')

            # 3. Construct the CREATE TABLE query
            columns_string = ", ".join(column_definitions)
            create_table_query = f"CREATE TABLE {table_name} ({columns_string})"
            
            with open("prompt.txt", "a", encoding="utf-8") as f:
                f.write(create_table_query.strip() + ";\n\n")


            with engine.connect() as conn:
                # 3. Execute the Create Table query
                conn.execute(text(create_table_query))
                conn.commit() # Ensure table creation is committed
                
                # 4. Append the data. DataFrame offers direct upload to Database
                df.to_sql(
                    table_name,
                    conn,
                    if_exists="append",
                    index=False
                )
                
            st.success(f"Table '{table_name}' created manually and data uploaded.")

            # 5. Generate and store schema metadata in ChromaDB for RAG retrieval 
            generate_and_store_schema_metadata(table_name,df,model) 

        else:
            # Get existing table columns
            table_columns = [
                col["name"]
                for col in inspector.get_columns(table_name)
            ]

            # Drop extra columns from DataFrame. 
            extra_cols = set(df.columns) - set(table_columns)
            if extra_cols:
                st.warning(f"Dropping extra columns: {extra_cols}")
                df = df.drop(columns=extra_cols)

            # Append data. 
            # Pandas generates an INSERT query only for the columns present in the DataFrame. 
            # This allows us to append data even if the Dtaframe has fewer columns than the Table.
            df.to_sql(
                table_name,
                engine,
                if_exists="append",
                index=False
            )

            st.success(f"Data appended to existing table '{table_name}'.")

    except Exception as e:
        st.error(f"Error uploading to database: {e}")