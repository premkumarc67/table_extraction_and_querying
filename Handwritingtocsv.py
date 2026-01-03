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

api_key =  os.getenv("google_api_key")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('models/gemini-2.5-flash')

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
st.image(image, use_column_width=True, caption="Uploaded Image") 

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
    response = model.generate_content([prompt, image])
    csv_data = response.text.strip()

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

    df = pd.read_csv(io.StringIO(csv_data)) # Dataframe is created here
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
            column_definitions = ['"id" SERIAL PRIMARY KEY']

            # 2. Generate remaining column definitions dynamically from DF
            for col_name, dtype in df.dtypes.items():
                sql_type = map_pandas_dtype_to_sql(dtype)
                column_definitions.append(f'"{col_name}" {sql_type}')

            # 3. Construct the CREATE TABLE query
            columns_string = ", ".join(column_definitions)
            create_table_query = f"CREATE TABLE {table_name} ({columns_string})"
            
            with engine.connect() as conn:
                # 3. Execute the Create Table query
                conn.execute(text(create_table_query))
                conn.commit() # Ensure table creation is committed
                
                # 4. Append the data
                df.to_sql(
                    table_name,
                    conn,
                    if_exists="append",
                    index=False
                )
                
            st.success(f"Table '{table_name}' created manually and data uploaded.")

        else:
            # Get existing table columns
            table_columns = [
                col["name"]
                for col in inspector.get_columns(table_name)
            ]

            # Drop extra columns from DataFrame
            extra_cols = set(df.columns) - set(table_columns)
            if extra_cols:
                st.warning(f"Dropping extra columns: {extra_cols}")
                df = df.drop(columns=extra_cols)

            # Append data
            df.to_sql(
                table_name,
                engine,
                if_exists="append",
                index=False
            )

            st.success(f"Data appended to existing table '{table_name}'.")

    except Exception as e:
        st.error(f"Error uploading to database: {e}")