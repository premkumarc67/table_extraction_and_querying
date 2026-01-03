import streamlit as st
import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from DB_Engine import engine
from sqlalchemy import inspect

# --- Setup ---
load_dotenv()
api_key = os.getenv("google_api_key")

st.set_page_config(page_title="AI SQL Query Builder", layout="wide")
st.title("🤖 AI Database Assistant")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-2.5-flash")

# --- Session State Init ---
if "df_preview" not in st.session_state:
    st.session_state.df_preview = None

# --- Inputs ---
table_name = st.text_input("Table Name", value="people")

# --- Button 1: Fetch Context ---
if st.button("Fetch Table Context"):
    inspector = inspect(engine)

    if not inspector.has_table(table_name):
        st.error(f"⚠️ Table '{table_name}' does not exist.")
    else:
        with st.spinner("Fetching table context..."):
            query = f"SELECT * FROM {table_name} LIMIT 5"
            st.session_state.df_preview = pd.read_sql(query, engine)

# --- Show Context if Available ---
if st.session_state.df_preview is not None:
    with st.expander("View Table Schema (Context)", expanded=True):
        st.dataframe(st.session_state.df_preview)

    user_query = st.text_area(
        "What would you like to know?",
        value="What are the unique batch numbers?"
    )

    # --- Button 2: Generate & Run ---
    if st.button("Generate & Run Query", type="primary"): # Need to seperate theese two actions
        try:
            with st.spinner("Generating SQL..."):
                prompt = f"""
                You are an expert PostgreSQL assistant. 
                I have a database table named '{table_name}'. 
                Here are the first 5 rows of the table to help you understand the column names and data types:

                {st.session_state.df_preview.to_string(index=False)}

                Based on this schema, write a valid PostgreSQL query to answer the following question:
                "{user_query}"

                Rules:
                1. Output ONLY the raw SQL query. 
                2. Do not use markdown formatting (no ```sql or blockquotes).
                3. Do not add explanations or conversational text.
                4. Always enclose table and column names in double quotes.
                """
                response = model.generate_content(prompt)
                generated_sql = response.text.strip().replace("```sql", "").replace("```", "")

                st.subheader("Generated SQL")
                st.code(generated_sql, language="sql")

            with st.spinner("Executing SQL..."):
                result_df = pd.read_sql(generated_sql, engine)

            with st.expander("View Query Results", expanded=True):
                st.dataframe(result_df)

        except Exception as e:
            st.error(f"Error: {e}")
