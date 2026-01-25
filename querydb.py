import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from DB_Engine import engine
from sqlalchemy import inspect
from llama_cpp import Llama # type: ignore
import sys
import plotly.express as px # type: ignore

# --- Setup ---
load_dotenv()

MODEL_PATH = "sqlcoder-7b.Q4_K_M.gguf"

try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,      # Context window size
        n_gpu_layers=0,  # Set to 0 for CPU only. Set to -1 if you have a GPU set up.
        verbose=False    # Suppress internal logs
    )
    print("Model loaded successfully.\n")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

with open("prompt.txt", "r", encoding="utf-8") as f:
    SCHEMA_CONTEXT = f.read()

st.set_page_config(page_title="AI SQL Query Builder", layout="wide")
st.title("🤖 AI Database Assistant")

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
                prompt = f"""### Task
                            Generate a SQL query to answer [QUESTION]{user_query}[/QUESTION]

                            ### Database Schema
                            The query will run on a database with the following schema:
                            {SCHEMA_CONTEXT}

                            ### SQL
                            Given the database schema, here is the SQL query that answers [QUESTION]{user_query}[/QUESTION]
                            [SQL]
                            """
                output = llm(
                    prompt, 
                    max_tokens=200, 
                    stop=["```", ";"], # Stop generation when query ends
                    echo=False
                )
                generated_sql = output["choices"][0]["text"].strip().replace("```sql", "").replace("```", "")

                st.subheader("Generated SQL")
                st.code(generated_sql, language="sql")

            with st.spinner("Executing SQL..."):
                result_df = pd.read_sql(generated_sql, engine)

            with st.expander("View Query Results", expanded=True):
                st.dataframe(result_df)

        except Exception as e:
            st.error(f"Error: {e}")
