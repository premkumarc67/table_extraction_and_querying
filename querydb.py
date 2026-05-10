import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from DB_Engine import engine
from sqlalchemy import inspect
from llama_cpp import Llama # type: ignore
import sys
import os
import chromadb # type: ignore

# --- Setup ---
load_dotenv()
MODEL_PATH = "google_gemma-3-1b-it-Q6_K.gguf" 

@st.cache_resource
def load_model():
    try:
        return Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False
        )
    except Exception as e:
        st.error(f"Error loading model: {e}")
        sys.exit(1)

llm = load_model()

# --- Initialize ChromaDB ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="table_schema_metadata")

st.set_page_config(page_title="AI SQL Query Builder", layout="wide")
st.title("🤖 AI Database Assistant")

# --- Session State Init ---
def initialize_session_state():
    if "step" not in st.session_state:
        st.session_state.step = "ask_knowledge"
    if "table_input" not in st.session_state:
        st.session_state.table_input = ""
    if "generated_sql" not in st.session_state:
        st.session_state.generated_sql = ""
    if "retrieved_context" not in st.session_state:
        st.session_state.retrieved_context = ""
    if "knows_table" not in st.session_state:
        st.session_state.knows_table = False

initialize_session_state()

# --- Phase 0: Knowledge Check ---
if st.session_state.step == "ask_knowledge":
    st.subheader("Do you know which tables are involved in your query?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, I know the tables", use_container_width=True):
            st.session_state.knows_table = True
            st.session_state.step = "get_tables"
            st.rerun()
    with col2:
        if st.button("No, help me find them", use_container_width=True):
            st.session_state.knows_table = False
            st.session_state.step = "get_query"
            st.rerun()

# --- Phase 1a: Ask for Table Names (If Known) ---
if st.session_state.step == "get_tables":
    if st.button("← Back"):
        st.session_state.step = "ask_knowledge"
        st.rerun()

    st.subheader("Which tables do you want to query?")
    table_input = st.text_input(
        "Enter Table Names (comma separated)", 
        value=st.session_state.table_input, 
        placeholder="e.g., emp, dept"
    )
    
    if st.button("Next step", type="primary"):
        if not table_input.strip():
            st.warning("⚠️ Please enter at least one table name, or go back and choose 'No'.")
        # Check if the table exists in the database using SQLAlchemy inspector
        else:
            requested_tables = [t.strip() for t in table_input.split(",")]
            inspector = inspect(engine)
            actual_tables = inspector.get_table_names()
            invalid_tables = [t for t in requested_tables if t not in actual_tables]
            
            if invalid_tables:
                st.error(f"⚠️ The following tables were not found in the database: {', '.join(invalid_tables)}. Please check your spelling.")
            else:
                st.session_state.table_input = table_input
                st.session_state.step = "get_query"
                st.rerun()

# --- Phase 1b: Ask for the Query ---
if st.session_state.step == "get_query":
    if st.button("← Back"):
        # Route back appropriately based on their first choice
        st.session_state.step = "get_tables" if st.session_state.knows_table else "ask_knowledge"
        st.rerun()

    st.subheader("What would you like to know?")
    if st.session_state.knows_table and st.session_state.table_input:
        st.info(f"Working with **{st.session_state.table_input}**")

    with st.form("query_form"):
        user_query = st.text_area("Enter your question:", placeholder="e.g., Show me all employees in sales")
        submit = st.form_submit_button("Generate Query", type="primary")

    if submit:
        if not user_query.strip():
            st.warning("⚠️ Please enter a question to generate a query.")
        else:
            with st.spinner("Interpreting intent • Retrieving schema • Generating SQL 🧠"):
                # RETRIEVAL LOGIC
                if st.session_state.knows_table and st.session_state.table_input:
                    # Direct Retrieval using collection.get
                    table_list = [t.strip() for t in st.session_state.table_input.split(",")]
                    results = collection.get(ids=table_list)
                    st.session_state.retrieved_context = "\n\n".join(results['documents']) if results and results.get('documents') else "Tables not found in metadata."
                else:
                    # Semantic Search using collection.query
                    results = collection.query(query_texts=[user_query], n_results=2)
                    st.session_state.retrieved_context = "\n\n".join(results['documents'][0]) if results and results.get('documents') else "No relevant schema found."

                system_prompt = (
                    "You are a Senior PostgreSQL Developer. Your task is to convert natural language into valid SQL.\n"
                    "RULES:\n"
                    "1. Use ONLY the provided table schemas.\n"
                    "2. Output ONLY raw SQL. No markdown, no triple backticks, no explanations.\n"
                    "3. If the schema is insufficient, return a comment: -- Insufficient schema data."
                )
                
                prompt_content = (
                    f"### DATABASE SCHEMA METADATA:\n{st.session_state.retrieved_context}\n\n"
                    f"### USER QUERY:\n{user_query}\n\n"
                    f"### SQL QUERY:"
                )

                output = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_content}
                    ],
                    max_tokens=300,
                    stop=[";"]
                )
                
                sql = output["choices"][0]["message"]["content"].strip()
                st.session_state.generated_sql = sql.replace("```sql", "").replace("```", "").strip()

if st.session_state.generated_sql:
    st.subheader("Edit, Optimize & Execute")
    
    edited_sql = st.text_area("Refine Generated SQL ✨", value=st.session_state.generated_sql, height=150)
    st.session_state.generated_sql = edited_sql

    if st.button("Run Query", type="secondary"):
        try:
            result_df = pd.read_sql(st.session_state.generated_sql, engine)
            st.dataframe(result_df)
        except Exception as e:
            st.error(f"Database error: {e}")

    with st.expander("View RAG Context Used"):
        st.info(st.session_state.retrieved_context)