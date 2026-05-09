import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from DB_Engine import engine
from sqlalchemy import inspect
from llama_cpp import Llama # type: ignore
import sys
import plotly.express as px # type: ignore
from pygwalker.api.streamlit import StreamlitRenderer # type: ignore

# --- Setup ---
load_dotenv()

MODEL_PATH = "google_gemma-3-1b-it-Q6_K.gguf" 

try:
    with st.spinner("Loading Gemma 3 model into memory..."):
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,      # Context window size 
            n_gpu_layers=-1, # Set to -1 to offload to GPU if available, or 0 for CPU only.
            verbose=False    # Suppress internal logs
        )
        print("Model loaded successfully.\n")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Get schema context
with open("prompt.txt", "r", encoding="utf-8") as f:
    SCHEMA_CONTEXT = f.read()

st.set_page_config(page_title="AI SQL Query Builder", layout="wide")
st.title("🤖 AI Database Assistant")

# --- Session State Init ---
if "df_preview" not in st.session_state:
    st.session_state.df_preview = None

# --- Inputs ---
table_name = st.text_input("Table Name", value="emp")

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
        value="What are the unique ages available?"
    )

# --- Button 2: Generate & Run ---
    if st.button("Generate & Run Query", type="primary"):
        try:
            with st.spinner("Generating SQL..."):
                # Use the Chat Completion API for Gemma models
                messages = [
                    {
                        "role": "system", 
                        "content": "You are an expert SQL assistant. Generate a valid SQL query to answer the user's question based on the provided schema. Output ONLY the SQL query. Do not include explanations or conversational text."
                    },
                    {
                        "role": "user", 
                        "content": f"### Database Schema\n{SCHEMA_CONTEXT}\n\n### Question\n{user_query}"
                    }
                ]
                
                output = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=250,
                    stop=[";"] # Removed "```" so it doesn't cut off immediately
                )
                print("Raw model output:", output)  # Debugging line to see the full response structure
                # Extract the message content instead of raw text
                generated_text = output["choices"][0]["message"]["content"].strip()
                
                # Clean up any potential markdown formatting from Gemma
                generated_sql = generated_text.replace("```sql", "").replace("```", "").strip()

                st.subheader("Generated SQL")
                st.code(generated_sql, language="sql")

            with st.spinner("Executing SQL..."):
                result_df = pd.read_sql(generated_sql, engine)

            with st.expander("View Query Results", expanded=True):
                st.dataframe(result_df)
                
                st.subheader("Explore Data")
                
                @st.cache_resource
                def get_pyg_renderer(df):
                    return StreamlitRenderer(df, spec_io_mode="RW")
                    
                renderer = get_pyg_renderer(result_df)
                renderer.explorer()

        except Exception as e:
            st.error(f"Error: {e}")