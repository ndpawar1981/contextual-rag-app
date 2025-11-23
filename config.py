# config.py
import os

# You can also use python-dotenv if you like:
# from dotenv import load_dotenv
# load_dotenv()

# ---- OpenAI API key ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    # You can handle this inside Streamlit instead; here we just keep a placeholder.
    # In Streamlit we'll set os.environ["OPENAI_API_KEY"] based on user input if needed.
    pass

# ---- Models ----
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# ---- Vector store path ----
VECTOR_DB_DIR = "./my_context_db"

# ---- Chunking params ----
CHUNK_SIZE = 3500
CHUNK_OVERLAP = 0

# ---- Retriever defaults ----
RETRIEVER_K = 5