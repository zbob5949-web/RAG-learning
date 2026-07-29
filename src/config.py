import os
from pathlib import Path
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")
LLM_URL="https://api.deepseek.com/v1"
LLM_MODEL="deepseek-chat"
EMBEDDING_MODEL="BAAI/bge-small-zh-v1.5"

CHUNK_SIZE=500
CHUNK_OVERLAP=50

TOP_K=3

CHROMA_PERSIST_DIR="./chroma_db"

DOCS_DIR = str(Path(__file__).parent.parent / "data" / "docs")


