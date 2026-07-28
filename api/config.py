import os
from dotenv import load_dotenv 


# --- Qdrant Configuration ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "interactions")

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/e5-base-v2")

EMBEDDING_TASK_PREFIX_QUERY = os.getenv("EMBEDDING_TASK_PREFIX_QUERY", "query: ")
EMBEDDING_TASK_PREFIX_PASSAGE = os.getenv("EMBEDDING_TASK_PREFIX_PASSAGE", "passage: ")

# --- Great Expectations Configuration ---
GE_EXPECTATION_SUITE_NAME = os.getenv("GE_EXPECTATION_SUITE_NAME", "interaction_suite")
# Path to the GE data context config file (usually great_expectations.yml)
GE_CONTEXT_ROOT_DIR = os.getenv("GE_CONTEXT_ROOT_DIR", "./great_expectations")

# --- API Configuration ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))