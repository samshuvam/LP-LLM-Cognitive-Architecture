import logging
from fastapi import FastAPI
from .api.interaction_router import router, set_dependencies
from .core.embedding_generator import EmbeddingGenerator
from .db.vector_db_manager import VectorDBManager
from .core.data_validation import DataValidator
from .config import API_HOST, API_PORT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize Components ---
logger.info("Initializing LP-LLM Application components...")
embedding_generator = EmbeddingGenerator() # Uses default model from config
vector_db_manager = VectorDBManager() # Uses default host/port/collection from config
data_validator = DataValidator() # Uses default suite name and context dir from config

# Initialize the Qdrant collection based on the embedding model's vector size
vector_db_manager.init_collection(vector_size=embedding_generator.vector_size)

# Set the initialized components as dependencies for the router
set_dependencies(embedding_generator, vector_db_manager, data_validator)

# --- Create FastAPI App Instance ---
app = FastAPI(title="LP-LLM Cognitive System API", description="Lifelong Personalized LLM Vector Logging API by Shuvam (https://github.com/samshuvam)", version="3.0.0", contact={"name": "Shuvam", "url": "https://github.com/samshuvam"})

# Include the interaction router
app.include_router(router)

@app.get("/")
def read_root():
    return {"status": "LP-LLM Interaction Logger is running!"}

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Uvicorn server on {API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)