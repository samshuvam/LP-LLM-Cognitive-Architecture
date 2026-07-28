import logging
from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Optional
from ..models.interaction_schema import InteractionLog
from ..core.embedding_generator import EmbeddingGenerator
from ..db.vector_db_manager import VectorDBManager
from ..core.data_validation import DataValidator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Interactions"])

# --- Dependency Injection placeholders (will be configured in main.py) ---
embedding_gen: Optional[EmbeddingGenerator] = None
vector_db_manager: Optional[VectorDBManager] = None
data_validator: Optional[DataValidator] = None

def set_dependencies(emb_gen: EmbeddingGenerator, vec_db_man: VectorDBManager, val: DataValidator):
    global embedding_gen, vector_db_manager, data_validator
    embedding_gen = emb_gen
    vector_db_manager = vec_db_man
    data_validator = val

@router.post("/log_interaction")
async def log_interaction(interaction: InteractionLog):
    try:
        logger.info(f"Received interaction log request for user: {interaction.user_id}")

        # 1. Set timestamp if not provided (by Pydantic or here)
        if interaction.timestamp is None:
            interaction.timestamp = datetime.utcnow().isoformat() + "Z"

        # 2. Validate data using Great Expectations
        interaction_dict = interaction.dict()
        if not data_validator or not data_validator.validate_interaction(interaction_dict):
             logger.warning(f"Interaction validation failed for user {interaction.user_id}. Data: {interaction_dict}")
             raise HTTPException(status_code=422, detail="Interaction data did not meet quality standards.")

        # 3. Generate embedding using the embedding generator
        # Combine prompt and response for a single embedding representing the interaction context
        combined_text = f"{interaction.prompt} {interaction.response}"
        embedding = embedding_gen.generate_embedding(combined_text, is_query=False) # Treat as a passage

        # 4. Store in Vector DB
        vector_db_manager.upsert_interaction(
            user_id=interaction.user_id,
            prompt=interaction.prompt,
            response=interaction.response,
            timestamp=interaction.timestamp,
            embedding=embedding,
            session_id=interaction.session_id,
            implicit_feedback=interaction.implicit_feedback
        )

        return {"message": "Interaction logged successfully"}

    except HTTPException:
        # Re-raise FastAPI HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error logging interaction for user {interaction.user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error occurred while logging interaction.")