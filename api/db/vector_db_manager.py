import logging
import uuid
from typing import List, Dict, Any
import numpy as np
import qdrant_client
from qdrant_client.http import models
from ..config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME

logger = logging.getLogger(__name__)

class VectorDBManager:
    def __init__(self, host: str = QDRANT_HOST, port: int = QDRANT_PORT, collection_name: str = QDRANT_COLLECTION_NAME):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client = qdrant_client.QdrantClient(host=host, port=port)
        logger.info(f"VectorDBManager initialized for collection: {collection_name} at {host}:{port}")

    def init_collection(self, vector_size: int, distance_type: str = "Cosine"):
        """Initializes the Qdrant collection if it doesn't exist."""
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=vector_size, distance=distance_type),
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection {self.collection_name} already exists.")
        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {e}")
            raise

    def upsert_interaction(self, user_id: str, prompt: str, response: str, timestamp: str, embedding: np.ndarray, session_id: str = None, implicit_feedback: Dict[str, Any] = None):
        """
        Inserts or updates an interaction record in the Qdrant collection.
        """
        try:
            point_id = str(uuid.uuid4()) # Generate a unique ID for this interaction point
            payload = {
                "user_id": user_id,
                "prompt": prompt,
                "response": response,
                "timestamp": timestamp,
                "session_id": session_id,
                "implicit_feedback": implicit_feedback or {},
            }

            # Prepare the PointStruct
            point = models.PointStruct(
                id=point_id,
                vector=embedding.tolist(), # Convert numpy array to list for Qdrant
                payload=payload
            )

            # Upsert the point
            self.client.upsert(collection_name=self.collection_name, points=[point])
            logger.info(f"Stored interaction for user {user_id} with point ID {point_id} in Qdrant.")

        except Exception as e:
            logger.error(f"Error upserting interaction for user {user_id} into Qdrant: {e}")
            raise # Re-raise so the API can handle the failure

    # Add other methods like search, delete, etc. later as needed
    # def search_interactions(self, query_embedding: np.ndarray, user_id: str = None, limit: int = 10):
    #     # Implementation for semantic search
    #     pass

# Global instance (or dependency injection can be used later)
# vector_db_manager = VectorDBManager() # Commented out for now, initialize in main app