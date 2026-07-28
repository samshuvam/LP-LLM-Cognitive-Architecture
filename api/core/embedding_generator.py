import logging
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from ..config import EMBEDDING_MODEL_NAME, EMBEDDING_TASK_PREFIX_QUERY, EMBEDDING_TASK_PREFIX_PASSAGE

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        logger.info(f"Initializing EmbeddingGenerator with model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval() # Set to evaluation mode
        # Determine vector size from the model config
        self.vector_size = self.model.config.hidden_size
        logger.info(f"Embedding vector size determined: {self.vector_size}")

    def generate_embedding(self, text: str, is_query: bool = True) -> np.ndarray:
        """
        Generates an embedding for the given text.
        Args:
            text (str): The input text.
            is_query (bool): If True, applies the query prefix (e.g., for E5 models).
                             If False, applies the passage prefix.
        Returns:
            np.ndarray: The embedding vector (dtype float32).
        """
        try:
            # Apply task prefix if required by the model
            prefix = EMBEDDING_TASK_PREFIX_QUERY if is_query else EMBEDDING_TASK_PREFIX_PASSAGE
            prefixed_text = f"{prefix}{text}"

            inputs = self.tokenizer(
                prefixed_text,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512 # Or whatever your model's max length is
            )

            with torch.no_grad(): # Disable gradient calculation for efficiency
                outputs = self.model(**inputs)
                # Use mean pooling of the last hidden states for a fixed-size representation
                # Alternative: Use the [CLS] token embedding (outputs.last_hidden_state[:, 0, :])
                # Mean pooling is often more robust for longer sequences.
                embeddings = outputs.last_hidden_state.mean(dim=1) # Shape: (1, hidden_size)

            # Extract the single embedding vector and convert to numpy
            embedding_np = embeddings.squeeze(0).numpy().astype(np.float32) # Shape: (hidden_size,)
            logger.debug(f"Generated embedding of shape {embedding_np.shape} for text: {text[:50]}...")
            return embedding_np

        except Exception as e:
            logger.error(f"Error generating embedding for text '{text[:50]}...': {e}")
            raise # Re-raise to handle upstream

# Global instance (or dependency injection can be used later)
# embedding_gen = EmbeddingGenerator() # Commented out for now, initialize in main app