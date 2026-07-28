from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

class InteractionLog(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user.", example="user_abc123")
    prompt: str = Field(..., description="The user's input query.", example="What is the capital of France?")
    response: str = Field(..., description="The model's generated response.", example="The capital of France is Paris.")
    timestamp: Optional[str] = Field(None, description="UTC timestamp of the interaction in ISO 8601 format (e.g., 2023-10-27T10:00:00Z). If not provided, the server will set it.")
    session_id: Optional[str] = Field(None, description="Optional session identifier.", example="sess_xyz789")
    implicit_feedback: Optional[Dict[str, Any]] = Field({}, description="Optional implicit feedback signals like regenerations or edits.", example={"regenerated": True, "edited_response": "Paris is the beautiful capital of France."})

    class Config:
        # Allow extra fields if needed in the future, though discouraged initially
        extra = "forbid"