from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime

class DocumentChunk(BaseModel):
    """Model for document chunks with metadata"""
    content: str
    metadata: Dict[str, Any] = {}
    embedding: Optional[list[float]] = None
    chunk_id: Optional[str] = None
    created_at: datetime = datetime.utcnow()
    
    class Config:
        arbitrary_types_allowed = True
