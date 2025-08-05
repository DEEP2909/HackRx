from pydantic_settings import BaseSettings
from typing import Optional, List
import os
from functools import lru_cache

class Settings(BaseSettings):
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "LLM Query Retrieval System"
    DEBUG: bool = True

    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    API_TOKEN: str

    # OpenAI Configuration - OPTIMIZED FOR COST
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-3.5-turbo"  # Changed from gpt-4 to save costs
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    MAX_TOKENS: int = 1500              # Reduced from 4000
    TEMPERATURE: float = 0.1            # Lower for more focused responses

    # Pinecone Configuration
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    PINECONE_INDEX_NAME: str = "document-embeddings"

    # Database Configuration
    DATABASE_URL: str
    DB_ECHO: bool = False

    # Document Processing - OPTIMIZED
    MAX_FILE_SIZE_MB: int = 25          # Reduced from 50
    ALLOWED_EXTENSIONS: str = "pdf,docx,txt,eml"
    CHUNK_SIZE: int = 800              # Reduced from 1000
    CHUNK_OVERLAP: int = 100           # Reduced from 200

    # Vector Store - OPTIMIZED
    VECTOR_STORE_TYPE: str = "faiss"
    EMBEDDING_DIMENSION: int = 1536
    TOP_K_RESULTS: int = 5             # Reduced from 10 to limit context

    # Performance Settings
    REQUEST_TIMEOUT: int = 60
    MAX_CONCURRENT_REQUESTS: int = 5    # Reduced from 10
    CACHE_TTL_SECONDS: int = 3600

    # Token Management - NEW
    MAX_CONTEXT_TOKENS: int = 3000     # Limit context size
    MAX_PROMPT_TOKENS: int = 4000      # Total prompt limit
    TOKEN_BUFFER: int = 500            # Safety buffer
    
    # Batch Processing - NEW
    EMBEDDING_BATCH_SIZE: int = 10     # Reduced from 20
    PARALLEL_QUESTIONS: int = 3        # Limit parallel processing

    @property
    def allowed_extensions_list(self) -> List[str]:
        return self.ALLOWED_EXTENSIONS.split(",")

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
