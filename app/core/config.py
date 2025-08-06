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

    # OpenAI Configuration - ULTRA OPTIMIZED FOR SPEED
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    MAX_TOKENS: int = 500                   # REDUCED FROM 1500
    TEMPERATURE: float = 0.0                # DETERMINISTIC

    # Pinecone Configuration
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    PINECONE_INDEX_NAME: str = "document-embeddings"

    # Database Configuration
    DATABASE_URL: str
    DB_ECHO: bool = False

    # Document Processing - ULTRA OPTIMIZED
    MAX_FILE_SIZE_MB: int = 10              # REDUCED FROM 25
    ALLOWED_EXTENSIONS: str = "pdf,docx,txt,eml"
    CHUNK_SIZE: int = 400                   # REDUCED FROM 800
    CHUNK_OVERLAP: int = 50                 # REDUCED FROM 100

    # Vector Store - ULTRA OPTIMIZED
    VECTOR_STORE_TYPE: str = "faiss"
    EMBEDDING_DIMENSION: int = 1536
    TOP_K_RESULTS: int = 2                  # REDUCED FROM 5

    # Performance Settings - AGGRESSIVE
    REQUEST_TIMEOUT: int = 25               # REDUCED FROM 60
    MAX_CONCURRENT_REQUESTS: int = 2        # REDUCED FROM 5
    CACHE_TTL_SECONDS: int = 3600

    # Token Management - ULTRA STRICT
    MAX_CONTEXT_TOKENS: int = 800           # REDUCED FROM 3000
    MAX_PROMPT_TOKENS: int = 1500           # REDUCED FROM 4000
    TOKEN_BUFFER: int = 200                 # REDUCED FROM 500
    
    # Batch Processing - FAST
    EMBEDDING_BATCH_SIZE: int = 50          # INCREASED FROM 10
    PARALLEL_QUESTIONS: int = 1             # SEQUENTIAL ONLY

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
