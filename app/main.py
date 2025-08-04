from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from app.core.config import settings
from app.api.routes import query_router
from app.services.vector_store import vector_store_service

# Initialize services on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting LLM Query Retrieval System...")
    
    # Initialize vector store
    vector_store = vector_store_service
    await vector_store.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Query Retrieval System...")
    await vector_store.close()

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="LLM-Powered Intelligent Query-Retrieval System for document processing",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    query_router,
    prefix=f"{settings.API_PREFIX}/hackrx",
    tags=["query"]
)

@app.get("/")
async def root(request: Request):
    base_url = str(request.base_url)
    return {
        "message": "LLM Query Retrieval System API",
        "version": "1.0.0",
        "docs": f"{base_url}docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "llm-query-retrieval-system"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
