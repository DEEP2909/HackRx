from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
from loguru import logger

from app.services.query_engine import query_engine
from app.services.authentication import authenticate_token

class QueryRequest(BaseModel):
    documents: str  # URL to the document
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# HTTP Bearer authentication
security = HTTPBearer()

# Initialize router
query_router = APIRouter()

@query_router.post("/run", response_model=QueryResponse)
async def run_query(request: QueryRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Handle authentication
    if not authenticate_token(credentials.credentials):
        logger.error("Unauthorized access attempt.")
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        # Process query
        answers = await query_engine.process_query(request.documents, request.questions)
        return JSONResponse(status_code=200, content={"answers": answers})

    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during query processing.")
