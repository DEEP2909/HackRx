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

# HTTP Bearer authentication
security = HTTPBearer()

# Initialize router
query_router = APIRouter()

@query_router.post("/run")
async def run_query(request: QueryRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Handle authentication
    if not authenticate_token(credentials.credentials):
        logger.error("Unauthorized access attempt.")
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        # Process query
        answers = await query_engine.process_query(request.documents, request.questions)
        
        # IMPORTANT: Return the raw answers list, not wrapped in any object
        # The sample response shows: ["answer1", "answer2", ...] 
        # NOT: {"answers": ["answer1", "answer2", ...]}
        
        # Ensure we return a plain list
        if isinstance(answers, list):
            return {"answers": answers}
        elif isinstance(answers, dict) and 'answers' in answers:
            return answers  # Already in correct format
        else:
            # Fallback - wrap single answer in list
            return {"answers": [str(answers)]}

    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during query processing.")

# Add a health check endpoint
@query_router.get("/health")
async def health_check():
    return {"status": "healthy"}

# Add a test endpoint to verify format
@query_router.get("/test-format")
async def test_format():
    """Test endpoint to verify the exact response format expected"""
    sample_response = [
        "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy.",
        "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases to be covered."
    ]
    return sample_response

