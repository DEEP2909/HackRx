import os
from app.core.config import settings
from loguru import logger

def authenticate_token(token: str) -> bool:
    """Authenticate the provided token"""
    
    # Remove 'Bearer ' prefix if present
    if token.startswith('Bearer '):
        token = token[7:]
    
    # Get API token from settings
    expected_token = settings.API_TOKEN
    
    # Debug logging (remove in production)
    logger.info(f"Expected token: {expected_token[:10]}...")
    logger.info(f"Received token: {token[:10]}...")
    
    # Check against configured token
    is_valid = token == expected_token
    
    if not is_valid:
        logger.warning(f"Invalid token attempted: {token[:10]}...")
    else:
        logger.info("Token authentication successful")
    
    return is_valid
