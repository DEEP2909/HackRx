from app.core.config import settings
from loguru import logger

def authenticate_token(token: str) -> bool:
    """Authenticate the provided token"""
    
    # Remove 'Bearer ' prefix if present
    if token.startswith('Bearer '):
        token = token[7:]
    
    # Check against configured token
    is_valid = token == settings.API_TOKEN
    
    if not is_valid:
        logger.warning(f"Invalid token attempted: {token[:10]}...")
    
    return is_valid
