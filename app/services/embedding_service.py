from typing import List
from loguru import logger
import httpx
from app.core.config import settings

class EmbeddingService:
    def __init__(self):
        self.model_name = settings.OPENAI_EMBEDDING_MODEL
        self.api_key = settings.OPENAI_API_KEY

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        logger.info("Fetching embeddings from OpenAI.")
        headers = {
            'Authorization': f'Bearer {self.api_key}'
        }
        data = {
            'model': self.model_name,
            'input': texts
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                'https://api.openai.com/v1/embeddings',
                headers=headers,
                json=data
            )
            response.raise_for_status()
            embeddings = response.json().get('data')
            return [embedding['embedding'] for embedding in embeddings]

embedding_service = EmbeddingService()
