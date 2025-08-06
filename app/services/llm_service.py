# app/services/llm_service.py - ULTRA-FAST VERSION

import openai
from typing import List, Dict, Any
from loguru import logger
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings

class LLMService:
    """ULTRA-FAST LLM service"""
    
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model = "gpt-3.5-turbo"  # FASTEST MODEL
        self.max_tokens = 300         # VERY LIMITED
        self.temperature = 0.0        # DETERMINISTIC
        openai.api_key = self.api_key
        
        # Token counter
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens"""
        return len(self.encoding.encode(text))
    
    def truncate_context(self, context: List[Dict[str, Any]], max_tokens: int = 500) -> str:
        """AGGRESSIVE context truncation"""
        if not context:
            return "No context available"
        
        # Take only the FIRST result (highest relevance)
        best_context = context[0]['content']
        
        # Truncate to fit token limit
        words = best_context.split()
        if len(words) > 100:  # MAXIMUM 100 WORDS
            best_context = ' '.join(words[:100])
        
        return best_context
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
    async def generate_answer(self, question: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ULTRA-FAST answer generation"""
        
        # MINIMAL context
        context_text = self.truncate_context(context, 500)
        
        # MINIMAL prompt
        prompt = f"""Context: {context_text}

Question: {question}

Answer in 10-60 words only:"""
        
        try:
            # FAST API call
            response = await self._call_openai_fast(prompt)
            answer_text = response['choices'][0]['message']['content']
            
            return {
                'answer': answer_text.strip(),
                'confidence': 0.9,
                'token_usage': response.get('usage', {})
            }
            
        except Exception as e:
            logger.error(f"FAST LLM error: {e}")
            return {
                'answer': 'Unable to process this question',
                'confidence': 0.0,
                'token_usage': {}
            }
    
    async def _call_openai_fast(self, prompt: str) -> Dict[str, Any]:
        """ULTRA-FAST OpenAI call"""
        import httpx
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 200,  # VERY LIMITED
            "timeout": 10       # FAST TIMEOUT
        }
        
        async with httpx.AsyncClient(timeout=15) as client:  # FAST CLIENT
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return response.json()

# Singleton instance
llm_service = LLMService()
