import openai
from typing import List, Dict, Any
from loguru import logger
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings

class LLMService:
    """Service for LLM interactions using OpenAI GPT-4"""
    
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL
        self.max_tokens = settings.MAX_TOKENS
        self.temperature = settings.TEMPERATURE
        openai.api_key = self.api_key
        
        # Token counter
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=20))
    async def generate_answer(self, question: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer using GPT-4 with relevant context"""
        
        # Prepare context
        context_text = self._prepare_context(context)
        
        # Create prompt
        prompt = self._create_prompt(question, context_text)
        
        # Check token count
        prompt_tokens = self.count_tokens(prompt)
        logger.info(f"Prompt tokens: {prompt_tokens}")
        
        try:
            # Call OpenAI API
            response = await self._call_openai(prompt)
            
            # Extract answer and metadata
            answer_text = response['choices'][0]['message']['content']
            
            # Parse structured response
            result = self._parse_response(answer_text)
            
            # Add token usage
            result['token_usage'] = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': response['usage']['completion_tokens'],
                'total_tokens': response['usage']['total_tokens']
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    def _prepare_context(self, context: List[Dict[str, Any]]) -> str:
        """Prepare context from search results"""
        context_parts = []
        
        for i, item in enumerate(context[:5]):  # Limit to top 5 results
            context_parts.append(f"[Source {i+1}]")
            context_parts.append(f"Content: {item['content']}")
            context_parts.append(f"Relevance Score: {item['score']:.2f}")
            
            # Add metadata if available
            metadata = item.get('metadata', {})
            if metadata:
                context_parts.append(f"Document: {metadata.get('filename', 'Unknown')}")
            
            context_parts.append("")  # Empty line for separation
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create prompt for GPT-4"""
        prompt = f"""You are an intelligent document analysis system specialized in insurance, legal, HR, and compliance domains. Your task is to answer questions based on the provided document context.

**Context from Documents:**
{context}

**Question:** {question}

**Instructions:**
1. Answer the question based ONLY on the provided context and in 50 to 60 words.
2. Read the input carefully, all the answer to the question is in there.
3. If the answer is found in the context, provide a clear and concise response
4. Include specific references to policy clauses, sections, or conditions when applicable
5. If the context doesn't contain enough information to answer the question, state "Information not Available"
6. Provide your response in the following JSON format:

{{
    "answer": "Your detailed answer here",
    "confidence": 0.95,  // Confidence score between 0 and 1
    "relevant_clauses": ["clause 1", "clause 2"],  // List of relevant clauses or sections
    "explanation": "Brief explanation of how you arrived at this answer",
    "found_in_context": true  // Whether the answer was found in the provided context
}}

Provide only the JSON response without any additional text."""
        
        return prompt
    
    async def _call_openai(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API"""
        import httpx
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a document analysis expert."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"}
        }
        
        async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return response.json()
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the structured response from GPT-4"""
        import json
        
        try:
            # Parse JSON response
            result = json.loads(response_text)
            
            # Ensure all required fields are present
            required_fields = ['answer', 'confidence', 'relevant_clauses', 'explanation', 'found_in_context']
            for field in required_fields:
                if field not in result:
                    result[field] = None
            
            return result
            
        except json.JSONDecodeError:
            # Fallback for non-JSON responses
            return {
                'answer': response_text,
                'confidence': 0.5,
                'relevant_clauses': [],
                'explanation': 'Response was not in expected format',
                'found_in_context': True
            }

# Singleton instance
llm_service = LLMService()

