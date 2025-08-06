import openai
from typing import List, Dict, Any
from loguru import logger
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings

class LLMService:
    """Token-optimized LLM service for efficient document analysis"""
    
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL
        self.max_tokens = 1500  # Reduced from 4000
        self.temperature = settings.TEMPERATURE
        openai.api_key = self.api_key
        
        # Token counter
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Token limits for optimization
        self.max_context_tokens = 3000  # Maximum tokens for context
        self.max_prompt_tokens = 4000   # Maximum total prompt tokens
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def truncate_context(self, context: List[Dict[str, Any]], max_tokens: int = 3000) -> List[Dict[str, Any]]:
        """Intelligently truncate context to stay within token limits"""
        total_tokens = 0
        truncated_context = []
        
        # Sort by relevance score (highest first)
        sorted_context = sorted(context, key=lambda x: x.get('score', 0), reverse=True)
        
        for item in sorted_context:
            content = item['content']
            content_tokens = self.count_tokens(content)
            
            if total_tokens + content_tokens <= max_tokens:
                truncated_context.append(item)
                total_tokens += content_tokens
            else:
                # Try to fit a truncated version
                remaining_tokens = max_tokens - total_tokens - 100  # Leave some buffer
                if remaining_tokens > 200:  # Only if we have meaningful space left
                    # Truncate content to fit
                    words = content.split()
                    truncated_words = []
                    temp_tokens = 0
                    
                    for word in words:
                        word_tokens = self.count_tokens(word + " ")
                        if temp_tokens + word_tokens <= remaining_tokens:
                            truncated_words.append(word)
                            temp_tokens += word_tokens
                        else:
                            break
                    
                    if len(truncated_words) > 10:  # Only if we got meaningful content
                        truncated_item = item.copy()
                        truncated_item['content'] = ' '.join(truncated_words) + "..."
                        truncated_context.append(truncated_item)
                
                break
        
        logger.info(f"Context truncated: {len(context)} -> {len(truncated_context)} items, ~{total_tokens} tokens")
        return truncated_context
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_answer(self, question: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer using GPT-4 with optimized token usage"""
        
        # Step 1: Optimize context
        optimized_context = self.truncate_context(context, self.max_context_tokens)
        context_text = self._prepare_context_optimized(optimized_context)
        
        # Step 2: Create optimized prompt
        prompt = self._create_optimized_prompt(question, context_text)
        
        # Step 3: Check token count
        prompt_tokens = self.count_tokens(prompt)
        logger.info(f"Optimized prompt tokens: {prompt_tokens}")
        
        if prompt_tokens > self.max_prompt_tokens:
            logger.warning(f"Prompt still too long ({prompt_tokens} tokens), further truncating...")
            # Further reduce context
            optimized_context = self.truncate_context(context, self.max_context_tokens // 2)
            context_text = self._prepare_context_optimized(optimized_context)
            prompt = self._create_optimized_prompt(question, context_text)
            prompt_tokens = self.count_tokens(prompt)
            logger.info(f"Re-optimized prompt tokens: {prompt_tokens}")
        
        try:
            # Call OpenAI API with optimized settings
            response = await self._call_openai_optimized(prompt)
            
            # Extract answer and metadata
            answer_text = response['choices'][0]['message']['content']
            
            # Parse structured response
            result = self._parse_response(answer_text)
            
            # Add token usage
            result['token_usage'] = {
                'prompt_tokens': response['usage']['prompt_tokens'],
                'completion_tokens': response['usage']['completion_tokens'],
                'total_tokens': response['usage']['total_tokens'],
                'context_items_used': len(optimized_context)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    def _prepare_context_optimized(self, context: List[Dict[str, Any]]) -> str:
        """Prepare context with minimal formatting to save tokens"""
        if not context:
            return "No relevant context found."
        
        context_parts = []
        
        for i, item in enumerate(context):
            # Minimal formatting to save tokens
            context_parts.append(f"[{i+1}] {item['content']}")
            
            # Only add metadata if it's really important
            metadata = item.get('metadata', {})
            if metadata.get('filename'):
                context_parts.append(f"Source: {metadata['filename']}")
        
        return "\n".join(context_parts)
    
    def _create_optimized_prompt(self, question: str, context: str) -> str:
        """Create a token-efficient prompt"""
        prompt = f"""Answer based on the provided context. Be concise but accurate.

Context:
{context}

Question: {question}

Provide a JSON response with these fields:
- "answer": 40 to 60 words explained answer
- "confidence": score 0-1
- "found_in_context": true/false

If no relevant info found, set answer to "Information not Available"."""
        
        return prompt
    
    async def _call_openai_optimized(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API with optimized settings"""
        import httpx
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5-turbo-instruct",  # Use GPT-3.5 instead of GPT-4 to save costs
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,  # Lower temperature for more focused responses
            "max_tokens": self.max_tokens,  # Limit response length
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
        """Parse the structured response from LLM"""
        import json
        
        try:
            result = json.loads(response_text)
            
            # Ensure required fields
            return {
                'answer': result.get('answer', 'Unable to generate answer'),
                'confidence': result.get('confidence', 0.5),
                'relevant_clauses': result.get('relevant_clauses', []),
                'explanation': result.get('explanation', ''),
                'found_in_context': result.get('found_in_context', True)
            }
            
        except json.JSONDecodeError:
            # Fallback for non-JSON responses
            return {
                'answer': response_text.strip(),
                'confidence': 0.5,
                'relevant_clauses': [],
                'explanation': 'Response parsing failed',
                'found_in_context': True
            }

# Singleton instance
llm_service = LLMService()



