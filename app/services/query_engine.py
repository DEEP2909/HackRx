import time
from typing import List, Dict, Any
from loguru import logger
import asyncio

from app.services.document_processor import document_processor
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store_service
from app.services.llm_service import llm_service
from app.core.config import settings

class QueryEngine:
    """Main query processing engine that orchestrates all services"""
    
    def __init__(self):
        self.document_cache = {}  # Simple in-memory cache
        
    async def process_query(self, document_url: str, questions: List[str]) -> List[str]:
        """Process a query with document URL and questions"""
        start_time = time.time()
        
        try:
            # Step 1: Process document if not cached
            logger.info(f"Processing query for document: {document_url}")
            await self._ensure_document_processed(document_url)
            
            # Step 2: Process each question
            answers = await self._process_questions(questions)
            
            # Log processing time
            processing_time = time.time() - start_time
            logger.info(f"Query processed in {processing_time:.2f} seconds")
            
            return answers
            
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            raise
    
    async def _ensure_document_processed(self, document_url: str):
        """Ensure document is processed and indexed"""
        
        # Check cache
        if document_url in self.document_cache:
            logger.info("Document found in cache")
            return
        
        # Process document
        logger.info("Processing new document")
        chunks = await document_processor.process_document(document_url)
        
        # Get embeddings for chunks
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        chunk_texts = [chunk.content for chunk in chunks]
        
        # Batch embeddings for efficiency
        batch_size = 20
        all_embeddings = []
        
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i:i+batch_size]
            embeddings = await embedding_service.get_embeddings(batch)
            all_embeddings.extend(embeddings)
        
        # Assign embeddings to chunks
        for chunk, embedding in zip(chunks, all_embeddings):
            chunk.embedding = embedding
        
        # Add to vector store
        await vector_store_service.add_documents(chunks)
        
        # Cache document
        self.document_cache[document_url] = {
            'chunks': len(chunks),
            'processed_at': time.time()
        }
        
        logger.info(f"Document processed and indexed: {len(chunks)} chunks")
    
    async def _process_questions(self, questions: List[str]) -> List[str]:
        """Process multiple questions in parallel"""
        
        # Process questions concurrently for efficiency
        tasks = [self._process_single_question(question) for question in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results
        answers = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing question {i}: {result}")
                answers.append(f"Error processing question: {str(result)}")
            else:
                answers.append(result)
        
        return answers
    
    async def _process_single_question(self, question: str) -> str:
        """Process a single question"""
        logger.info(f"Processing question: {question}")
        
        try:
            # Step 1: Get embedding for question
            question_embeddings = await embedding_service.get_embeddings([question])
            question_embedding = question_embeddings[0]
            
            # Step 2: Search for relevant chunks
            search_results = await vector_store_service.search(
                query_embedding=question_embedding,
                top_k=settings.TOP_K_RESULTS
            )
            
            if not search_results:
                return "No relevant information found in the document to answer this question."
            
            # Step 3: Generate answer using LLM
            llm_response = await llm_service.generate_answer(question, search_results)
            
            # Extract the answer
            answer = llm_response.get('answer', 'Unable to generate answer')
            
            # Log token usage
            token_usage = llm_response.get('token_usage', {})
            logger.info(f"Token usage - Total: {token_usage.get('total_tokens', 0)}")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            return f"Error processing question: {str(e)}"
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'cached_documents': len(self.document_cache),
            'cache_details': self.document_cache
        }

# Singleton instance
query_engine = QueryEngine()
