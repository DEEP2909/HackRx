# app/services/query_engine.py - ULTRA-FAST VERSION

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
    """ULTRA-FAST query processing engine"""
    
    def __init__(self):
        self.document_cache = {}
        self.answer_cache = {}  # NEW: Answer caching
        
    async def process_query(self, document_url: str, questions: List[str]) -> List[str]:
        """ULTRA-FAST query processing"""
        start_time = time.time()
        
        try:
            # Step 1: Process document ONCE only
            logger.info(f"FAST processing: {document_url}")
            await self._ensure_document_processed(document_url)
            
            # Step 2: Process questions SEQUENTIALLY (avoid API rate limits)
            answers = []
            for question in questions:
                answer = await self._process_single_question(question)
                answers.append(answer)
            
            processing_time = time.time() - start_time
            logger.info(f"FAST query completed in {processing_time:.2f}s")
            
            return answers
            
        except Exception as e:
            logger.error(f"FAST query error: {e}")
            # Return fallback answers to avoid timeout
            return [f"Error processing question: {str(e)}" for _ in questions]
    
    async def _ensure_document_processed(self, document_url: str):
        """FAST document processing with aggressive caching"""
        
        # Check cache FIRST
        if document_url in self.document_cache:
            logger.info("Document in cache - SKIP processing")
            return
        
        # Process document FAST
        logger.info("FAST document processing")
        chunks = await document_processor.process_document(document_url)
        
        # Get embeddings in ONE batch
        logger.info(f"FAST embedding: {len(chunks)} chunks")
        chunk_texts = [chunk.content for chunk in chunks]
        
        # Single batch for speed
        embeddings = await embedding_service.get_embeddings(chunk_texts)
        
        # Assign embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        # Add to vector store
        await vector_store_service.add_documents(chunks)
        
        # Cache immediately
        self.document_cache[document_url] = {
            'chunks': len(chunks),
            'processed_at': time.time()
        }
        
        logger.info(f"FAST processing cached: {len(chunks)} chunks")
    
    async def _process_single_question(self, question: str) -> str:
        """FAST single question processing"""
        
        # Check answer cache
        cache_key = hash(question)
        if cache_key in self.answer_cache:
            logger.info("Answer from cache")
            return self.answer_cache[cache_key]
        
        try:
            # FAST embedding
            question_embeddings = await embedding_service.get_embeddings([question])
            question_embedding = question_embeddings[0]
            
            # FAST search - only top 2 results
            search_results = await vector_store_service.search(
                query_embedding=question_embedding,
                top_k=2  # REDUCED FROM 5
            )
            
            if not search_results:
                answer = "Information not available in the document"
            else:
                # FAST LLM processing
                llm_response = await llm_service.generate_answer(question, search_results)
                answer = llm_response.get('answer', 'Unable to generate answer')
            
            # Cache answer
            self.answer_cache[cache_key] = answer
            
            return answer
            
        except Exception as e:
            logger.error(f"FAST question error: {e}")
            return f"Error: {str(e)}"

# Singleton instance
query_engine = QueryEngine()
