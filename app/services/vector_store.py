import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger
import json
import os
import pickle

from app.core.config import settings
from app.models.document import DocumentChunk

class VectorStoreService:
    """Service for managing vector storage using FAISS or Pinecone"""
    
    def __init__(self):
        self.store_type = settings.VECTOR_STORE_TYPE
        self.dimension = settings.EMBEDDING_DIMENSION
        self.top_k = settings.TOP_K_RESULTS
        self.index = None
        self.metadata_store = {}
        
    async def initialize(self):
        """Initialize the vector store"""
        logger.info(f"Initializing {self.store_type} vector store")
        
        if self.store_type == "faiss":
            await self._init_faiss()
        elif self.store_type == "pinecone":
            await self._init_pinecone()
        else:
            raise ValueError(f"Unknown vector store type: {self.store_type}")
    
    async def _init_faiss(self):
        """Initialize FAISS index"""
        try:
            import faiss
            
            # Try to load existing index
            index_path = "faiss_index.bin"
            metadata_path = "faiss_metadata.pkl"
            
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                self.index = faiss.read_index(index_path)
                with open(metadata_path, 'rb') as f:
                    self.metadata_store = pickle.load(f)
                logger.info("Loaded existing FAISS index")
            else:
                # Create new index
                self.index = faiss.IndexFlatL2(self.dimension)
                logger.info("Created new FAISS index")
                
        except ImportError:
            logger.error("FAISS not installed. Please install faiss-cpu or faiss-gpu")
            raise
    
    async def _init_pinecone(self):
        """Initialize Pinecone index"""
        try:
            from pinecone import Pinecone, ServerlessSpec
            
            pc=Pinecone(
                api_key=settings.PINECONE_API_KEY,
                environment=settings.PINECONE_ENVIRONMENT
            )
            
            # Check if index exists
            if settings.PINECONE_INDEX_NAME not in pc.list_indexes().names():
                pc.create_index(
                    name=settings.PINECONE_INDEX_NAME,
                    dimension=self.dimension,
                    metric='cosine',
                    spec=ServerlessSpec(
                        region="us-east-1",
                        cloud="aws"
                    )
                )
                logger.info(f"Created new Pinecone index: {settings.PINECONE_INDEX_NAME}")
            
            self.index = pc.Index(settings.PINECONE_INDEX_NAME)
            logger.info(f"Connected to Pinecone index: {settings.PINECONE_INDEX_NAME}")
            
        except ImportError:
            logger.error("Pinecone not installed. Please install pinecone-client")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    async def add_documents(self, chunks: List[DocumentChunk]):
        """Add document chunks to the vector store"""
        if self.index is None:
            raise RuntimeError("Vector store not initialized. Call `initialize()` first.")
        
        logger.info(f"Adding {len(chunks)} document chunks to vector store")
        
        if self.store_type == "faiss":
            await self._add_to_faiss(chunks)
        elif self.store_type == "pinecone":
            await self._add_to_pinecone(chunks)
    
    async def _add_to_faiss(self, chunks: List[DocumentChunk]):
        """Add chunks to FAISS index"""
        import faiss
        
        # Extract embeddings
        embeddings = np.array([chunk.embedding for chunk in chunks if chunk.embedding])
        
        if len(embeddings) == 0:
            logger.warning("No embeddings found in chunks")
            return
        
        # Add to index
        start_id = self.index.ntotal
        self.index.add(embeddings)
        
        # Store metadata
        for i, chunk in enumerate(chunks):
            if chunk.embedding:
                self.metadata_store[start_id + i] = {
                    'content': chunk.content,
                    'metadata': chunk.metadata
                }
        
        # Save index and metadata
        faiss.write_index(self.index, "faiss_index.bin")
        with open("faiss_metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata_store, f)
        
        logger.info(f"Added {len(embeddings)} vectors to FAISS index")
    
    async def _add_to_pinecone(self, chunks: List[DocumentChunk]):
        """Add chunks to Pinecone index"""
        vectors = []
        
        for i, chunk in enumerate(chunks):
            if chunk.embedding:
                vector_data = {
                    'id': f"{chunk.metadata.get('source_url', '')}_{i}",
                    'values': chunk.embedding,
                    'metadata': {
                        'content': chunk.content,
                        **chunk.metadata
                    }
                }
                vectors.append(vector_data)
        
        if vectors:
            # Batch upsert for efficiency
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Added {len(vectors)} vectors to Pinecone index")
    
    async def search(self, query_embedding: List[float], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if top_k is None:
            top_k = self.top_k
        
        logger.info(f"Searching for top {top_k} similar documents")
        
        if self.store_type == "faiss":
            return await self._search_faiss(query_embedding, top_k)
        elif self.store_type == "pinecone":
            return await self._search_pinecone(query_embedding, top_k)
    
    async def _search_faiss(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search in FAISS index"""
        query_vector = np.array([query_embedding])
        
        # Search
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1 and idx in self.metadata_store:
                result = {
                    'score': float(1 / (i + dist)),  # Convert distance to similarity score
                    'content': self.metadata_store[idx]['content'],
                    'metadata': self.metadata_store[idx]['metadata']
                }
                results.append(result)
        
        return results
    
    async def _search_pinecone(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search in Pinecone index"""
        response = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        results = []
        for match in response['matches']:
            result = {
                'score': match['score'],
                'content': match['metadata'].get('content', ''),
                'metadata': {k: v for k, v in match['metadata'].items() if k != 'content'}
            }
            results.append(result)
        
        return results
    
    async def close(self):
        """Close vector store connections"""
        logger.info("Closing vector store connections")
        # Additional cleanup if needed

# Singleton instance
vector_store_service = VectorStoreService()
