# app/services/document_processor.py - ULTRA-FAST VERSION

import io
import re
from typing import List, Dict, Any
from urllib.parse import urlparse
import httpx
from loguru import logger

# Document processing libraries
import PyPDF2
from docx import Document
import pdfplumber
from bs4 import BeautifulSoup

from app.core.config import settings
from app.models.document import DocumentChunk

class DocumentProcessor:
    """ULTRA-OPTIMIZED document processor for speed"""
    
    def __init__(self):
        # AGGRESSIVE chunking for maximum speed
        self.chunk_size = 400        # REDUCED FROM 800
        self.chunk_overlap = 50      # REDUCED FROM 100
        self.max_file_size = settings.max_file_size_bytes
        
        # Simple chunking only
        self.sentence_endings = r'[.!?]\s+'
        
    async def download_document(self, url: str) -> bytes:
        """Download document with aggressive timeout"""
        logger.info(f"Downloading document from: {url}")
        
        async with httpx.AsyncClient(timeout=10) as client:  # REDUCED TIMEOUT
            response = await client.get(url)
            response.raise_for_status()
            
            # Check file size
            content_length = int(response.headers.get('content-length', 0))
            if content_length > self.max_file_size:
                raise ValueError(f"File size too large")
            
            return response.content
    
    def extract_text_from_pdf(self, content: bytes) -> str:
        """FAST PDF extraction - first 5 pages only"""
        text = ""
        
        try:
            # Use pdfplumber for speed - LIMIT TO 5 PAGES
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                max_pages = min(5, len(pdf.pages))  # ONLY FIRST 5 PAGES
                for i in range(max_pages):
                    page_text = pdf.pages[i].extract_text()
                    if page_text:
                        # Minimal cleaning
                        page_text = re.sub(r'\s+', ' ', page_text)
                        text += page_text + " "
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise
        
        return text.strip()
    
    def extract_text_from_docx(self, content: bytes) -> str:
        """FAST DOCX extraction"""
        doc = Document(io.BytesIO(content))
        text_parts = []
        
        # LIMIT TO FIRST 20 PARAGRAPHS
        for i, paragraph in enumerate(doc.paragraphs[:20]):
            if paragraph.text.strip():
                text_parts.append(paragraph.text.strip())
        
        return ' '.join(text_parts)
    
    def extract_text_from_html(self, content: bytes) -> str:
        """FAST HTML extraction"""
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "header", "footer"]):
            element.decompose()
        
        text = soup.get_text(separator=' ', strip=True)
        return re.sub(r'\s+', ' ', text)
    
    def simple_chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """SIMPLE chunking for maximum speed"""
        if metadata is None:
            metadata = {}
        
        chunks = []
        words = text.split()
        
        # Simple word-based chunking
        for i in range(0, len(words), self.chunk_size):
            chunk_words = words[i:i + self.chunk_size]
            chunk_content = ' '.join(chunk_words)
            
            chunks.append(DocumentChunk(
                content=chunk_content.strip(),
                metadata={
                    **metadata,
                    'chunk_index': len(chunks),
                    'word_count': len(chunk_words)
                }
            ))
            
            # LIMIT TO 10 CHUNKS MAX
            if len(chunks) >= 10:
                break
        
        logger.info(f"Fast chunking created {len(chunks)} chunks")
        return chunks
    
    async def process_document(self, url: str) -> List[DocumentChunk]:
        """ULTRA-FAST document processing"""
        logger.info(f"FAST processing: {url}")
        
        # Download with timeout
        content = await self.download_document(url)
        
        # Determine file type
        parsed_url = urlparse(url)
        filename = parsed_url.path.split('/')[-1].lower()
        
        # Extract text - FAST methods only
        if filename.endswith('.pdf'):
            text = self.extract_text_from_pdf(content)
            doc_type = 'pdf'
        elif filename.endswith('.docx'):
            text = self.extract_text_from_docx(content)
            doc_type = 'docx'
        elif filename.endswith('.html') or filename.endswith('.eml'):
            text = self.extract_text_from_html(content)
            doc_type = 'email'
        else:
            text = content.decode('utf-8', errors='ignore')
            doc_type = 'text'
        
        # LIMIT TEXT LENGTH
        words = text.split()
        if len(words) > 2000:  # MAXIMUM 2000 WORDS
            text = ' '.join(words[:2000])
        
        if not text or len(text.strip()) < 50:
            raise ValueError("Document too short")
        
        # Create chunks
        metadata = {
            'source_url': url,
            'document_type': doc_type,
            'filename': filename
        }
        
        # Use simple chunking
        chunks = self.simple_chunk_text(text, metadata)
        
        logger.info(f"FAST processing complete: {len(chunks)} chunks")
        return chunks

# Singleton instance
document_processor = DocumentProcessor()
