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
    """Service for processing various document formats"""
    
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.max_file_size = settings.max_file_size_bytes
        
    async def download_document(self, url: str) -> bytes:
        """Download document from URL"""
        logger.info(f"Downloading document from: {url}")
        
        async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            # Check file size
            content_length = int(response.headers.get('content-length', 0))
            if content_length > self.max_file_size:
                raise ValueError(f"File size {content_length} exceeds maximum allowed size")
            
            return response.content
    
    def extract_text_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF content"""
        text = ""
        
        # Try PyPDF2 first
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
                
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed, trying pdfplumber: {e}")
            
            # Fallback to pdfplumber
            try:
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e2:
                logger.error(f"PDF extraction failed: {e2}")
                raise
        
        return text.strip()
    
    def extract_text_from_docx(self, content: bytes) -> str:
        """Extract text from DOCX content"""
        doc = Document(io.BytesIO(content))
        text = ""
        
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        # Also extract from tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    text += cell.text + " "
                text += "\n"
        
        return text.strip()
    
    def extract_text_from_html(self, content: bytes) -> str:
        """Extract text from HTML/Email content"""
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def create_chunks(self, text: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Split text into overlapping chunks"""
        if metadata is None:
            metadata = {}
            
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if chunk_text:
                chunk = DocumentChunk(
                    content=chunk_text,
                    metadata={
                        **metadata,
                        'chunk_index': len(chunks),
                        'start_word': i,
                        'end_word': min(i + self.chunk_size, len(words))
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    async def process_document(self, url: str) -> List[DocumentChunk]:
        """Main method to process document from URL"""
        logger.info(f"Processing document: {url}")
        
        # Download document
        content = await self.download_document(url)
        
        # Determine file type from URL
        parsed_url = urlparse(url)
        filename = parsed_url.path.split('/')[-1].lower()
        
        # Extract text based on file type
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
            # Try to decode as plain text
            text = content.decode('utf-8', errors='ignore')
            doc_type = 'text'
        
        # Clean text
        text = self.clean_text(text)
        
        # Create chunks with metadata
        metadata = {
            'source_url': url,
            'document_type': doc_type,
            'filename': filename
        }
        
        chunks = self.create_chunks(text, metadata)
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks

# Singleton instance
document_processor = DocumentProcessor()
