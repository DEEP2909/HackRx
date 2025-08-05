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
    """Optimized document processor with better chunking strategy"""
    
    def __init__(self):
        # Optimized chunking for better token efficiency
        self.chunk_size = 800        # Reduced from 1000
        self.chunk_overlap = 100     # Reduced from 200
        self.max_file_size = settings.max_file_size_bytes
        
        # Smart chunking parameters
        self.sentence_endings = r'[.!?]\s+'
        self.paragraph_separators = r'\n\s*\n'
        
    async def download_document(self, url: str) -> bytes:
        """Download document from URL with caching"""
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
        """Extract text from PDF with better formatting preservation"""
        text = ""
        
        try:
            # Use pdfplumber for better text extraction
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        # Clean and format text
                        page_text = self._clean_page_text(page_text)
                        text += page_text + "\n\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed, trying PyPDF2: {e}")
            
            # Fallback to PyPDF2
            try:
                pdf_file = io.BytesIO(content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        page_text = self._clean_page_text(page_text)
                        text += page_text + "\n\n"
            except Exception as e2:
                logger.error(f"PDF extraction failed: {e2}")
                raise
        
        return text.strip()
    
    def _clean_page_text(self, text: str) -> str:
        """Clean extracted text while preserving structure"""
        # Fix common PDF extraction issues
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'([a-z])([A-Z])', r'\1. \2', text)  # Add periods between sentences
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words
        text = re.sub(r'\n+', ' ', text)  # Remove extra newlines
        
        return text.strip()
    
    def extract_text_from_docx(self, content: bytes) -> str:
        """Extract text from DOCX content"""
        doc = Document(io.BytesIO(content))
        text_parts = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text.strip())
        
        # Extract from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    text_parts.append(' | '.join(row_text))
        
        return '\n\n'.join(text_parts)
    
    def extract_text_from_html(self, content: bytes) -> str:
        """Extract text from HTML/Email content"""
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "header", "footer"]):
            element.decompose()
        
        # Get text with better formatting
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def smart_chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Smart chunking that respects sentence and paragraph boundaries"""
        if metadata is None:
            metadata = {}
        
        chunks = []
        
        # First, try to split by paragraphs
        paragraphs = re.split(self.paragraph_separators, text)
        
        current_chunk = ""
        current_word_count = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            paragraph_words = len(paragraph.split())
            
            # If paragraph alone is too big, split by sentences
            if paragraph_words > self.chunk_size:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, metadata, len(chunks)))
                    current_chunk = ""
                    current_word_count = 0
                
                # Split large paragraph by sentences
                sentences = re.split(self.sentence_endings, paragraph)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    sentence_words = len(sentence.split())
                    
                    if current_word_count + sentence_words > self.chunk_size:
                        if current_chunk:
                            chunks.append(self._create_chunk(current_chunk, metadata, len(chunks)))
                        current_chunk = sentence
                        current_word_count = sentence_words
                    else:
                        current_chunk += (" " if current_chunk else "") + sentence
                        current_word_count += sentence_words
            
            # Normal paragraph processing
            elif current_word_count + paragraph_words > self.chunk_size:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, metadata, len(chunks)))
                current_chunk = paragraph
                current_word_count = paragraph_words
            else:
                # Add to current chunk
                current_chunk += ("\n\n" if current_chunk else "") + paragraph
                current_word_count += paragraph_words
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, metadata, len(chunks)))
        
        logger.info(f"Smart chunking created {len(chunks)} chunks")
        return chunks
    
    def _create_chunk(self, content: str, metadata: Dict[str, Any], index: int) -> DocumentChunk:
        """Create a document chunk with metadata"""
        return DocumentChunk(
            content=content.strip(),
            metadata={
                **metadata,
                'chunk_index': index,
                'word_count': len(content.split()),
                'char_count': len(content)
            }
        )
    
    async def process_document(self, url: str) -> List[DocumentChunk]:
        """Main method to process document from URL with optimizations"""
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
        
        if not text or len(text.strip()) < 50:
            raise ValueError("Document appears to be empty or too short")
        
        # Create chunks with metadata
        metadata = {
            'source_url': url,
            'document_type': doc_type,
            'filename': filename,
            'total_chars': len(text),
            'total_words': len(text.split())
        }
        
        # Use smart chunking
        chunks = self.smart_chunk_text(text, metadata)
        
        logger.info(f"Created {len(chunks)} optimized chunks from document")
        return chunks

# Singleton instance
document_processor = DocumentProcessor()
