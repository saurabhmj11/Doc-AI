"""
Document processing - handles PDFs, Word docs, and plain text

The chunking logic here tries to be smart about keeping paragraphs
together rather than just splitting at arbitrary character counts.
Took some trial and error to get this working well with logistics docs.
"""

import os
import uuid
import fitz  # PyMuPDF
from docx import Document
import chardet
from typing import Optional
from dataclasses import dataclass

from config import get_settings
from core.model_loader import get_embedding_model

settings = get_settings()


@dataclass
class DocumentChunk:
    """A piece of a document with its metadata"""
    chunk_id: str
    document_id: str
    text: str
    page: Optional[int]
    chunk_index: int
    embedding: Optional[list[float]] = None


class DocumentProcessor:
    """
    Takes uploaded files and turns them into searchable chunks.
    
    Supports: PDF, DOCX, TXT
    """
    
    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
    
    def process_file(self, file_path: str, file_type: str) -> tuple[str, list[DocumentChunk]]:
        """
        Main processing function.
        Give it a file, get back chunks with embeddings ready for search.
        """
        doc_id = str(uuid.uuid4())
        
        # parse based on file type
        if file_type == "pdf":
            pages = self._parse_pdf(file_path)
        elif file_type == "docx":
            pages = self._parse_docx(file_path)
        elif file_type == "txt":
            pages = self._parse_txt(file_path)
        else:
            raise ValueError(f"Can't handle {file_type} files")
        
        # chunk it up
        chunks = self._create_chunks(doc_id, pages)
        
        # embed everything
        chunks = self._generate_embeddings(chunks)
        
        return doc_id, chunks
    
    def _parse_pdf(self, file_path: str) -> list[tuple[int, str]]:
        """Pull text from PDF pages"""
        pages = []
        with fitz.open(file_path) as doc:
            for i, page in enumerate(doc, 1):
                text = page.get_text("text")
                if text.strip():
                    pages.append((i, text))
        return pages
    
    def _parse_docx(self, file_path: str) -> list[tuple[int, str]]:
        """Extract text from Word doc"""
        doc = Document(file_path)
        
        text_parts = []
        
        # get paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)
        
        # don't forget tables - important for rate sheets
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                if row_text:
                    text_parts.append(row_text)
        
        # word docs don't have page numbers, just treat as one page
        full_text = "\n".join(text_parts)
        return [(1, full_text)] if full_text else []
    
    def _parse_txt(self, file_path: str) -> list[tuple[int, str]]:
        """Read text file with auto encoding detection"""
        with open(file_path, 'rb') as f:
            raw = f.read()
        
        # figure out the encoding
        detected = chardet.detect(raw)
        enc = detected.get('encoding', 'utf-8') or 'utf-8'
        
        text = raw.decode(enc, errors='replace')
        return [(1, text)] if text.strip() else []
    
    def _create_chunks(self, doc_id: str, pages: list[tuple[int, str]]) -> list[DocumentChunk]:
        """
        Smart chunking - tries to keep paragraphs together.
        Uses overlap so we don't lose context at chunk boundaries.
        """
        chunks = []
        idx = 0
        
        for page_num, page_text in pages:
            paragraphs = self._split_paragraphs(page_text)
            current = ""
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # check if adding this would make chunk too big
                # rough estimate: ~4 chars per token
                if len(current) + len(para) > self.chunk_size * 4:
                    if current:
                        chunks.append(DocumentChunk(
                            chunk_id=f"{doc_id}_{idx}",
                            document_id=doc_id,
                            text=current.strip(),
                            page=page_num,
                            chunk_index=idx
                        ))
                        idx += 1
                        
                        # keep some overlap for context
                        overlap_chars = self.chunk_overlap * 4
                        overlap = current[-overlap_chars:] if len(current) > overlap_chars else ""
                        current = overlap + " " + para
                else:
                    current = current + "\n" + para if current else para
            
            # don't forget leftover text
            if current.strip():
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc_id}_{idx}",
                    document_id=doc_id,
                    text=current.strip(),
                    page=page_num,
                    chunk_index=idx
                ))
                idx += 1
        
        return chunks
    
    def _split_paragraphs(self, text: str) -> list[str]:
        """Break text into paragraphs - handles various formats"""
        # try double newlines first (standard paragraphs)
        paras = text.split('\n\n')
        
        # if that didn't work, try single newlines but be smart about it
        if len(paras) == 1:
            lines = text.split('\n')
            paras = []
            current = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    if current:
                        paras.append(' '.join(current))
                        current = []
                else:
                    current.append(line)
            
            if current:
                paras.append(' '.join(current))
        
        return paras
    
    def _generate_embeddings(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Batch embed all chunks at once - much faster than one at a time"""
        if not chunks:
            return chunks
        
        texts = [c.text for c in chunks]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb.tolist()
        
        return chunks
    
    def get_full_text(self, file_path: str, file_type: str) -> str:
        """Get all text from a doc - used for structured extraction"""
        if file_type == "pdf":
            pages = self._parse_pdf(file_path)
        elif file_type == "docx":
            pages = self._parse_docx(file_path)
        elif file_type == "txt":
            pages = self._parse_txt(file_path)
        else:
            raise ValueError(f"Unsupported: {file_type}")
        
        return "\n\n".join(text for _, text in pages)
