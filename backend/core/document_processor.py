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

    def __init__(self):

        self._embedding_model = None
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    # ==============================================
    # PDF PARSER (LAYOUT AWARE)
    # ==============================================

    def _parse_pdf(self, file_path):

        pages = []

        with fitz.open(file_path) as doc:

            for i, page in enumerate(doc, 1):

                blocks = page.get_text("blocks")

                blocks.sort(key=lambda b: (b[1], b[0]))

                text = "\n".join(b[4] for b in blocks)

                text = text.replace("Shipper Consignee", "Shipper:\n")

                if text.strip():
                    pages.append((i, text))

        return pages

    # ==============================================

    def _create_chunks(self, doc_id, pages):

        chunks = []
        idx = 0

        current = ""   # FIXED: moved outside page loop

        for page_num, page_text in pages:

            paragraphs = self._split_paragraphs(page_text)

            for para in paragraphs:

                if len(current.split()) + len(para.split()) > self.chunk_size:

                    if current:

                        chunks.append(DocumentChunk(
                            chunk_id=f"{doc_id}_{idx}",
                            document_id=doc_id,
                            text=current.strip(),
                            page=page_num,
                            chunk_index=idx
                        ))

                        idx += 1

                        # SENTENCE AWARE OVERLAP
                        sentences = current.split('.')
                        overlap = '.'.join(sentences[-2:])
                        current = overlap + " " + para

                else:
                    current += "\n" + para if current else para

        if current.strip():

            chunks.append(DocumentChunk(
                chunk_id=f"{doc_id}_{idx}",
                document_id=doc_id,
                text=current.strip(),
                page=page_num,
                chunk_index=idx
            ))

        return chunks

    # ==============================================

    def _split_paragraphs(self, text):

        paras = []

        lines = text.split('\n')

        current = []

        for line in lines:

            line = line.strip()

            if not line:
                if current:
                    paras.append(" ".join(current))
                    current = []
                continue

            # TABLE LINE DETECTION
            if "|" in line:
                paras.append(line)
                continue

            current.append(line)

        if current:
            paras.append(" ".join(current))

        return paras

    # ==============================================

    def _generate_embeddings(self, chunks):

        texts = [c.text for c in chunks]

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True
        )

        if embeddings is None:
            raise RuntimeError("Embedding generation failed")

        for chunk, emb in zip(chunks, embeddings):

            if emb is None:
                raise RuntimeError("Invalid embedding returned")

            chunk.embedding = emb.tolist()

        return chunks

    # ==============================================

    def _parse_docx(self, file_path):

        doc = Document(file_path)
        text_parts = []

        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)

        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                row_text = " | ".join(cells)
                if row_text:
                    text_parts.append(row_text)

        full_text = "\n".join(text_parts)
        return [(1, full_text)] if full_text else []

    # ==============================================

    def _parse_txt(self, file_path):

        with open(file_path, 'rb') as f:
            raw = f.read()

        detected = chardet.detect(raw)
        enc = detected.get('encoding', 'utf-8') or 'utf-8'

        text = raw.decode(enc, errors='replace')
        return [(1, text)] if text.strip() else []

    # ==============================================

    def get_full_text(self, file_path, file_type):

        if file_type == "pdf":
            pages = self._parse_pdf(file_path)
        elif file_type == "docx":
            pages = self._parse_docx(file_path)
        elif file_type == "txt":
            pages = self._parse_txt(file_path)
        else:
            raise ValueError(f"Unsupported type: {file_type}")

        return "\n\n".join(text for _, text in pages)

    # ==============================================

    def process_file(self, file_path, file_type):

        # 1. Parse
        if file_type == "pdf":
            pages = self._parse_pdf(file_path)
        elif file_type == "docx":
            pages = self._parse_docx(file_path)
        elif file_type == "txt":
            pages = self._parse_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # 2. Chunk
        doc_id = str(uuid.uuid4())
        chunks = self._create_chunks(doc_id, pages)

        # 3. Embed
        chunks = self._generate_embeddings(chunks)

        return doc_id, chunks
