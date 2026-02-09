"""
API Routes Module

FastAPI endpoints for document upload, Q&A, and structured extraction.
"""

import os
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional

from config import get_settings
from api.schemas import (
    DocumentUploadResponse,
    AskRequest,
    AskResponse,
    SourceChunk,
    ExtractRequest,
    ExtractResponse,
    ErrorResponse
)

settings = get_settings()
router = APIRouter()

# Upload directory
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Lazy-loaded singletons to prevent blocking server startup
_document_processor = None
_vector_store = None
_rag_pipeline = None
_structured_extractor = None


def get_document_processor():
    """Lazy-load DocumentProcessor to avoid import-time heavy loading."""
    global _document_processor
    if _document_processor is None:
        from core.document_processor import DocumentProcessor
        _document_processor = DocumentProcessor()
    return _document_processor


def get_vector_store():
    """Lazy-load VectorStore."""
    global _vector_store
    if _vector_store is None:
        from core.vector_store import get_vector_store as _get_vs
        _vector_store = _get_vs()
    return _vector_store


def get_rag_pipeline():
    """Lazy-load RAGPipeline."""
    global _rag_pipeline
    if _rag_pipeline is None:
        from core.rag_pipeline import get_rag_pipeline as _get_rag
        _rag_pipeline = _get_rag()
    return _rag_pipeline


def get_structured_extractor():
    """Lazy-load StructuredExtractor."""
    global _structured_extractor
    if _structured_extractor is None:
        from core.structured_extractor import get_structured_extractor as _get_ext
        _structured_extractor = _get_ext()
    return _structured_extractor


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a logistics document.
    
    Accepts PDF, DOCX, or TXT files.
    Returns document ID for subsequent queries.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Get file extension
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {settings.allowed_extensions}"
        )
    
    # Check file size
    content = await file.read()
    if len(content) > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
        )
    
    # Save file temporarily
    temp_path = UPLOAD_DIR / file.filename
    try:
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Process document
        document_id, chunks = get_document_processor().process_file(
            str(temp_path),
            file_ext
        )
        
        # Store in vector database
        vector_store = get_vector_store()
        vector_store.add_document(
            document_id=document_id,
            chunks=chunks,
            filename=file.filename,
            file_type=file_ext,
            file_path=str(temp_path)
        )
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            file_type=file_ext,
            chunks_created=len(chunks),
            status="processed",
            message=f"Document processed successfully. {len(chunks)} chunks created."
        )
        
    except Exception as e:
        # Clean up on error
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question about an uploaded document.
    
    Returns answer with confidence score, sources, and guardrail status.
    """
    # Validate document exists
    vector_store = get_vector_store()
    if not vector_store.document_exists(request.document_id):
        raise HTTPException(
            status_code=404,
            detail=f"Document {request.document_id} not found. Please upload a document first."
        )
    
    try:
        # Get answer from RAG pipeline
        rag = get_rag_pipeline()
        result = rag.ask(
            document_id=request.document_id,
            question=request.question
        )
        
        # Format sources
        sources = [
            SourceChunk(
                text=s["text"],
                page=s.get("page"),
                chunk_id=s["chunk_id"],
                similarity_score=s["similarity_score"]
            )
            for s in result.get("sources", [])
        ]
        
        return AskResponse(
            answer=result["answer"],
            confidence=result["confidence"],
            confidence_level=result["confidence_level"],
            sources=sources,
            guardrail_status=result["guardrail_status"],
            guardrail_message=result.get("guardrail_message")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@router.post("/extract", response_model=ExtractResponse)
async def extract_structured_data(request: ExtractRequest):
    """
    Extract structured shipment data from an uploaded document.
    
    Returns JSON with shipment details and nulls for missing fields.
    """
    # Validate document exists
    vector_store = get_vector_store()
    if not vector_store.document_exists(request.document_id):
        raise HTTPException(
            status_code=404,
            detail=f"Document {request.document_id} not found. Please upload a document first."
        )
    
    try:
        # Get document metadata
        metadata = vector_store.get_document_metadata(request.document_id)
        file_path = metadata["file_path"]
        file_type = metadata["file_type"]
        
        # Get full document text
        full_text = get_document_processor().get_full_text(file_path, file_type)
        
        # Extract structured data
        extractor = get_structured_extractor()
        shipment_data, confidence = extractor.extract(full_text)
        
        # Count non-null fields
        fields = [
            shipment_data.shipment_id,
            shipment_data.shipper,
            shipment_data.consignee,
            shipment_data.pickup_datetime,
            shipment_data.delivery_datetime,
            shipment_data.equipment_type,
            shipment_data.mode,
            shipment_data.rate,
            shipment_data.currency,
            shipment_data.weight,
            shipment_data.carrier_name
        ]
        fields_found = sum(1 for f in fields if f is not None)
        
        return ExtractResponse(
            document_id=request.document_id,
            extraction=shipment_data,
            extraction_confidence=round(confidence, 3),
            fields_found=fields_found,
            total_fields=11
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting data: {str(e)}")


@router.get("/documents/{document_id}")
async def get_document_info(document_id: str):
    """Get information about an uploaded document."""
    vector_store = get_vector_store()
    
    if not vector_store.document_exists(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    
    metadata = vector_store.get_document_metadata(document_id)
    return {
        "document_id": document_id,
        "filename": metadata["filename"],
        "file_type": metadata["file_type"],
        "chunks": metadata["chunk_count"]
    }


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete an uploaded document."""
    vector_store = get_vector_store()
    
    if not vector_store.document_exists(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get file path before deleting from vector store
    metadata = vector_store.get_document_metadata(document_id)
    file_path = Path(metadata["file_path"])
    
    # Delete from vector store
    vector_store.delete_document(document_id)
    
    # Delete file
    if file_path.exists():
        file_path.unlink()
    
    return {"status": "deleted", "document_id": document_id}
