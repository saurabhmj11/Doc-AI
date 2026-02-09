"""
Vector Store Module

ChromaDB-based vector storage and retrieval for document chunks.
Imports are deferred until first use to prevent blocking server startup.
"""

from typing import Optional, TYPE_CHECKING
import os

from config import get_settings

# Use TYPE_CHECKING to avoid importing document_processor at module load time
# This prevents heavy imports (fitz, docx, sentence_transformers) from blocking startup
if TYPE_CHECKING:
    from core.document_processor import DocumentChunk

settings = get_settings()


class VectorStore:
    """ChromaDB-based vector store for document embeddings."""
    
    def __init__(self):
        # Lazy import chromadb - don't block server startup
        import chromadb
        
        # Ensure persist directory exists
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
        
        # Use PersistentClient for ChromaDB 0.4.x+
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )
        
        # Document metadata storage (in-memory for POC)
        self._document_metadata: dict[str, dict] = {}
    
    def add_document(
        self, 
        document_id: str, 
        chunks: "list[DocumentChunk]",
        filename: str,
        file_type: str,
        file_path: str
    ) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            document_id: Unique document identifier
            chunks: List of DocumentChunk objects with embeddings
            filename: Original filename
            file_type: File extension
            file_path: Path to stored file
        """
        # Create or get collection for this document
        collection = self.client.get_or_create_collection(
            name=f"doc_{document_id.replace('-', '_')}",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "page": chunk.page or 0,
                "chunk_index": chunk.chunk_index
            }
            for chunk in chunks
        ]
        
        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        # Store document metadata
        self._document_metadata[document_id] = {
            "filename": filename,
            "file_type": file_type,
            "file_path": file_path,
            "chunk_count": len(chunks),
            "collection_name": f"doc_{document_id.replace('-', '_')}"
        }
    
    def search(
        self, 
        document_id: str, 
        query_embedding: list[float], 
        top_k: int = 10
    ) -> list[dict]:
        """
        Search for similar chunks in a document.
        
        Args:
            document_id: Document to search in
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of dicts with text, metadata, and similarity score
        """
        if document_id not in self._document_metadata:
            raise ValueError(f"Document {document_id} not found")
        
        collection_name = self._document_metadata[document_id]["collection_name"]
        collection = self.client.get_collection(collection_name)
        
        # Ensure query_embedding is a list
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, 100),  # Limit to avoid warning
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert to list of dicts
        search_results = []
        for i in range(len(results["ids"][0])):
            # ChromaDB with cosine space returns distance where 0 = identical, 2 = opposite
            # Convert to similarity score (0-1 range)
            distance = results["distances"][0][i]
            # Cosine distance is in [0, 2], convert to similarity [0, 1]
            similarity = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
            
            search_results.append({
                "chunk_id": results["ids"][0][i],
                "document_id": document_id,
                "filename": self._document_metadata[document_id]["filename"],
                "text": results["documents"][0][i],
                "page": results["metadatas"][0][i].get("page"),
                "chunk_index": results["metadatas"][0][i].get("chunk_index"),
                "similarity_score": similarity
            })
        
        return search_results
    
    def search_parallel(
        self, 
        document_ids: list[str], 
        query_embedding: list[float], 
        top_k: int = 5
    ) -> list[dict]:
        """
        Search multiple documents in parallel using threads.
        Merges and re-ranks results by similarity.
        """
        import concurrent.futures
        
        all_results = []
        
        # Define the work function
        def _search_single(doc_id):
            try:
                if not self.document_exists(doc_id):
                    return []
                return self.search(doc_id, query_embedding, top_k)
            except Exception as e:
                print(f"Error searching doc {doc_id}: {e}")
                return []

        # Execute in parallel
        # ChromaDB client is thread-safe for read operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(document_ids))) as executor:
            future_to_doc = {executor.submit(_search_single, doc_id): doc_id for doc_id in document_ids}
            for future in concurrent.futures.as_completed(future_to_doc):
                results = future.result()
                all_results.extend(results)
        
        # Sort combined results by similarity score (descending)
        all_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Return top K from all docs combined (e.g. top 10 overall)
        # Maybe allow more to give broad context? Let's say top_k * 2
        return all_results[:top_k * 2]
    
    def get_document_metadata(self, document_id: str) -> Optional[dict]:
        """Get metadata for a document."""
        return self._document_metadata.get(document_id)
    
    def get_all_chunks(self, document_id: str) -> list[dict]:
        """Get all chunks for a document (for full-text extraction)."""
        if document_id not in self._document_metadata:
            raise ValueError(f"Document {document_id} not found")
        
        collection_name = self._document_metadata[document_id]["collection_name"]
        collection = self.client.get_collection(collection_name)
        
        # Get all items
        results = collection.get(include=["documents", "metadatas"])
        
        chunks = []
        for i in range(len(results["ids"])):
            chunks.append({
                "chunk_id": results["ids"][i],
                "text": results["documents"][i],
                "page": results["metadatas"][i].get("page"),
                "chunk_index": results["metadatas"][i].get("chunk_index")
            })
        
        # Sort by chunk index
        chunks.sort(key=lambda x: x["chunk_index"])
        return chunks
    
    def document_exists(self, document_id: str) -> bool:
        """Check if a document exists in the store."""
        return document_id in self._document_metadata
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its collection."""
        if document_id not in self._document_metadata:
            return False
        
        collection_name = self._document_metadata[document_id]["collection_name"]
        self.client.delete_collection(collection_name)
        del self._document_metadata[document_id]
        return True


# Singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create vector store singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
