"""
Shared Model Loader

Lazy loading for expensive ML models to prevent blocking server startup.
All imports are deferred until first use.
"""

from config import get_settings

settings = get_settings()

# Global singleton instance
_embedding_model = None


def get_embedding_model():
    """Get or create shared embedding model singleton with lazy imports."""
    global _embedding_model
    if _embedding_model is None:
        if settings.llm_mode == "offline":
            print(f"Loading embedding model (Ollama: {settings.ollama_embedding_model})...")
            import ollama
            
            class OllamaEmbeddingModel:
                def __init__(self, model_name, base_url):
                    self.model_name = model_name
                    self.client = ollama.Client(host=base_url)
                    
                def encode(self, texts, convert_to_numpy=True, task_type="retrieval_document"):
                    is_single = isinstance(texts, str)
                    if is_single:
                        texts = [texts]
                    
                    embeddings = []
                    for text in texts:
                        try:
                            resp = self.client.embeddings(model=self.model_name, prompt=text)
                            embeddings.append(resp['embedding'])
                        except Exception as e:
                            print(f"Error embedding with Ollama: {e}")
                            # Fallback with zeros (assuming 768 dim for nomic-embed-text, but could vary)
                            # Ideally we should know the dim, but 768 is a safe default for now
                            embeddings.append([0.0] * 768)

                    if is_single and embeddings:
                        if convert_to_numpy:
                            import numpy as np
                            return np.array(embeddings[0])
                        return embeddings[0]

                    if convert_to_numpy:
                        import numpy as np
                        return np.array(embeddings)
                    return embeddings
            
            _embedding_model = OllamaEmbeddingModel(settings.ollama_embedding_model, settings.ollama_base_url)
            print("Ollama embedding model loaded!")
            
        else:
            print("Loading embedding model (Gemini API)...")
            # Lightweight wrapper for Gemini Embeddings
            import google.generativeai as genai
            import numpy as np
            
            class GeminiEmbeddingModel:
                def __init__(self, api_key):
                    genai.configure(api_key=api_key)
                    # Use the specific model available to the user's key
                    self.model_name = "models/gemini-embedding-001"
                    
                def encode(self, texts, convert_to_numpy=True, task_type="retrieval_document"):
                    # Batch embed using Gemini API
                    is_single = isinstance(texts, str)
                    if is_single:
                        texts = [texts]
                        
                    embeddings = []
                    # Process in small batches to be safe
                    batch_size = 20 
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i+batch_size]
                        try:
                            result = genai.embed_content(
                                model=self.model_name,
                                content=batch,
                                task_type=task_type
                            )
                            # Result is a dict with 'embedding' key which is a list
                            batch_embeddings = result['embedding']
                            
                            # DEBUG: Print structure if empty or single
                            if not batch_embeddings:
                                 print(f"DEBUG: Empty embeddings returned for batch of size {len(batch)}")
                            elif batch_embeddings and isinstance(batch_embeddings[0], float):
                                 # Should be list of list of float, not list of float
                                 print(f"DEBUG: Found list of float instead of list of lists? Length: {len(batch_embeddings)}")
                                 # If we sent a list of strings but got a single embedding back?
                                 if len(batch) == 1:
                                      batch_embeddings = [batch_embeddings]
                                 
                            embeddings.extend(batch_embeddings)
                        except Exception as e:
                            print(f"Error embedding batch: {e}")
                            # Fallback with zeros
                            embeddings.extend([[0.0] * 768] * len(batch))
                    
                    # Unwrap if it was a single string request
                    if is_single and embeddings:
                        if convert_to_numpy:
                            return np.array(embeddings[0])
                        return embeddings[0]

                    if convert_to_numpy:
                        return np.array(embeddings)
                    return embeddings

            _embedding_model = GeminiEmbeddingModel(settings.gemini_api_key)
            print("Gemini embedding model loaded!")
    return _embedding_model
