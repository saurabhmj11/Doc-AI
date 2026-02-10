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

    if _embedding_model is not None:
        return _embedding_model

    # ==========================================================
    # OFFLINE MODE (OLLAMA)
    # ==========================================================
    if settings.llm_mode == "offline":

        print(f"Loading embedding model (Ollama: {settings.ollama_embedding_model})...")

        import ollama
        import numpy as np

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
                        # ✅ Correct Ollama embeddings API
                        resp = self.client.embeddings(
                            model=self.model_name,
                            input=text
                        )

                        embedding = resp.get("embedding")

                        # ✅ Validate embedding
                        if not embedding or not isinstance(embedding, list):
                            raise RuntimeError("Invalid embedding response from Ollama")

                        embeddings.append(embedding)

                    except Exception as e:
                        print(f"Error embedding with Ollama: {e}")
                        raise RuntimeError(f"Ollama embedding failed: {e}")

                # Return format handling
                if is_single:
                    return np.array(embeddings[0]) if convert_to_numpy else embeddings[0]

                return np.array(embeddings) if convert_to_numpy else embeddings

        _embedding_model = OllamaEmbeddingModel(
            settings.ollama_embedding_model,
            settings.ollama_base_url
        )

        print("Ollama embedding model loaded!")

    # ==========================================================
    # ONLINE MODE (GEMINI)
    # ==========================================================
    else:

        print("Loading embedding model (Gemini API)...")

        import google.generativeai as genai
        import numpy as np

        class GeminiEmbeddingModel:

            def __init__(self, api_key):
                if not api_key:
                    raise ValueError("Gemini API Key is missing. Please check your backend/.env file.")
                
                genai.configure(api_key=api_key)
                self.model_name = "models/gemini-embedding-001"

            def encode(self, texts, convert_to_numpy=True, task_type="retrieval_document"):

                is_single = isinstance(texts, str)
                if is_single:
                    texts = [texts]

                embeddings = []
                batch_size = 20

                for i in range(0, len(texts), batch_size):

                    batch = texts[i:i + batch_size]

                    try:
                        result = genai.embed_content(
                            model=self.model_name,
                            content=batch,
                            task_type=task_type
                        )

                        batch_embeddings = result.get("embedding")

                        # ✅ Validate response structure
                        if not batch_embeddings:
                            raise RuntimeError("Empty embeddings returned from Gemini")

                        # Handle edge case where single embedding returned
                        if isinstance(batch_embeddings[0], float):
                            if len(batch) == 1:
                                batch_embeddings = [batch_embeddings]
                            else:
                                raise RuntimeError("Unexpected embedding structure")

                        embeddings.extend(batch_embeddings)

                    except Exception as e:
                        print(f"Error embedding batch: {e}")
                        raise RuntimeError(f"Embedding generation failed: {e}")

                if is_single:
                    return np.array(embeddings[0]) if convert_to_numpy else embeddings[0]

                return np.array(embeddings) if convert_to_numpy else embeddings

        _embedding_model = GeminiEmbeddingModel(settings.gemini_api_key)

        print("Gemini embedding model loaded!")

    return _embedding_model
