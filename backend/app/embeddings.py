"""Local ONNX sentence embeddings (fastembed, BAAI/bge-small-en-v1.5, 384-d). Shared by ingestion and API."""
from __future__ import annotations

import threading

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DIM = 384


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: str | None = None,
                 threads: int | None = None):
        from fastembed import TextEmbedding  # heavy import kept lazy

        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir, threads=threads)
        self._lock = threading.Lock()

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts, batch_size=batch_size)]

    def embed_query(self, text: str) -> list[float]:
        with self._lock:  # onnxruntime session is shared; keep single-request use simple
            return next(iter(self.model.embed([QUERY_PREFIX + text]))).tolist()
