from __future__ import annotations

from typing import Protocol

import torch
from sentence_transformers import SentenceTransformer

from litsearch.config import settings


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class SentenceTransformerEmbeddingProvider:
    def __init__(self, model_name: str | None = None, device: str | None = None, normalize: bool | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self.normalize = settings.embedding_normalize if normalize is None else normalize
        # OMP_NUM_THREADS alone isn't reliably honored by torch's CPU thread pool (backend/build
        # dependent); setting it explicitly is the only mechanism guaranteed to take effect.
        if settings.torch_num_threads:
            torch.set_num_threads(settings.torch_num_threads)
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(list(texts), batch_size=settings.embedding_batch_size, normalize_embeddings=self.normalize)
        # model.encode already normalizes when self.normalize is set, so skip the redundant per-vector renormalization pass.
        return [list(map(float, emb)) for emb in embeddings]

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=self.normalize)
        if isinstance(vector, list):
            return [float(v) for v in vector]
        return [float(v) for v in list(vector)]
