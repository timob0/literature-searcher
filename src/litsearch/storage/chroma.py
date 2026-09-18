from __future__ import annotations

from typing import Any

import chromadb

from litsearch.config import settings


class ChromaVectorStore:
    def __init__(self, host: str | None = None, port: int | None = None, collection_name: str | None = None) -> None:
        self.host = host or settings.chroma_host
        self.port = port or settings.chroma_port
        self.collection_name = collection_name or settings.chroma_collection
        self.client = chromadb.HttpClient(host=self.host, port=self.port)
        self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict[str, Any]] | None = None, embeddings: list[list[float]] | None = None) -> None:
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas or [{} for _ in texts], embeddings=embeddings)

    def query(self, embedding: list[float], n_results: int = 10, where: dict[str, Any] | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)

    def count(self) -> int:
        return self.collection.count()

    def delete_by_ids(self, ids: list[str]) -> None:
        if ids:
            self.collection.delete(ids=ids)

    def ids_for_attachment(self, attachment_key: str) -> list[str]:
        result = self.collection.get(where={"attachment_key": attachment_key}, include=["metadatas"])
        return [str(chunk_id) for chunk_id in result.get("ids", [])]

    def replace_attachment(self, *, old_ids: list[str], ids: list[str], texts: list[str], metadatas: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        stale_ids = [chunk_id for chunk_id in old_ids if chunk_id not in ids]
        self.delete_by_ids(stale_ids)

    def reset(self) -> None:
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            # The local server can retain stale collection metadata after its
            # storage directory has been removed outside the Chroma process.
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})
