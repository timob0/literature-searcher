from __future__ import annotations

from litsearch.indexing.embedder import SentenceTransformerEmbeddingProvider
from litsearch.retrieval.models import ChunkHit, SearchFilters
from litsearch.storage.chroma import ChromaVectorStore


class SimpleRetriever:
    def __init__(self, embedding_provider: SentenceTransformerEmbeddingProvider | None = None, vector_store: ChromaVectorStore | None = None) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

        if self.embedding_provider is None:
            try:
                self.embedding_provider = SentenceTransformerEmbeddingProvider()
            except Exception:
                self.embedding_provider = None
        if self.vector_store is None:
            try:
                self.vector_store = ChromaVectorStore()
            except Exception:
                self.vector_store = None

    def retrieve(self, query: str, filters: SearchFilters | None = None, top_k: int = 10) -> list[ChunkHit]:
        if not self.embedding_provider or not self.vector_store:
            return []
        try:
            query_embedding = self.embedding_provider.embed_query(query)
            where = None
            if filters is not None:
                conditions: list[dict] = []
                for field in ("parent_key", "collection", "item_type"):
                    value = getattr(filters, field, None)
                    if value is not None:
                        conditions.append({field: value})
                if filters.citation_key:
                    sources = [source.strip() for source in filters.citation_key.split(",") if source.strip()]
                    if len(sources) == 1:
                        conditions.append({"citation_key": sources[0]})
                    elif sources:
                        conditions.append({"$or": [{"citation_key": source} for source in sources]})
                if filters.author:
                    authors = [author.strip() for author in filters.author.split(",") if author.strip()]
                    if len(authors) == 1:
                        conditions.append({"authors": {"$contains": authors[0]}})
                    elif authors:
                        conditions.append({"$or": [{"authors": {"$contains": author}} for author in authors]})
                if conditions:
                    where = conditions[0] if len(conditions) == 1 else {"$and": conditions}
            result = self.vector_store.query(query_embedding, n_results=top_k, where=where)
        except Exception:
            return []

        return self._hits_from_result(result)

    def related_chunks(self, anchor: ChunkHit) -> list[ChunkHit]:
        if not self.vector_store:
            return [anchor]
        try:
            result = self.vector_store.collection.get(
                where={"attachment_key": anchor.attachment_key},
                include=["documents", "metadatas"],
            )
            return self._hits_from_result(result)
        except Exception:
            return [anchor]

    def _hits_from_result(self, result: dict) -> list[ChunkHit]:
        hits: list[ChunkHit] = []
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        distances = result.get("distances", [])
        if documents and isinstance(documents[0], list):
            documents = documents[0]
        if metadatas and isinstance(metadatas[0], list):
            metadatas = metadatas[0]
        if distances and isinstance(distances[0], list):
            distances = distances[0]
        if isinstance(documents, str):
            documents = [documents]
        if metadatas and isinstance(metadatas, dict):
            metadatas = [metadatas]
        if distances and isinstance(distances, (int, float)):
            distances = [distances]
        for idx, text in enumerate(documents):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            text = self._clean_text(text)
            score = 1.0 - float(distances[idx]) if idx < len(distances) else 0.0
            hits.append(
                ChunkHit(
                    chunk_id=str(metadata.get("chunk_id", f"chunk-{idx}")),
                    attachment_key=str(metadata.get("attachment_key", "")),
                    parent_key=str(metadata.get("parent_key", "")),
                    citation_key=self._clean_optional_text(metadata.get("citation_key")),
                    title=self._clean_optional_text(metadata.get("title")),
                    authors=[self._clean_text(metadata.get("authors", ""))],
                    year=metadata.get("year"),
                    doi=self._clean_optional_text(metadata.get("doi")),
                    page_start=int(metadata.get("page_start", 0)),
                    page_end=int(metadata.get("page_end", 0)),
                    section_type=metadata.get("section_type"),
                    section_title=metadata.get("section_title"),
                    text=text,
                    score=score,
                    zotero_link="zotero://open-pdf/library/items/{}?page={}".format(metadata.get("attachment_key", ""), metadata.get("page_start", 1)),
                )
            )
        return hits

    @staticmethod
    def _clean_text(value: object) -> str:
        return str(value or "").replace("\x00", "")

    @classmethod
    def _clean_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        return cls._clean_text(value)
