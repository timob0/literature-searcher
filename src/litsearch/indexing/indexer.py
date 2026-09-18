from __future__ import annotations

from pathlib import Path

from litsearch.documents.pdf import extract_pdf_text
from litsearch.indexing.chunker import SimpleChunker
from litsearch.indexing.embedder import SentenceTransformerEmbeddingProvider
from litsearch.storage.chroma import ChromaVectorStore


class PDFIndexer:
    pipeline_version = "2"

    def __init__(self, embedding_provider: SentenceTransformerEmbeddingProvider | None = None, vector_store: ChromaVectorStore | None = None, chunker: SimpleChunker | None = None) -> None:
        self.embedding_provider = embedding_provider or SentenceTransformerEmbeddingProvider()
        self.vector_store = vector_store or ChromaVectorStore()
        self.chunker = chunker or SimpleChunker()

    def index_pdf(self, pdf_path: str, *, attachment_key: str, parent_key: str, citation_key: str | None, title: str | None = None, authors: list[str] | None = None) -> list[str]:
        chunks, number_of_pages = self.extract_chunks(pdf_path, attachment_key=attachment_key, parent_key=parent_key, citation_key=citation_key)
        self.last_number_of_pages = number_of_pages
        return self.store_chunks(chunks, attachment_key=attachment_key, title=title, authors=authors)

    def index_text(self, text: str, *, attachment_key: str, parent_key: str, citation_key: str, title: str | None = None, authors: list[str] | None = None) -> list[str]:
        self.last_number_of_pages = 0
        chunks = self.extract_text_chunks(text, attachment_key=attachment_key, parent_key=parent_key, citation_key=citation_key)
        return self.store_chunks(chunks, attachment_key=attachment_key, title=title, authors=authors)

    def extract_chunks(self, pdf_path: str, *, attachment_key: str, parent_key: str, citation_key: str | None) -> tuple[list, int]:
        """PDF parsing + chunking only (CPU-bound, no GPU/network access) -- safe to call from a worker thread."""
        extraction = extract_pdf_text(pdf_path)
        if extraction.error:
            raise ValueError(extraction.error)
        chunks = self.chunker.chunk_pages(extraction.pages, attachment_key=attachment_key, parent_key=parent_key, citation_key=citation_key)
        return chunks, len(extraction.pages)

    def extract_text_chunks(self, text: str, *, attachment_key: str, parent_key: str, citation_key: str | None) -> list:
        """Chunking for abstract-only sources -- safe to call from a worker thread."""
        return self.chunker.chunk_document(text, attachment_key=attachment_key, parent_key=parent_key, citation_key=citation_key)

    def store_chunks(self, chunks, *, attachment_key: str, title: str | None, authors: list[str] | None = None) -> list[str]:
        """Embedding + Chroma upsert -- keep single-threaded (shared GPU model / HTTP client)."""
        if not chunks:
            return []

        vectors = self.embedding_provider.embed_documents([chunk.text for chunk in chunks])
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [{
            "chunk_id": chunk.chunk_id,
            "attachment_key": chunk.attachment_key,
            "parent_key": chunk.parent_key,
            "citation_key": chunk.citation_key or "",
            "title": title or "",
            "authors": "; ".join(authors or []),
            "year": 0,
            "doi": "",
            "document_id": chunk.parent_key,
            "chunk_index": chunk.chunk_index,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "section_type": chunk.section_type,
            "section_title": chunk.section_title or "",
        } for chunk in chunks]
        old_ids = self.chunk_ids_for_attachment(attachment_key)
        if hasattr(self.vector_store, "replace_attachment"):
            self.vector_store.replace_attachment(
                old_ids=old_ids,
                ids=ids,
                texts=[chunk.text for chunk in chunks],
                metadatas=metadatas,
                embeddings=vectors,
            )
        else:
            self.vector_store.delete_by_ids(old_ids)
            self.vector_store.add(ids=ids, texts=[chunk.text for chunk in chunks], metadatas=metadatas, embeddings=vectors)
        return ids

    def chunk_ids_for_attachment(self, attachment_key: str) -> list[str]:
        if hasattr(self.vector_store, "ids_for_attachment"):
            return self.vector_store.ids_for_attachment(attachment_key)
        return []
