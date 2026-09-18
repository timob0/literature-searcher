from __future__ import annotations

from dataclasses import dataclass

from litsearch.indexing.incremental import INDEX_PIPELINE_VERSION
from litsearch.indexing.manifest import IndexManifest
from litsearch.config import settings
from litsearch.storage.chroma import ChromaVectorStore


@dataclass(slots=True)
class IndexAudit:
    documents: int = 0
    chunks: int = 0
    oversized_chunks: int = 0
    invalid_page_ranges: int = 0
    all_page_one_anomalies: int = 0
    manifest_page_mismatches: int = 0
    potential_duplicate_chunks: int = 0
    stale_pipeline_documents: int = 0

    @classmethod
    def run(cls, manifest: IndexManifest, vector_store: ChromaVectorStore) -> "IndexAudit":
        audit = cls()
        entries = {entry.attachment_key: entry for entry in manifest.all_entries() if entry.index_status == "indexed"}
        audit.documents = len(entries)
        result = vector_store.collection.get(include=["documents", "metadatas"])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        audit.chunks = len(metadatas)
        by_attachment: dict[str, list[dict]] = {}
        duplicate_text: set[tuple[str, str]] = set()
        seen_text: set[tuple[str, str]] = set()
        for text, metadata in zip(documents, metadatas):
            attachment_key = str(metadata.get("attachment_key", ""))
            by_attachment.setdefault(attachment_key, []).append(metadata)
            if len(str(text).split()) > settings.chunk_max_tokens:
                audit.oversized_chunks += 1
            start = int(metadata.get("page_start", 0))
            end = int(metadata.get("page_end", 0))
            entry = entries.get(attachment_key)
            if start < 1 or end < start or (entry and entry.number_of_pages and end > entry.number_of_pages):
                audit.invalid_page_ranges += 1
            text_key = (attachment_key, str(text))
            if text_key in seen_text:
                duplicate_text.add(text_key)
            seen_text.add(text_key)
        audit.potential_duplicate_chunks = len(duplicate_text)
        for attachment_key, entry in entries.items():
            chunks = by_attachment.get(attachment_key, [])
            if len(chunks) != (entry.number_of_chunks or 0):
                audit.manifest_page_mismatches += 1
            if entry.pipeline_version != INDEX_PIPELINE_VERSION:
                audit.stale_pipeline_documents += 1
            if entry.number_of_pages and len(chunks) > 5 and entry.number_of_pages > 3 and all(
                int(metadata.get("page_start", 0)) == 1 and int(metadata.get("page_end", 0)) == 1 for metadata in chunks
            ):
                audit.all_page_one_anomalies += 1
        return audit

    @property
    def passed(self) -> bool:
        return not any((self.oversized_chunks, self.invalid_page_ranges, self.all_page_one_anomalies, self.manifest_page_mismatches, self.stale_pipeline_documents, self.potential_duplicate_chunks))