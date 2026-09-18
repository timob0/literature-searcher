from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from litsearch.indexing.manifest import IndexManifest, ManifestEntry

INDEX_PIPELINE_VERSION = "2"


@dataclass(slots=True)
class SyncDecision:
    action: str
    attachment_key: str
    parent_key: str | None = None
    citation_key: str | None = None
    pdf_path: str | None = None


class IncrementalIndexer:
    def __init__(self, manifest: IndexManifest | None = None) -> None:
        self.manifest = manifest or IndexManifest()

    def should_index(self, attachment_key: str, parent_key: str | None, citation_key: str | None, pdf_path: str | None, pdf_size: int | None, pdf_mtime: float | None, pipeline_version: str = INDEX_PIPELINE_VERSION) -> str:
        entry = self.manifest.get_entry(attachment_key)
        if entry is None:
            return "new"
        if entry.pipeline_version != pipeline_version:
            return "changed"
        if pdf_path and pdf_path.startswith("abstract:"):
            if entry.pdf_size != pdf_size:
                return "changed"
            return "unchanged"
        if not pdf_path or not Path(pdf_path).exists():
            return "deleted"
        if entry.pdf_size != pdf_size or entry.pdf_mtime != pdf_mtime:
            return "changed"
        return "unchanged"

    def mark_indexed(self, attachment_key: str, parent_key: str | None, citation_key: str | None, pdf_path: str | None, pdf_size: int | None, pdf_mtime: float | None, *, number_of_pages: int, number_of_chunks: int, content_hash: str | None, embedding_model: str | None, chunker_version: str | None, chunker_config_hash: str | None, pipeline_version: str = INDEX_PIPELINE_VERSION) -> None:
        entry = ManifestEntry(
            attachment_key=attachment_key,
            parent_key=parent_key,
            citation_key=citation_key,
            zotero_item_version=None,
            pdf_path=pdf_path or "",
            pdf_mtime=pdf_mtime,
            pdf_size=pdf_size,
            pdf_hash=IndexManifest.compute_pdf_hash(pdf_path) if pdf_path and not pdf_path.startswith("abstract:") else None,
            content_hash=content_hash,
            embedding_model=embedding_model,
            embedding_config_hash=None,
            chunker_version=chunker_version,
            chunker_config_hash=chunker_config_hash,
            indexed_at=datetime.now(timezone.utc).isoformat(),
            number_of_pages=number_of_pages,
            number_of_chunks=number_of_chunks,
            index_status="indexed",
            error_message=None,
            pipeline_version=pipeline_version,
        )
        self.manifest.upsert_entry(entry)

    def mark_failed(self, attachment_key: str, error_message: str) -> None:
        entry = self.manifest.get_entry(attachment_key)
        if entry is None:
            entry = ManifestEntry(
                attachment_key=attachment_key,
                parent_key=None,
                citation_key=None,
                zotero_item_version=None,
                pdf_path="",
                pdf_mtime=None,
                pdf_size=None,
                pdf_hash=None,
                content_hash=None,
                embedding_model=None,
                embedding_config_hash=None,
                chunker_version=None,
                chunker_config_hash=None,
                indexed_at=datetime.now(timezone.utc).isoformat(),
                number_of_pages=None,
                number_of_chunks=None,
                index_status="failed",
                error_message=error_message,
            )
        else:
            entry.index_status = "failed"
            entry.error_message = error_message
            entry.indexed_at = datetime.now(timezone.utc).isoformat()
        self.manifest.upsert_entry(entry)
