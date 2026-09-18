from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
import inspect

from litsearch.config import settings
from litsearch.indexing.incremental import IncrementalIndexer


@dataclass(slots=True)
class SyncSummary:
    pdfs_discovered: int = 0
    new_pdfs_indexed: int = 0
    changed_pdfs_reindexed: int = 0
    unchanged_pdfs_skipped: int = 0
    deleted_pdfs_removed: int = 0
    pdfs_requiring_ocr: int = 0
    failed_pdfs: int = 0
    skipped_without_citation: int = 0
    abstract_sources_indexed: int = 0
    pages_processed: int = 0
    chunks_created: int = 0
    embedding_time: float = 0.0
    total_indexing_time: float = 0.0
    indexed_items: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _PreparedItem:
    """Everything a worker thread needs to extract+chunk one item, independent of the manifest/vector store."""

    attachment_key: str
    citation_key: str
    chunk_parent_key: str
    raw_parent_key: str | None
    pdf_path: str | None
    is_abstract: bool
    abstract_text: str | None
    source_size: int | None
    source_mtime: float | None
    decision: str
    title: str | None
    authors: list[str] | None


class ZoteroSyncService:
    def __init__(self, client: object | None = None, indexer: object | None = None, incremental: IncrementalIndexer | None = None) -> None:
        self.client = client
        self.indexer = indexer
        self.incremental = incremental or IncrementalIndexer()

    def sync(self, limit: int = 100, collection_id: str | None = None, collection_ids: list[str] | None = None, progress=None, items: list[object] | None = None) -> SyncSummary:
        summary = SyncSummary()
        if self.client is None or self.indexer is None:
            return summary

        if items is None:
            if collection_ids is not None:
                items = self.client.get_library_items_for_collections(collection_ids, limit=limit)
            else:
                items = self.client.get_library_items(limit=limit, collection_id=collection_id)
        summary.pdfs_discovered = len(items)
        discovered_keys: set[str] = set()
        # A collection filter or item limit only ever yields a subset of the library, so "not
        # discovered" would not mean "deleted from Zotero" -- only prune on a full, unrestricted sync.
        can_prune = collection_id is None and collection_ids is None and limit is None

        total = len(items)
        supports_pipeline = all(hasattr(self.indexer, name) for name in ("extract_chunks", "extract_text_chunks", "store_chunks"))
        if supports_pipeline:
            self._run_pipelined(items, total, progress, summary, discovered_keys)
        else:
            self._run_sequential(items, total, progress, summary, discovered_keys)

        if not can_prune:
            return summary

        manifest = self.incremental.manifest
        for entry in manifest.all_entries():
            if entry.attachment_key in discovered_keys:
                continue
            if entry.index_status == "indexed":
                chunk_ids = getattr(self.indexer, "chunk_ids_for_attachment", lambda _key: [])(entry.attachment_key)
                if chunk_ids:
                    self.indexer.vector_store.delete_by_ids(chunk_ids)
                manifest.delete_entry(entry.attachment_key)
                summary.deleted_pdfs_removed += 1

        return summary

    def _run_sequential(self, items: list[object], total: int, progress, summary: SyncSummary, discovered_keys: set[str]) -> None:
        """Original one-item-at-a-time path, kept for indexers that don't expose the split extract/store API."""
        for position, item in enumerate(items, start=1):
            if progress:
                progress(position, total, item)
            attachment_key = getattr(item, "attachment_key", None)
            citation_key = getattr(item, "citation_key", None)
            if not citation_key:
                summary.skipped_without_citation += 1
                continue
            if not attachment_key:
                continue
            discovered_keys.add(attachment_key)

            pdf_path = item.pdf_path or self.client.resolve_pdf_path(attachment_key)
            is_abstract = bool(pdf_path and pdf_path.startswith("abstract:"))
            if not is_abstract and (not pdf_path or not Path(pdf_path).exists()):
                continue

            abstract = getattr(item, "abstract", None)
            pdf_file = Path(pdf_path) if not is_abstract else None
            source_size = len(abstract.encode("utf-8")) if is_abstract and abstract else pdf_file.stat().st_size
            source_mtime = None if is_abstract else pdf_file.stat().st_mtime

            decision = self.incremental.should_index(
                attachment_key,
                getattr(item, "parent_key", None),
                citation_key,
                pdf_path,
                source_size,
                source_mtime,
                pipeline_version=getattr(self.indexer, "pipeline_version", "2"),
            )

            if decision == "unchanged":
                entry = self.incremental.manifest.get_entry(attachment_key)
                citation_key = getattr(item, "citation_key", None)
                if entry is not None and citation_key and entry.citation_key != citation_key:
                    entry.citation_key = citation_key
                    self.incremental.manifest.upsert_entry(entry)
                summary.unchanged_pdfs_skipped += 1
                continue

            try:
                title = getattr(item, "title", None)
                if is_abstract:
                    indexed_ids = self._index_with_optional_title(
                        self.indexer.index_text,
                        abstract or "",
                        attachment_key=attachment_key,
                        parent_key=getattr(item, "parent_key", None) or getattr(item, "item_key", ""),
                        citation_key=citation_key,
                        title=title,
                        authors=getattr(item, "authors", None),
                    )
                else:
                    indexed_ids = self._index_with_optional_title(
                        self.indexer.index_pdf,
                        pdf_path,
                        attachment_key=attachment_key,
                        parent_key=getattr(item, "parent_key", "") or "",
                        citation_key=citation_key,
                        title=title,
                        authors=getattr(item, "authors", None),
                    )
            except Exception as exc:
                self.incremental.mark_failed(attachment_key, str(exc))
                summary.failed_pdfs += 1
                continue

            self.incremental.mark_indexed(
                attachment_key,
                getattr(item, "parent_key", None),
                citation_key,
                pdf_path,
                source_size,
                source_mtime,
                number_of_pages=getattr(self.indexer, "last_number_of_pages", 0),
                number_of_chunks=len(indexed_ids),
                content_hash=None,
                embedding_model=getattr(getattr(self.indexer, "embedding_provider", None), "model_name", None),
                chunker_version=None,
                chunker_config_hash=None,
                pipeline_version=getattr(self.indexer, "pipeline_version", "2"),
            )
            summary.indexed_items.append(attachment_key)
            summary.chunks_created += len(indexed_ids)
            summary.abstract_sources_indexed += int(is_abstract)

            if decision == "new":
                summary.new_pdfs_indexed += 1
            elif decision == "changed":
                summary.changed_pdfs_reindexed += 1

    def _prepare_item(self, item: object, summary: SyncSummary) -> _PreparedItem | None:
        """Cheap, sequential pre-checks (file existence, incremental decision). Returns None if the item
        should be skipped -- the appropriate summary counter is already updated in that case."""
        attachment_key = getattr(item, "attachment_key", None)
        citation_key = getattr(item, "citation_key", None)
        pdf_path = item.pdf_path or self.client.resolve_pdf_path(attachment_key)
        is_abstract = bool(pdf_path and pdf_path.startswith("abstract:"))
        if not is_abstract and (not pdf_path or not Path(pdf_path).exists()):
            return None

        abstract = getattr(item, "abstract", None)
        pdf_file = Path(pdf_path) if not is_abstract else None
        source_size = len(abstract.encode("utf-8")) if is_abstract and abstract else pdf_file.stat().st_size
        source_mtime = None if is_abstract else pdf_file.stat().st_mtime

        decision = self.incremental.should_index(
            attachment_key,
            getattr(item, "parent_key", None),
            citation_key,
            pdf_path,
            source_size,
            source_mtime,
            pipeline_version=getattr(self.indexer, "pipeline_version", "2"),
        )
        if decision == "unchanged":
            entry = self.incremental.manifest.get_entry(attachment_key)
            if entry is not None and citation_key and entry.citation_key != citation_key:
                entry.citation_key = citation_key
                self.incremental.manifest.upsert_entry(entry)
            summary.unchanged_pdfs_skipped += 1
            return None

        chunk_parent_key = (getattr(item, "parent_key", None) or getattr(item, "item_key", "")) if is_abstract else (getattr(item, "parent_key", "") or "")
        return _PreparedItem(
            attachment_key=attachment_key,
            citation_key=citation_key,
            chunk_parent_key=chunk_parent_key,
            raw_parent_key=getattr(item, "parent_key", None),
            pdf_path=pdf_path,
            is_abstract=is_abstract,
            abstract_text=abstract,
            source_size=source_size,
            source_mtime=source_mtime,
            decision=decision,
            title=getattr(item, "title", None),
            authors=getattr(item, "authors", None),
        )

    def _extract_job(self, prepared: _PreparedItem) -> tuple[list, int]:
        """Runs in a worker thread: PDF parsing + chunking only, no GPU/manifest/vector-store access."""
        if prepared.is_abstract:
            chunks = self.indexer.extract_text_chunks(
                prepared.abstract_text or "",
                attachment_key=prepared.attachment_key,
                parent_key=prepared.chunk_parent_key,
                citation_key=prepared.citation_key,
            )
            return chunks, 0
        return self.indexer.extract_chunks(
            prepared.pdf_path,
            attachment_key=prepared.attachment_key,
            parent_key=prepared.chunk_parent_key,
            citation_key=prepared.citation_key,
        )

    def _consume_job(self, future: Future, prepared: _PreparedItem, summary: SyncSummary) -> None:
        """Runs on the main thread: embedding + Chroma upsert + manifest bookkeeping, kept single-threaded."""
        try:
            chunks, number_of_pages = future.result()
        except Exception as exc:
            self.incremental.mark_failed(prepared.attachment_key, str(exc))
            summary.failed_pdfs += 1
            return

        indexed_ids = self.indexer.store_chunks(chunks, attachment_key=prepared.attachment_key, title=prepared.title, authors=prepared.authors)
        self.incremental.mark_indexed(
            prepared.attachment_key,
            prepared.raw_parent_key,
            prepared.citation_key,
            prepared.pdf_path,
            prepared.source_size,
            prepared.source_mtime,
            number_of_pages=number_of_pages,
            number_of_chunks=len(indexed_ids),
            content_hash=None,
            embedding_model=getattr(getattr(self.indexer, "embedding_provider", None), "model_name", None),
            chunker_version=None,
            chunker_config_hash=None,
            pipeline_version=getattr(self.indexer, "pipeline_version", "2"),
        )
        summary.indexed_items.append(prepared.attachment_key)
        summary.chunks_created += len(indexed_ids)
        summary.abstract_sources_indexed += int(prepared.is_abstract)
        if prepared.decision == "new":
            summary.new_pdfs_indexed += 1
        elif prepared.decision == "changed":
            summary.changed_pdfs_reindexed += 1

    def _run_pipelined(self, items: list[object], total: int, progress, summary: SyncSummary, discovered_keys: set[str]) -> None:
        """Overlaps PDF extraction/chunking (worker threads) with embedding+storage (main thread, single GPU/HTTP client)."""
        max_workers = max(1, settings.extraction_workers)
        lookahead = max_workers + 2
        position_iter = enumerate(items, start=1)
        pending: deque[tuple[Future, _PreparedItem]] = deque()

        def submit_next(executor: ThreadPoolExecutor) -> bool:
            for position, item in position_iter:
                if progress:
                    progress(position, total, item)
                attachment_key = getattr(item, "attachment_key", None)
                citation_key = getattr(item, "citation_key", None)
                if not citation_key:
                    summary.skipped_without_citation += 1
                    continue
                if not attachment_key:
                    continue
                discovered_keys.add(attachment_key)

                prepared = self._prepare_item(item, summary)
                if prepared is None:
                    continue

                future = executor.submit(self._extract_job, prepared)
                pending.append((future, prepared))
                return True
            return False

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _ in range(lookahead):
                if not submit_next(executor):
                    break
            while pending:
                future, prepared = pending.popleft()
                self._consume_job(future, prepared, summary)
                submit_next(executor)

    @staticmethod
    def _index_with_optional_title(index_method, source: str, *, attachment_key: str, parent_key: str, citation_key: str, title: str | None, authors: list[str] | None = None):
        kwargs = {
            "attachment_key": attachment_key,
            "parent_key": parent_key,
            "citation_key": citation_key,
        }
        if "title" in inspect.signature(index_method).parameters:
            kwargs["title"] = title
        if "authors" in inspect.signature(index_method).parameters:
            kwargs["authors"] = authors
        return index_method(source, **kwargs)
