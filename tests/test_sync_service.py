from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from litsearch.indexing.incremental import IncrementalIndexer
from litsearch.indexing.manifest import IndexManifest, ManifestEntry
from litsearch.indexing.sync_service import SyncSummary, ZoteroSyncService


@dataclass
class FakeZoteroClient:
    items: list[dict]

    def get_library_items(self, limit: int = 100, collection_id: str | None = None):
        return [
            type("Item", (), {
                "item_key": item["item_key"],
                "title": item["title"],
                "publication_year": item.get("year"),
                "item_type": item.get("item_type"),
                "authors": item.get("authors", []),
                "attachment_key": item.get("attachment_key"),
                "pdf_path": item.get("pdf_path"),
                "citation_key": item.get("citation_key"),
                "abstract": item.get("abstract"),
                "parent_key": item.get("parent_key"),
            })() for item in self.items
        ]

    def get_pdf_attachments(self, item_key: str):
        return []

    def resolve_pdf_path(self, attachment_key: str):
        return None


class FakeIndexer:
    def index_pdf(self, pdf_path: str, *, attachment_key: str, parent_key: str, citation_key: str | None):
        return ["chunk-1"]
    def index_text(self, text: str, *, attachment_key: str, parent_key: str, citation_key: str):
        return ["abstract-chunk-1"]


class FakePipelinedIndexer:
    """Exposes the split extract/store API so ZoteroSyncService takes the threaded pipeline path."""

    pipeline_version = "2"

    def __init__(self) -> None:
        self.stored_attachment_keys: list[str] = []

    def extract_chunks(self, pdf_path: str, *, attachment_key: str, parent_key: str, citation_key: str | None):
        return [f"chunk-{attachment_key}"], 1

    def extract_text_chunks(self, text: str, *, attachment_key: str, parent_key: str, citation_key: str | None):
        return [f"abstract-chunk-{attachment_key}"]

    def store_chunks(self, chunks, *, attachment_key: str, title: str | None, authors: list[str] | None = None):
        self.stored_attachment_keys.append(attachment_key)
        return list(chunks)

    def chunk_ids_for_attachment(self, attachment_key: str) -> list[str]:
        return []


def test_sync_indexes_abstract_when_pdf_is_missing(tmp_path: Path) -> None:
    service = ZoteroSyncService(client=FakeZoteroClient([
        {"item_key": "item-1", "title": "Abstract only", "attachment_key": "item-1", "pdf_path": "abstract:item-1", "abstract": "Useful abstract text.", "citation_key": "authorAbstract2024", "parent_key": "item-1"}
    ]), indexer=FakeIndexer(), incremental=IncrementalIndexer(IndexManifest(str(tmp_path / "manifest.sqlite"))))

    summary = service.sync()

    assert summary.abstract_sources_indexed == 1
    assert summary.chunks_created == 1


def test_sync_refreshes_citation_key_without_reindexing(tmp_path: Path) -> None:
    manifest = IndexManifest(str(tmp_path / "manifest.sqlite"))
    pdf_path = tmp_path / "example.pdf"
    pdf_path.write_bytes(b"fake pdf content")
    pdf_stat = pdf_path.stat()
    manifest.upsert_entry(ManifestEntry(
        attachment_key="att-1",
        parent_key="parent-1",
        citation_key=None,
        zotero_item_version=None,
        pdf_path=str(pdf_path),
        pdf_mtime=pdf_stat.st_mtime,
        pdf_size=pdf_stat.st_size,
        pdf_hash=None,
        content_hash=None,
        embedding_model=None,
        embedding_config_hash=None,
        chunker_version=None,
        chunker_config_hash=None,
        indexed_at=None,
        number_of_pages=1,
        number_of_chunks=1,
        index_status="indexed",
        error_message=None,
        pipeline_version="2",
    ))
    service = ZoteroSyncService(
        client=FakeZoteroClient([{"item_key": "item-1", "title": "Example", "attachment_key": "att-1", "pdf_path": str(pdf_path), "citation_key": "authorExample2024", "parent_key": "parent-1"}]),
        indexer=FakeIndexer(),
        incremental=IncrementalIndexer(manifest),
    )
    summary = service.sync()

    assert summary.unchanged_pdfs_skipped == 1
    assert manifest.get_entry("att-1").citation_key == "authorExample2024"


def test_sync_summary_counts_new_files(tmp_path: Path) -> None:
    pdf_path = tmp_path / "example.pdf"
    pdf_path.write_bytes(b"fake pdf content")

    service = ZoteroSyncService(client=FakeZoteroClient([
        {"item_key": "item-1", "title": "Example", "year": 2024, "item_type": "journalArticle", "authors": ["Author"], "attachment_key": "att-1", "pdf_path": str(pdf_path), "citation_key": "authorExample2024", "parent_key": "parent-1"}
    ]), indexer=FakeIndexer(), incremental=IncrementalIndexer(IndexManifest(str(tmp_path / "manifest.sqlite"))))
    summary = service.sync(limit=10)
    assert summary.pdfs_discovered == 1
    assert summary.new_pdfs_indexed == 1
    assert summary.chunks_created == 1


def test_sync_uses_pipelined_path_when_indexer_supports_split_api(tmp_path: Path) -> None:
    pdf_paths = []
    items = []
    for index in range(6):
        pdf_path = tmp_path / f"example-{index}.pdf"
        pdf_path.write_bytes(b"fake pdf content")
        pdf_paths.append(pdf_path)
        items.append({
            "item_key": f"item-{index}", "title": f"Example {index}", "attachment_key": f"att-{index}",
            "pdf_path": str(pdf_path), "citation_key": f"authorExample{index}2024", "parent_key": f"parent-{index}",
        })

    indexer = FakePipelinedIndexer()
    service = ZoteroSyncService(
        client=FakeZoteroClient(items),
        indexer=indexer,
        incremental=IncrementalIndexer(IndexManifest(str(tmp_path / "manifest.sqlite"))),
    )
    summary = service.sync(limit=10)

    assert summary.pdfs_discovered == 6
    assert summary.new_pdfs_indexed == 6
    assert summary.failed_pdfs == 0
    assert sorted(indexer.stored_attachment_keys) == [f"att-{i}" for i in range(6)]
    assert sorted(summary.indexed_items) == [f"att-{i}" for i in range(6)]


def test_sync_pipelined_path_marks_extraction_failures(tmp_path: Path) -> None:
    class FailingPipelinedIndexer(FakePipelinedIndexer):
        def extract_chunks(self, pdf_path: str, *, attachment_key: str, parent_key: str, citation_key: str | None):
            if attachment_key == "att-bad":
                raise ValueError("broken pdf")
            return super().extract_chunks(pdf_path, attachment_key=attachment_key, parent_key=parent_key, citation_key=citation_key)

    good_path = tmp_path / "good.pdf"
    good_path.write_bytes(b"fake pdf content")
    bad_path = tmp_path / "bad.pdf"
    bad_path.write_bytes(b"fake pdf content")

    indexer = FailingPipelinedIndexer()
    service = ZoteroSyncService(
        client=FakeZoteroClient([
            {"item_key": "item-good", "title": "Good", "attachment_key": "att-good", "pdf_path": str(good_path), "citation_key": "goodAuthor2024", "parent_key": "parent-good"},
            {"item_key": "item-bad", "title": "Bad", "attachment_key": "att-bad", "pdf_path": str(bad_path), "citation_key": "badAuthor2024", "parent_key": "parent-bad"},
        ]),
        indexer=indexer,
        incremental=IncrementalIndexer(IndexManifest(str(tmp_path / "manifest.sqlite"))),
    )
    summary = service.sync(limit=10)

    assert summary.failed_pdfs == 1
    assert summary.new_pdfs_indexed == 1
    assert indexer.stored_attachment_keys == ["att-good"]

