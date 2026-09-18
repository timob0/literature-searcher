from __future__ import annotations

from pathlib import Path

from litsearch.indexing.incremental import IncrementalIndexer
from litsearch.indexing.manifest import IndexManifest, ManifestEntry


def test_manifest_round_trip(tmp_path: Path) -> None:
    db_path = tmp_path / "manifest.sqlite"
    manifest = IndexManifest(str(db_path))
    entry = ManifestEntry(
        attachment_key="att-1",
        parent_key="item-1",
        citation_key="smithTest2024",
        zotero_item_version=1,
        pdf_path="/tmp/test.pdf",
        pdf_mtime=123.0,
        pdf_size=999,
        pdf_hash="abc",
        content_hash="def",
        embedding_model="BAAI/bge-m3",
        embedding_config_hash="cfg1",
        chunker_version="1.0",
        chunker_config_hash="ck1",
        indexed_at="2026-01-01",
        number_of_pages=12,
        number_of_chunks=4,
        index_status="indexed",
        error_message=None,
        pipeline_version="1",
    )

    manifest.upsert_entry(entry)
    saved = manifest.get_entry("att-1")

    assert saved is not None
    assert saved.citation_key == "smithTest2024"
    assert saved.index_status == "indexed"


def test_pipeline_version_change_requires_reindex(tmp_path: Path) -> None:
    manifest = IndexManifest(str(tmp_path / "manifest.sqlite"))
    manifest.upsert_entry(ManifestEntry(
        attachment_key="att-1", parent_key="item-1", citation_key="cite", zotero_item_version=None,
        pdf_path=str(tmp_path / "paper.pdf"), pdf_mtime=1.0, pdf_size=10, pdf_hash="hash",
        content_hash=None, embedding_model=None, embedding_config_hash=None, chunker_version=None,
        chunker_config_hash=None, indexed_at=None, number_of_pages=2, number_of_chunks=1,
        index_status="indexed", error_message=None, pipeline_version="1",
    ))
    (tmp_path / "paper.pdf").write_bytes(b"0123456789")

    assert IncrementalIndexer(manifest).should_index("att-1", "item-1", "cite", str(tmp_path / "paper.pdf"), 10, 1.0, pipeline_version="2") == "changed"
