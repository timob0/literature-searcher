from __future__ import annotations

from pathlib import Path

import fitz
from typer.testing import CliRunner

from litsearch.cli import app
from litsearch.documents.pdf import extract_pdf_text
from litsearch.documents.models import DocumentPage, ExtractionResult
from litsearch.indexing.chunker import SimpleChunker
from litsearch.indexing.indexer import PDFIndexer
from litsearch.zotero.better_bibtex import BetterBibTeXAdapter


def _create_pdf(path: Path, contents: list[str]) -> None:
    doc = fitz.open()
    for text in contents:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_extract_pdf_text_preserves_pages_and_sentences(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _create_pdf(pdf_path, ["This is the first page. It has a clear sentence.", "This is the second page. Another sentence follows."])

    result = extract_pdf_text(str(pdf_path))

    assert result.error is None
    assert result.requires_ocr is False
    assert len(result.pages) == 2
    assert sum(len(page.paragraphs) for page in result.pages) >= 2
    assert sum(len(sentence.text) for page in result.pages for paragraph in page.paragraphs for sentence in paragraph.sentences) >= 2


def test_chunker_creates_reproducible_chunks() -> None:
    text = "This is a paragraph about supportive leadership in organizations. It has a few concrete examples and useful context. " * 3
    chunks = SimpleChunker().chunk_document(text, attachment_key="att-1", parent_key="item-1", citation_key="smithSupportiveLeadership2024")

    assert len(chunks) >= 1
    assert chunks[0].attachment_key == "att-1"
    assert chunks[0].citation_key == "smithSupportiveLeadership2024"
    assert chunks[0].content_hash


def test_indexer_preserves_extracted_page_ranges(monkeypatch) -> None:
    class FakeEmbeddingProvider:
        def embed_documents(self, texts):
            return [[float(len(text))] for text in texts]

    class FakeVectorStore:
        def __init__(self):
            self.metadata = []

        def add(self, *, ids, texts, metadatas, embeddings):
            self.metadata.extend(metadatas)

        def delete_by_ids(self, ids):
            return None

    extraction = ExtractionResult(
        attachment_key="att-1", parent_key="parent-1", citation_key="cite2026", pdf_path="paper.pdf",
        pages=[DocumentPage(1, "Alpha " * 10), DocumentPage(2, "Beta " * 10), DocumentPage(3, "Gamma " * 10)],
    )
    monkeypatch.setattr("litsearch.indexing.indexer.extract_pdf_text", lambda path: extraction)
    store = FakeVectorStore()
    indexer = PDFIndexer(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
        chunker=SimpleChunker(target_tokens=12, min_tokens=2, max_tokens=12, overlap_tokens=2),
    )

    indexer.index_pdf("paper.pdf", attachment_key="att-1", parent_key="parent-1", citation_key="cite2026")

    assert indexer.last_number_of_pages == 3
    assert [(metadata["page_start"], metadata["page_end"]) for metadata in store.metadata] == [(1, 2), (2, 3), (3, 3)]


def test_better_bibtex_adapter_handles_unavailable_service() -> None:
    adapter = BetterBibTeXAdapter(api_url="http://localhost:1")
    assert adapter.get_citation_key("not-real") is None


def test_cli_status_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Status" in result.output
    assert "Indexed documents" in result.output
    assert "Zotero collections" in result.output


def test_cli_documents_command_lists_manifest_entries(tmp_path, monkeypatch) -> None:
    from litsearch.indexing.manifest import IndexManifest, ManifestEntry

    manifest = IndexManifest(str(tmp_path / "manifest.sqlite"))
    manifest.upsert_entry(ManifestEntry(
        attachment_key="att-1",
        parent_key="parent-1",
        citation_key="smith2024",
        zotero_item_version=None,
        pdf_path="/library/paper.pdf",
        pdf_mtime=None,
        pdf_size=None,
        pdf_hash=None,
        content_hash=None,
        embedding_model=None,
        embedding_config_hash=None,
        chunker_version=None,
        chunker_config_hash=None,
        indexed_at=None,
        number_of_pages=None,
        number_of_chunks=4,
        index_status="indexed",
        error_message=None,
    ))
    monkeypatch.setattr("litsearch.cli.IndexManifest", lambda: manifest)

    result = CliRunner().invoke(app, ["documents", "--status", "indexed", "--path"])
    assert result.exit_code == 0
    assert "/library/paper.pdf" in result.output
    assert "4 chunks" in result.output
