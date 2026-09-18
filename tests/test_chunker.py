from __future__ import annotations

from litsearch.indexing.chunker import SimpleChunker
from litsearch.documents.models import DocumentPage


def test_chunker_splits_long_documents_with_configured_limits() -> None:
    chunker = SimpleChunker(target_tokens=20, min_tokens=5, max_tokens=25, overlap_tokens=4)
    text = " ".join(f"word{index}" for index in range(100))

    chunks = chunker.chunk_document(text, attachment_key="att-1", parent_key="paper-1", citation_key="paper2025")

    assert len(chunks) > 1
    assert all(len(chunk.text.split()) <= 25 for chunk in chunks)
    assert chunks[0].text.split()[-4:] == chunks[1].text.split()[:4]


def test_chunker_keeps_short_documents_together() -> None:
    chunker = SimpleChunker(target_tokens=20, min_tokens=5, max_tokens=25, overlap_tokens=4)

    chunks = chunker.chunk_document("one two three", attachment_key="att-1", parent_key="paper-1", citation_key=None)

    assert len(chunks) == 1
    assert chunks[0].text == "one two three"


def test_page_aware_chunker_preserves_single_and_cross_page_ranges() -> None:
    chunker = SimpleChunker(target_tokens=20, min_tokens=2, max_tokens=20, overlap_tokens=2)
    chunks = chunker.chunk_pages(
        [DocumentPage(7, "alpha " * 8), DocumentPage(8, "beta " * 8)],
        attachment_key="att-1", parent_key="paper-1", citation_key="paper2025",
    )

    assert chunks[0].page_start == 7
    assert chunks[0].page_end == 8


def test_page_aware_chunker_keeps_physical_numbers_across_blank_pages() -> None:
    chunks = SimpleChunker().chunk_pages(
        [DocumentPage(7, "alpha"), DocumentPage(8, ""), DocumentPage(9, "omega")],
        attachment_key="att-1", parent_key="paper-1", citation_key=None,
    )

    assert chunks[0].page_start == 7
    assert chunks[0].page_end == 9