from __future__ import annotations

from litsearch.zotero.links import build_zotero_pdf_link


def test_build_zotero_pdf_link_includes_page_when_present() -> None:
    link = build_zotero_pdf_link("abc123", 42)
    assert link == "zotero://open-pdf/library/items/abc123?page=42"


def test_build_zotero_pdf_link_without_page() -> None:
    link = build_zotero_pdf_link("abc123")
    assert link == "zotero://open-pdf/library/items/abc123"
