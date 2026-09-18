from __future__ import annotations


def build_zotero_pdf_link(attachment_key: str, page: int | None = None) -> str:
    if not attachment_key:
        return ""
    page_str = f"?page={page}" if page and page > 0 else ""
    return f"zotero://open-pdf/library/items/{attachment_key}{page_str}"
