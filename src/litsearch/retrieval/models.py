from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ChunkHit:
    chunk_id: str
    attachment_key: str
    parent_key: str
    citation_key: str | None
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    page_start: int = 0
    page_end: int = 0
    section_type: str | None = None
    section_title: str | None = None
    text: str = ""
    score: float = 0.0
    zotero_link: str = ""


@dataclass(slots=True)
class SearchFilters:
    year: int | None = None
    item_type: str | None = None
    collection: str | None = None
    tag: str | None = None
    author: str | None = None
    citation_key: str | None = None
    parent_key: str | None = None
