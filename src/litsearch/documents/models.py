from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Sentence:
    sentence_id: str
    paragraph_id: str
    page_number: int
    section_type: str
    section_title: str | None
    text: str
    normalized_text: str | None = None
    ordering: int = 0


@dataclass(slots=True)
class Paragraph:
    paragraph_id: str
    page_number: int
    section_type: str
    section_title: str | None
    text: str
    sentences: list[Sentence] = field(default_factory=list)
    ordering: int = 0


@dataclass(slots=True)
class DocumentPage:
    page_number: int
    text: str
    paragraphs: list[Paragraph] = field(default_factory=list)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    attachment_key: str
    parent_key: str
    citation_key: str | None
    text: str
    page_start: int
    page_end: int
    section_type: str
    section_title: str | None
    sentence_ids: list[str] = field(default_factory=list)
    paragraph_ids: list[str] = field(default_factory=list)
    chunk_index: int = 0
    content_hash: str = ""


@dataclass(slots=True)
class ExtractionResult:
    attachment_key: str
    parent_key: str
    citation_key: str | None
    pdf_path: str
    pages: list[DocumentPage] = field(default_factory=list)
    requires_ocr: bool = False
    error: str | None = None
    raw_stats: dict[str, Any] = field(default_factory=dict)
