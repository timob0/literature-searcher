from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ZoteroCreator:
    creatorType: str | None = None
    firstName: str | None = None
    lastName: str | None = None

    @property
    def full_name(self) -> str:
        parts = [part for part in (self.firstName, self.lastName) if part]
        return " ".join(parts) if parts else "Unknown"


@dataclass(slots=True)
class ZoteroItem:
    item_key: str
    parent_key: str | None = None
    title: str | None = None
    publication_year: int | None = None
    publication_title: str | None = None
    doi: str | None = None
    url: str | None = None
    item_type: str | None = None
    creators: list[ZoteroCreator] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    collections: list[str] = field(default_factory=list)
    attachment_key: str | None = None
    pdf_path: str | None = None
    citation_key: str | None = None
    abstract: str | None = None
    zotero_version: int | None = None
    json: dict[str, Any] = field(default_factory=dict)

    @property
    def authors(self) -> list[str]:
        return [creator.full_name for creator in self.creators]

    @property
    def citation_label(self) -> str:
        return self.citation_key or self.item_key
