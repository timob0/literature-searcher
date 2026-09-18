from __future__ import annotations

from pathlib import Path

from litsearch.zotero.client import ZoteroClient


class FakeZoteroClient(ZoteroClient):
    def __init__(self, storage_dir: Path) -> None:
        super().__init__(api_url="http://zotero.test/api")
        self.storage_dir = storage_dir

    def _get(self, path: str, params=None):
        if path == "/users/0/collections":
            return [
                {"key": "collection-1", "data": {"key": "collection-1", "name": "Reading List", "parentCollection": False}},
                {"key": "collection-2", "data": {"key": "collection-2", "name": "Methods", "parentCollection": "collection-1"}},
            ]
        if path == "/users/0/items":
            return [{"key": "parent-1", "data": {"key": "parent-1", "itemType": "journalArticle", "title": "Example", "date": "2024", "citationKey": "authorExample2024"}}]
        if path == "/users/0/items/parent-1/children":
            return [{"key": "attachment-1", "data": {"itemType": "attachment", "contentType": "application/pdf", "path": "storage:paper.pdf"}}]
        raise AssertionError(path)


def test_library_items_resolve_child_pdf_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("litsearch.zotero.client.settings.zotero_storage_dir", str(tmp_path))
    client = FakeZoteroClient(tmp_path)
    items = client.get_library_items()

    assert len(items) == 1
    assert items[0].attachment_key == "attachment-1"
    assert items[0].pdf_path == str(tmp_path / "attachment-1" / "paper.pdf")
    assert items[0].publication_year == 2024
    assert items[0].citation_key == "authorExample2024"


def test_get_collections_returns_keys_and_names(tmp_path: Path) -> None:
    client = FakeZoteroClient(tmp_path)

    assert client.get_collections() == [("collection-1", "Reading List"), ("collection-2", "Methods")]
    assert client.get_collection_ids_for_names(["Reading List"]) == ["collection-1", "collection-2"]


def test_parse_item_keeps_abstract_and_requires_citation_key(tmp_path: Path) -> None:
    client = FakeZoteroClient(tmp_path)
    item = client._parse_item({"key": "item-1", "data": {
        "key": "item-1",
        "itemType": "journalArticle",
        "title": "Abstract only",
        "abstractNote": "An abstract for indexing.",
        "citationKey": "authorAbstract2024",
    }})

    assert item.abstract == "An abstract for indexing."
    assert item.citation_key == "authorAbstract2024"


def test_library_items_fall_back_to_abstract_without_pdf(tmp_path: Path, monkeypatch) -> None:
    client = FakeZoteroClient(tmp_path)
    monkeypatch.setattr(client, "get_pdf_attachments", lambda item_key: [])
    monkeypatch.setattr(client, "_get", lambda path, params=None: [{
        "key": "item-1",
        "data": {
            "key": "item-1",
            "itemType": "journalArticle",
            "title": "Abstract only",
            "abstractNote": "An abstract for indexing.",
            "citationKey": "authorAbstract2024",
        },
    }] if path == "/users/0/items" else [])

    items = client.get_library_items()

    assert items[0].attachment_key == "item-1"
    assert items[0].pdf_path == "abstract:item-1"


def test_collection_item_lookup_reports_progress(tmp_path: Path) -> None:
    client = FakeZoteroClient(tmp_path)
    seen: list[tuple[int, int | None, str]] = []

    client.get_library_items = lambda limit=100, collection_id=None, start=0: []
    client.get_library_items_for_collections(["collection-1", "collection-2"], progress=lambda position, total, key: seen.append((position, total, key)))

    assert seen == [(1, None, "library page starting at 0")]