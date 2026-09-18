from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from litsearch.config import settings
from litsearch.zotero.models import ZoteroCreator, ZoteroItem


class ZoteroClient:
    def __init__(self, api_url: str | None = None) -> None:
        self.api_url = (api_url or settings.zotero_api_url).rstrip("/")

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = httpx.get(f"{self.api_url}{path}", params=params, timeout=20.0)
        resp.raise_for_status()
        return resp.json()

    def is_available(self) -> bool:
        try:
            response = httpx.get(f"{self.api_url}/", timeout=10.0)
            return response.is_success
        except Exception:
            return False

    @staticmethod
    def _library_path(path: str = "") -> str:
        return f"/users/0{path}"

    def get_library_items(self, limit: int | None = 100, collection_id: str | None = None, start: int = 0) -> list[ZoteroItem]:
        params: dict[str, Any] = {
            "limit": limit,
            "format": "json",
            "include": "data,bib,coins",
            "itemType": "-attachment",
            "start": start,
        }
        if collection_id:
            params["collectionKey"] = collection_id
        data = self._get(self._library_path("/items"), params=params)
        items = data.get("results", []) if isinstance(data, dict) else data
        return self._parse_library_items(items if isinstance(items, list) else [])

    def _parse_library_items(self, items: list[dict[str, Any]]) -> list[ZoteroItem]:
        with ThreadPoolExecutor(max_workers=settings.zotero_workers) as executor:
            parsed_items = list(executor.map(self._parse_library_item, items))
        return [item for item in parsed_items if item is not None]

    def _parse_library_item(self, raw_item: dict[str, Any]) -> ZoteroItem | None:
        item = self._parse_item(raw_item)
        if item.item_type == "attachment" or not item.citation_key:
            return None
        attachments = self.get_pdf_attachments(item.item_key)
        pdf_attachments = [
            attachment for attachment in attachments
            if attachment.get("data", {}).get("contentType") == "application/pdf"
            or str(attachment.get("data", {}).get("filename", "")).lower().endswith(".pdf")
        ]
        if pdf_attachments:
            attachment = pdf_attachments[0]
            attachment_data = attachment.get("data", {})
            item.attachment_key = attachment.get("key") or attachment_data.get("key")
            item.pdf_path = self._resolve_attachment_path(
                item.attachment_key,
                attachment_data.get("path"),
                attachment_data.get("filename"),
            )
        if item.pdf_path is None and item.abstract:
            item.attachment_key = item.item_key
            item.pdf_path = f"abstract:{item.item_key}"
        elif item.pdf_path is None:
            return None
        return item

    def get_collections(self) -> list[tuple[str, str]]:
        data = self._get(self._library_path("/collections"), {"format": "json"})
        records = data.get("results", []) if isinstance(data, dict) else data
        collections: list[tuple[str, str]] = []
        for record in records if isinstance(records, list) else []:
            if not isinstance(record, dict):
                continue
            record_data = record.get("data", {})
            key = record_data.get("key") or record.get("key")
            name = record_data.get("name")
            if key and name:
                collections.append((str(key), str(name)))
        return collections

    def get_collection_ids_for_names(self, names: list[str]) -> list[str]:
        data = self._get(self._library_path("/collections"), {"format": "json"})
        records = data.get("results", []) if isinstance(data, dict) else data
        collection_rows = [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []
        by_key: dict[str, dict[str, Any]] = {}
        children: dict[str, list[str]] = {}
        for record in collection_rows:
            record_data = record.get("data", {})
            key = record_data.get("key") or record.get("key")
            if not key:
                continue
            by_key[str(key)] = record_data
            parent = record_data.get("parentCollection")
            if parent:
                children.setdefault(str(parent), []).append(str(key))

        wanted = {name.strip().casefold() for name in names if name.strip()}
        selected = {
            key for key, record_data in by_key.items()
            if str(record_data.get("name", "")).casefold() in wanted
        }
        queue = list(selected)
        while queue:
            key = queue.pop()
            for child in children.get(key, []):
                if child not in selected:
                    selected.add(child)
                    queue.append(child)
        return sorted(selected)

    def get_item_collections(self, item_key: str) -> list[str]:
        item = self.get_item_by_key(item_key)
        if item is None:
            return []
        collection_names = dict(self.get_collections())
        return [collection_names[key] for key in item.json.get("collections", []) if key in collection_names]

    def get_library_items_for_collections(self, collection_ids: list[str], limit: int | None = 100, progress=None) -> list[ZoteroItem]:
        items: dict[str, ZoteroItem] = {}
        selected = set(collection_ids)
        page_size = 100
        start = 0
        page = 0
        while limit is None or len(items) < limit:
            page += 1
            if progress:
                progress(page, None, f"library page starting at {start}")
            params = {
                "limit": page_size,
                "start": start,
                "format": "json",
                "include": "data,bib,coins",
                "itemType": "-attachment",
            }
            raw_page = self._get(self._library_path("/items"), params=params)
            raw_items = raw_page.get("results", []) if isinstance(raw_page, dict) else raw_page
            page_items = self._parse_library_items(raw_items if isinstance(raw_items, list) else [])
            for item in page_items:
                if item.attachment_key and selected.intersection(item.json.get("collections", [])):
                    items[item.attachment_key] = item
                    if limit is not None and len(items) >= limit:
                        break
            if len(raw_items) < page_size:
                break
            start += page_size
        return list(items.values())

    def get_item_by_key(self, item_key: str) -> ZoteroItem | None:
        try:
            data = self._get(self._library_path(f"/items/{item_key}"), {"format": "json"})
            if not isinstance(data, dict):
                return None
            return self._parse_item(data)
        except Exception:
            return None

    def get_pdf_attachments(self, item_key: str) -> list[dict[str, Any]]:
        try:
            data = self._get(self._library_path(f"/items/{item_key}/children"), {"format": "json"})
        except Exception:
            return []
        results = data.get("results", []) if isinstance(data, dict) else data
        return [
            item for item in results
            if isinstance(item, dict)
            and item.get("data", {}).get("itemType") == "attachment"
        ] if isinstance(results, list) else []

    def resolve_pdf_path(self, attachment_key: str) -> str | None:
        try:
            data = self._get(self._library_path(f"/items/{attachment_key}/file"), {})
            if isinstance(data, dict):
                return self._resolve_attachment_path(attachment_key, data.get("path") or data.get("file"), data.get("filename"))
        except Exception:
            return None
        return None

    @staticmethod
    def _resolve_attachment_path(attachment_key: str | None, path: str | None, filename: str | None = None) -> str | None:
        if not path and attachment_key and filename:
            return str(Path(settings.zotero_storage_dir).expanduser() / attachment_key / filename)
        if not path:
            return None
        if path.startswith("storage:") and attachment_key:
            filename = path.removeprefix("storage:")
            return str(Path(settings.zotero_storage_dir).expanduser() / attachment_key / filename)
        return path

    def _parse_item(self, item: dict[str, Any]) -> ZoteroItem:
        data = item.get("data", {}) if isinstance(item, dict) else {}
        creators_raw = data.get("creators", [])
        creators = [
            ZoteroCreator(
                creatorType=creator.get("creatorType"),
                firstName=creator.get("firstName"),
                lastName=creator.get("lastName"),
            )
            for creator in creators_raw
            if isinstance(creator, dict)
        ]
        item_key = data.get("key") or item.get("key") or ""
        raw_date = data.get("date")
        year = int(raw_date[:4]) if isinstance(raw_date, str) and len(raw_date) >= 4 and raw_date[:4].isdigit() else None
        return ZoteroItem(
            item_key=item_key,
            parent_key=data.get("parentItem"),
            title=data.get("title"),
            publication_year=year,
            publication_title=data.get("publicationTitle"),
            doi=data.get("DOI"),
            url=data.get("url"),
            item_type=data.get("itemType"),
            citation_key=data.get("citationKey"),
            abstract=data.get("abstractNote"),
            creators=creators,
            tags=[tag.get("tag") for tag in data.get("tags", []) if isinstance(tag, dict) and tag.get("tag")],
            collections=[],
            attachment_key=data.get("attachmentKey"),
            pdf_path=data.get("path"),
            zotero_version=data.get("version"),
            json=data,
        )
