from __future__ import annotations

import json
from typing import Any

import httpx

from litsearch.config import settings


class BetterBibTeXAdapter:
    def __init__(self, api_url: str | None = None) -> None:
        self.api_url = (api_url or settings.zotero_api_url).rstrip("/")

    def is_available(self) -> bool:
        try:
            resp = httpx.get(f"{self.api_url}/better-bibtex", timeout=5.0)
            return resp.status_code in {200, 204}
        except Exception:
            return False

    def get_citation_key(self, item_key: str) -> str | None:
        if not settings.better_bibtex_enabled:
            return None
        try:
            payload = {"method": "better-bibtex.citationKey", "params": [item_key]}
            resp = httpx.post(f"{self.api_url}/better-bibtex", json=payload, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                return data.get("result") or data.get("citationKey")
            if isinstance(data, list):
                if data and isinstance(data[0], dict):
                    return data[0].get("citationKey")
            if isinstance(data, str):
                return data
        except Exception:
            return None
        return None
