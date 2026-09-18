from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from litsearch.config import settings


@dataclass(slots=True)
class ManifestEntry:
    attachment_key: str
    parent_key: str
    citation_key: str | None
    zotero_item_version: int | None
    pdf_path: str
    pdf_mtime: float | None
    pdf_size: int | None
    pdf_hash: str | None
    content_hash: str | None
    embedding_model: str | None
    embedding_config_hash: str | None
    chunker_version: str | None
    chunker_config_hash: str | None
    indexed_at: str | None
    number_of_pages: int | None
    number_of_chunks: int | None
    index_status: str = "pending"
    error_message: str | None = None
    pipeline_version: str | None = None


class IndexManifest:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or settings.manifest_path
        self.path = Path(self.db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS manifest (
                    attachment_key TEXT PRIMARY KEY,
                    parent_key TEXT,
                    citation_key TEXT,
                    zotero_item_version INTEGER,
                    pdf_path TEXT,
                    pdf_mtime REAL,
                    pdf_size INTEGER,
                    pdf_hash TEXT,
                    content_hash TEXT,
                    embedding_model TEXT,
                    embedding_config_hash TEXT,
                    chunker_version TEXT,
                    chunker_config_hash TEXT,
                    indexed_at TEXT,
                    number_of_pages INTEGER,
                    number_of_chunks INTEGER,
                    index_status TEXT,
                    error_message TEXT
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(manifest)")}
            if "pipeline_version" not in columns:
                conn.execute("ALTER TABLE manifest ADD COLUMN pipeline_version TEXT")

    def get_entry(self, attachment_key: str) -> ManifestEntry | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM manifest WHERE attachment_key = ?",
                (attachment_key,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def upsert_entry(self, entry: ManifestEntry) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO manifest (
                    attachment_key, parent_key, citation_key, zotero_item_version, pdf_path, pdf_mtime,
                    pdf_size, pdf_hash, content_hash, embedding_model, embedding_config_hash,
                    chunker_version, chunker_config_hash, indexed_at, number_of_pages, number_of_chunks,
                    index_status, error_message
                    , pipeline_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(attachment_key) DO UPDATE SET
                    parent_key=excluded.parent_key,
                    citation_key=excluded.citation_key,
                    zotero_item_version=excluded.zotero_item_version,
                    pdf_path=excluded.pdf_path,
                    pdf_mtime=excluded.pdf_mtime,
                    pdf_size=excluded.pdf_size,
                    pdf_hash=excluded.pdf_hash,
                    content_hash=excluded.content_hash,
                    embedding_model=excluded.embedding_model,
                    embedding_config_hash=excluded.embedding_config_hash,
                    chunker_version=excluded.chunker_version,
                    chunker_config_hash=excluded.chunker_config_hash,
                    indexed_at=excluded.indexed_at,
                    number_of_pages=excluded.number_of_pages,
                    number_of_chunks=excluded.number_of_chunks,
                    index_status=excluded.index_status,
                    error_message=excluded.error_message
                    , pipeline_version=excluded.pipeline_version
                """,
                (
                    entry.attachment_key,
                    entry.parent_key,
                    entry.citation_key,
                    entry.zotero_item_version,
                    entry.pdf_path,
                    entry.pdf_mtime,
                    entry.pdf_size,
                    entry.pdf_hash,
                    entry.content_hash,
                    entry.embedding_model,
                    entry.embedding_config_hash,
                    entry.chunker_version,
                    entry.chunker_config_hash,
                    entry.indexed_at,
                    entry.number_of_pages,
                    entry.number_of_chunks,
                    entry.index_status,
                    entry.error_message,
                    entry.pipeline_version,
                ),
            )

    def delete_entry(self, attachment_key: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM manifest WHERE attachment_key = ?", (attachment_key,))

    def all_entries(self) -> list[ManifestEntry]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM manifest ORDER BY attachment_key").fetchall()
        return [self._row_to_entry(row) for row in rows]

    def _row_to_entry(self, row: tuple) -> ManifestEntry:
        return ManifestEntry(
            attachment_key=row[0],
            parent_key=row[1],
            citation_key=row[2],
            zotero_item_version=row[3],
            pdf_path=row[4],
            pdf_mtime=row[5],
            pdf_size=row[6],
            pdf_hash=row[7],
            content_hash=row[8],
            embedding_model=row[9],
            embedding_config_hash=row[10],
            chunker_version=row[11],
            chunker_config_hash=row[12],
            indexed_at=row[13],
            number_of_pages=row[14],
            number_of_chunks=row[15],
            index_status=row[16],
            error_message=row[17],
            pipeline_version=row[18] if len(row) > 18 else None,
        )

    @staticmethod
    def compute_pdf_hash(pdf_path: str) -> str:
        digest = hashlib.sha256()
        with open(pdf_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
