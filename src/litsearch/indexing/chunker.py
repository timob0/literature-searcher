from __future__ import annotations

import hashlib

from litsearch.documents.models import DocumentPage

from litsearch.documents.models import Chunk


class SimpleChunker:
    def __init__(self, target_tokens: int = 500, min_tokens: int = 200, max_tokens: int = 800, overlap_tokens: int = 75):
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_text(self, text: str, *, attachment_key: str, parent_key: str, citation_key: str | None, chunk_index: int = 0, page_start: int = 1, page_end: int | None = None) -> Chunk:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        chunk_id = f"{attachment_key}:{chunk_index}:{digest}"
        return Chunk(
            chunk_id=chunk_id,
            attachment_key=attachment_key,
            parent_key=parent_key,
            citation_key=citation_key,
            text=text.strip(),
            page_start=page_start,
            page_end=page_end if page_end is not None else page_start,
            section_type="other",
            section_title=None,
            sentence_ids=[],
            paragraph_ids=[],
            chunk_index=chunk_index,
            content_hash=digest,
        )

    def chunk_document(self, document_text: str, *, attachment_key: str, parent_key: str, citation_key: str | None) -> list[Chunk]:
        text = document_text.strip()
        if not text:
            return []

        words = text.split()
        if len(words) <= self.max_tokens:
            segments = [text]
        else:
            window_size = min(self.target_tokens, self.max_tokens)
            overlap = min(self.overlap_tokens, window_size - 1)
            step = max(1, window_size - overlap)
            segments = []
            start = 0
            while start < len(words):
                end = min(start + window_size, len(words))
                remaining = len(words) - end
                if remaining and remaining < self.min_tokens:
                    end = len(words)
                segments.append(" ".join(words[start:end]))
                if end >= len(words):
                    break
                start += step

        chunks: list[Chunk] = []
        for idx, segment in enumerate(segments):
            chunks.append(self.chunk_text(segment, attachment_key=attachment_key, parent_key=parent_key, citation_key=citation_key, chunk_index=idx))
        return chunks

    def chunk_pages(self, pages: list[DocumentPage], *, attachment_key: str, parent_key: str, citation_key: str | None) -> list[Chunk]:
        """Chunk ordered PDF text while retaining each token's physical page."""
        tokens: list[tuple[str, int]] = []
        for page in pages:
            if page.text.strip():
                tokens.extend((word, page.page_number) for word in page.text.split())
        if not tokens:
            return []

        if len(tokens) <= self.max_tokens:
            ranges = [(0, len(tokens))]
        else:
            window_size = min(self.target_tokens, self.max_tokens)
            overlap = min(self.overlap_tokens, window_size - 1)
            step = max(1, window_size - overlap)
            ranges = []
            start = 0
            while start < len(tokens):
                end = min(start + window_size, len(tokens))
                remaining = len(tokens) - end
                if remaining and remaining < self.min_tokens:
                    end = len(tokens)
                ranges.append((start, end))
                if end >= len(tokens):
                    break
                start += step

        chunks: list[Chunk] = []
        for index, (start, end) in enumerate(ranges):
            segment = tokens[start:end]
            chunks.append(self.chunk_text(
                " ".join(word for word, _ in segment),
                attachment_key=attachment_key,
                parent_key=parent_key,
                citation_key=citation_key,
                chunk_index=index,
                page_start=segment[0][1],
                page_end=segment[-1][1],
            ))
        return chunks
