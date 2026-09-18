from __future__ import annotations

import re
from dataclasses import dataclass, field

from litsearch.retrieval.models import ChunkHit


@dataclass(slots=True)
class ExpandedEvidence:
    evidence_id: str
    document_id: str
    parent_key: str | None
    attachment_key: str | None
    citation_key: str | None
    anchor_chunk_id: str
    anchor_chunk_index: int
    anchor_score: float | None
    context_chunk_ids: list[str]
    context_start_index: int
    context_end_index: int
    text: str
    expansion_reason: list[str] = field(default_factory=list)


class ContextExpander:
    def __init__(self, chunks: list[ChunkHit]) -> None:
        self.chunks = chunks

    @staticmethod
    def chunk_index(chunk: ChunkHit) -> int:
        try:
            return int(chunk.chunk_id.split(":", 2)[1])
        except (IndexError, ValueError):
            return 0

    @staticmethod
    def _tokens(text: str) -> int:
        return len(text.split())

    @staticmethod
    def _stitch(previous: str, current: str) -> str:
        left = previous.strip()
        right = current.strip()
        left_normalized = re.sub(r"\s+", " ", left)
        right_normalized = re.sub(r"\s+", " ", right)
        left_words = left_normalized.split()
        right_words = right_normalized.split()
        maximum = min(75, len(left_words), len(right_words))
        overlap = 0
        for size in range(maximum, 3, -1):
            if left_words[-size:] == right_words[:size]:
                overlap = size
                break
        remainder = " ".join(right_words[overlap:])
        if not remainder:
            return left
        separator = "\n" if re.match(r"(?:[•*+-]|\d+[.)]|[a-z][.)]|\([ivx]+\))", remainder, re.IGNORECASE) else " "
        return f"{left}{separator}{remainder}".strip()

    def expand(self, anchor: ChunkHit, *, before: int = 1, after: int = 1, max_tokens: int = 2500, continue_structures: bool = True) -> ExpandedEvidence:
        same_document = sorted(
            (chunk for chunk in self.chunks if chunk.attachment_key == anchor.attachment_key),
            key=self.chunk_index,
        )
        anchor_index = self.chunk_index(anchor)
        by_index = {self.chunk_index(chunk): chunk for chunk in same_document}
        selected = [anchor_index]
        reasons: list[str] = []
        for index in range(anchor_index - before, anchor_index + after + 1):
            if index in by_index and index not in selected:
                selected.append(index)
        selected.sort()
        if continue_structures:
            while selected and selected[-1] + 1 in by_index and self._continues_structure(by_index[selected[-1]], by_index[selected[-1] + 1]):
                selected.append(selected[-1] + 1)
                reasons.append("list or incomplete-sentence continuation")
            while selected and selected[0] - 1 in by_index and self._continues_structure(by_index[selected[0] - 1], by_index[selected[0]]):
                selected.insert(0, selected[0] - 1)
                reasons.append("list introduction or incomplete-sentence continuation")
        text = ""
        final_indices: list[int] = []
        for index in selected:
            if index not in by_index:
                continue
            candidate = by_index[index]
            stitched = self._stitch(text, candidate.text) if text else candidate.text.strip()
            if self._tokens(stitched) > max_tokens and index != anchor_index:
                continue
            text = stitched
            final_indices.append(index)
        return ExpandedEvidence(
            evidence_id="",
            document_id=anchor.parent_key or anchor.attachment_key,
            parent_key=anchor.parent_key,
            attachment_key=anchor.attachment_key,
            citation_key=anchor.citation_key,
            anchor_chunk_id=anchor.chunk_id,
            anchor_chunk_index=anchor_index,
            anchor_score=anchor.score,
            context_chunk_ids=[by_index[index].chunk_id for index in final_indices],
            context_start_index=min(final_indices, default=anchor_index),
            context_end_index=max(final_indices, default=anchor_index),
            text=text,
            expansion_reason=reasons,
        )

    @staticmethod
    def _continues_structure(previous: ChunkHit, current: ChunkHit) -> bool:
        previous_text = previous.text.rstrip()
        current_text = current.text.lstrip()
        marker = r"(?:[•*+-]|\d+[.)]|[a-z][.)]|\([ivx]+\))"
        return (
            previous_text.endswith(":")
            or bool(re.match(marker, current_text, re.IGNORECASE))
            or not re.search(r"[.!?][\"')\]]?$", previous_text)
        )