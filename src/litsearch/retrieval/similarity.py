from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from litsearch.config import settings
from litsearch.retrieval.summary_boilerplate import classify_boilerplate
from litsearch.storage.chroma import ChromaVectorStore


@dataclass(slots=True)
class SimilarityChunk:
    chunk_id: str
    attachment_key: str
    citation_key: str
    document_id: str
    chunk_index: int
    page_start: int | None
    page_end: int | None
    title: str
    section_type: str
    text: str
    embedding: list[float]


@dataclass(slots=True)
class SimilarityOverlap:
    source_chunk_id: str
    source_attachment_key: str
    source_chunk_index: int
    source_page_start: int | None
    source_page_end: int | None
    source_text: str
    source_excerpt: str
    candidate_chunk_id: str
    candidate_attachment_key: str
    candidate_chunk_index: int
    candidate_page_start: int | None
    candidate_page_end: int | None
    candidate_text: str
    candidate_excerpt: str
    similarity: float


@dataclass(slots=True)
class SimilarDocumentResult:
    citation_key: str
    title: str
    similarity_score: float
    semantic_strength: float
    coverage_ratio: float
    covered_source_chunks: int
    total_source_chunks: int
    overlaps: list[SimilarityOverlap] = field(default_factory=list)


@dataclass(slots=True)
class SimilarDocumentsSearch:
    source_citation_key: str
    source_title: str
    usable_source_chunks: int
    representative_source_chunks: int
    queries_executed: int
    candidate_chunk_hits: int
    candidate_publications: int
    candidates_passing_threshold: int
    results: list[SimilarDocumentResult] = field(default_factory=list)


class CitationKeyNotFoundError(ValueError):
    pass


class SimilaritySearchStrategy:
    """Paper-to-paper semantic retrieval using only stored Chroma embeddings."""

    _MIN_TEXT_LENGTH = 80
    _MIN_WORDS = 10
    _MAX_EXCERPT_LENGTH = 240
    _MAX_STRENGTH_PAIRS = 5
    _STRENGTH_WEIGHT = 0.70
    _COVERAGE_WEIGHT = 0.30
    _REPRESENTATIVE_CENTRALITY_WEIGHT = 0.20

    def __init__(self, vector_store: ChromaVectorStore | None = None) -> None:
        self.vector_store = vector_store

    def search(self, citation_key: str) -> SimilarDocumentsSearch:
        if self.vector_store is None:
            raise RuntimeError("Chroma is unavailable.")

        source_rows = self.vector_store.get_by_citation_key(citation_key)
        source_chunks = self._chunks_from_get(source_rows, citation_key)
        if not source_chunks:
            raise CitationKeyNotFoundError(f"Cannot resolve citation key: {citation_key}")

        usable_chunks = [chunk for chunk in source_chunks if self._is_usable(chunk)]
        source_title = next((chunk.title for chunk in source_chunks if chunk.title), citation_key)
        if not usable_chunks:
            return SimilarDocumentsSearch(
                source_citation_key=citation_key,
                source_title=source_title,
                usable_source_chunks=0,
                representative_source_chunks=0,
                queries_executed=0,
                candidate_chunk_hits=0,
                candidate_publications=0,
                candidates_passing_threshold=0,
            )

        representatives = self.select_representative_chunks(
            usable_chunks, max(1, settings.similar_source_chunks)
        )
        candidate_pairs: dict[str, dict[str, tuple[SimilarityChunk, SimilarityChunk, float]]] = {}
        candidate_titles: dict[str, str] = {}
        candidate_hit_count = 0

        for source in representatives:
            result = self.vector_store.query(
                source.embedding,
                n_results=max(1, settings.similar_neighbors_per_source_chunk),
                where={"citation_key": {"$ne": citation_key}},
            )
            hits = self._chunks_from_query(result)
            candidate_hit_count += len(hits)
            for candidate, similarity in hits:
                if candidate.citation_key == citation_key or not candidate.citation_key:
                    continue
                if not self._is_usable(candidate):
                    continue
                candidate_key = candidate.citation_key
                candidate_titles.setdefault(candidate_key, candidate.title or candidate_key)
                per_source = candidate_pairs.setdefault(candidate_key, {})
                existing = per_source.get(source.chunk_id)
                if existing is None or similarity > existing[2]:
                    per_source[source.chunk_id] = (source, candidate, similarity)

        ranked: list[SimilarDocumentResult] = []
        for candidate_key, per_source in candidate_pairs.items():
            ordered_pairs = sorted(per_source.values(), key=lambda pair: pair[2], reverse=True)
            if not ordered_pairs:
                continue
            covered = sum(
                similarity >= settings.similar_coverage_min_similarity
                for _, _, similarity in ordered_pairs
            )
            if not covered:
                continue
            coverage = covered / len(representatives)
            strength_pairs = ordered_pairs[: self._MAX_STRENGTH_PAIRS]
            strength = sum(pair[2] for pair in strength_pairs) / len(strength_pairs)
            final_score = self._STRENGTH_WEIGHT * strength + self._COVERAGE_WEIGHT * coverage
            ranked.append(
                SimilarDocumentResult(
                    citation_key=candidate_key,
                    title=candidate_titles[candidate_key],
                    similarity_score=final_score,
                    semantic_strength=strength,
                    coverage_ratio=coverage,
                    covered_source_chunks=covered,
                    total_source_chunks=len(representatives),
                    overlaps=self._select_overlaps(ordered_pairs),
                )
            )

        ranked.sort(key=lambda result: (result.similarity_score, result.semantic_strength), reverse=True)
        return SimilarDocumentsSearch(
            source_citation_key=citation_key,
            source_title=source_title,
            usable_source_chunks=len(usable_chunks),
            representative_source_chunks=len(representatives),
            queries_executed=len(representatives),
            candidate_chunk_hits=candidate_hit_count,
            candidate_publications=len(candidate_pairs),
            candidates_passing_threshold=len(ranked),
            results=ranked[: max(0, settings.similar_max_results)],
        )

    @classmethod
    def select_representative_chunks(
        cls, chunks: list[SimilarityChunk], limit: int
    ) -> list[SimilarityChunk]:
        if len(chunks) <= limit:
            return list(chunks)
        dimensions = len(chunks[0].embedding)
        centroid = [
            sum(
                chunk.embedding[dimension]
                for chunk in chunks
                if len(chunk.embedding) == dimensions
            ) / len(chunks)
            for dimension in range(dimensions)
        ]
        centrality = {
            chunk.chunk_id: cls._cosine(chunk.embedding, centroid)
            for chunk in chunks
        }
        selected = [max(chunks, key=lambda chunk: centrality[chunk.chunk_id])]
        remaining = [chunk for chunk in chunks if chunk.chunk_id != selected[0].chunk_id]
        while remaining and len(selected) < limit:
            def selection_score(chunk: SimilarityChunk) -> float:
                novelty = 1.0 - max(cls._cosine(chunk.embedding, chosen.embedding) for chosen in selected)
                normalized_centrality = (centrality[chunk.chunk_id] + 1.0) / 2.0
                return (
                    (1.0 - cls._REPRESENTATIVE_CENTRALITY_WEIGHT) * novelty
                    + cls._REPRESENTATIVE_CENTRALITY_WEIGHT * normalized_centrality
                )

            chosen = max(remaining, key=selection_score)
            selected.append(chosen)
            remaining.remove(chosen)
        return selected

    def _select_overlaps(
        self, pairs: list[tuple[SimilarityChunk, SimilarityChunk, float]]
    ) -> list[SimilarityOverlap]:
        candidates = [
            pair for pair in pairs
            if pair[2] >= settings.similar_coverage_min_similarity
        ]
        selected: list[tuple[SimilarityChunk, SimilarityChunk, float]] = []
        while candidates and len(selected) < max(0, settings.similar_overlaps_per_result):
            if not selected:
                chosen = max(candidates, key=lambda pair: pair[2])
            else:
                chosen = max(
                    candidates,
                    key=lambda pair: pair[2] * (
                        1.0 - max(
                            max(0.0, self._cosine(pair[0].embedding, prior[0].embedding))
                            for prior in selected
                        )
                    ),
                )
            selected.append(chosen)
            candidates.remove(chosen)
        return [
            SimilarityOverlap(
                source_chunk_id=source.chunk_id,
                source_attachment_key=source.attachment_key,
                source_chunk_index=source.chunk_index,
                source_page_start=source.page_start,
                source_page_end=source.page_end,
                source_text=source.text,
                source_excerpt=self._excerpt(source.text),
                candidate_chunk_id=candidate.chunk_id,
                candidate_attachment_key=candidate.attachment_key,
                candidate_chunk_index=candidate.chunk_index,
                candidate_page_start=candidate.page_start,
                candidate_page_end=candidate.page_end,
                candidate_text=candidate.text,
                candidate_excerpt=self._excerpt(candidate.text),
                similarity=similarity,
            )
            for source, candidate, similarity in selected
        ]

    @classmethod
    def _chunks_from_get(cls, result: dict[str, Any], citation_key: str) -> list[SimilarityChunk]:
        ids = cls._as_list(result.get("ids"))
        documents = cls._as_list(result.get("documents"))
        metadatas = cls._as_list(result.get("metadatas"))
        embeddings = cls._as_list(result.get("embeddings"))
        chunks: list[SimilarityChunk] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            if metadata.get("citation_key") != citation_key:
                continue
            embedding = embeddings[index] if index < len(embeddings) else None
            if embedding is None:
                continue
            chunk = cls._make_chunk(
                str(chunk_id),
                str(documents[index] or "") if index < len(documents) else "",
                metadata,
                [float(value) for value in embedding],
            )
            chunks.append(chunk)
        chunks.sort(key=lambda chunk: (chunk.chunk_index, chunk.chunk_id))
        return chunks

    def _chunks_from_query(
        self, result: dict[str, Any]
    ) -> list[tuple[SimilarityChunk, float]]:
        ids = self._first_result_list(result.get("ids", []))
        documents = self._first_result_list(result.get("documents", []))
        metadatas = self._first_result_list(result.get("metadatas", []))
        distances = self._first_result_list(result.get("distances", []))
        chunks: list[tuple[SimilarityChunk, float]] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            chunk = self._make_chunk(
                str(chunk_id),
                str(documents[index] or "") if index < len(documents) else "",
                metadata,
                [],
            )
            distance = float(distances[index]) if index < len(distances) else 1.0
            converter = getattr(self.vector_store, "similarity_from_distance", None)
            similarity = converter(distance) if converter is not None else 1.0 - distance
            chunks.append((chunk, similarity))
        return chunks

    @staticmethod
    def _first_result_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list) and value and isinstance(value[0], list):
            return value[0]
        return list(value) if isinstance(value, (list, tuple)) else [value]

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        return list(value) if not isinstance(value, list) else value

    @staticmethod
    def _make_chunk(
        chunk_id: str, text: str, metadata: dict[str, Any], embedding: list[float]
    ) -> SimilarityChunk:
        def optional_page(name: str) -> int | None:
            value = metadata.get(name)
            return int(value) if value not in (None, "") and int(value) > 0 else None

        return SimilarityChunk(
            chunk_id=str(metadata.get("chunk_id") or chunk_id),
            attachment_key=str(metadata.get("attachment_key") or ""),
            citation_key=str(metadata.get("citation_key") or ""),
            document_id=str(metadata.get("document_id") or metadata.get("parent_key") or ""),
            chunk_index=int(metadata.get("chunk_index") or 0),
            page_start=optional_page("page_start"),
            page_end=optional_page("page_end"),
            title=str(metadata.get("title") or ""),
            section_type=str(metadata.get("section_type") or "").casefold(),
            text=text.replace("\x00", "").strip(),
            embedding=embedding,
        )

    @classmethod
    def _is_usable(cls, chunk: SimilarityChunk) -> bool:
        text = chunk.text
        if len(text) < cls._MIN_TEXT_LENGTH or len(text.split()) < cls._MIN_WORDS:
            return False
        if classify_boilerplate(text):
            return False
        return chunk.section_type not in {"references", "bibliography", "index"}

    @classmethod
    def _excerpt(cls, text: str) -> str:
        normalized = " ".join(text.split())
        first_sentence = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)[0]
        excerpt = first_sentence if len(first_sentence) >= 40 else normalized
        if len(excerpt) > cls._MAX_EXCERPT_LENGTH:
            excerpt = excerpt[: cls._MAX_EXCERPT_LENGTH - 3].rsplit(" ", 1)[0] + "..."
        return excerpt

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if len(left) != len(right) or not left:
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if not left_norm or not right_norm:
            return 0.0
        return max(-1.0, min(1.0, dot / (left_norm * right_norm)))
