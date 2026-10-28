from __future__ import annotations

from dataclasses import dataclass
from io import StringIO

import pytest
import chromadb
from rich.console import Console

from litsearch.config import settings
from litsearch.retrieval.similarity import (
    CitationKeyNotFoundError,
    SimilarityChunk,
    SimilaritySearchStrategy,
)
from litsearch.shell.renderer import ShellRenderer
from litsearch.storage.chroma import ChromaVectorStore


SUBSTANTIVE_TEXT = (
    "Organizational culture shapes how people interpret shared experiences, "
    "coordinate their actions, and reproduce assumptions through everyday work."
)


def _metadata(
    citation_key: str,
    chunk_index: int,
    *,
    page_start: int = 1,
    page_end: int | None = None,
    title: str | None = None,
) -> dict:
    return {
        "chunk_id": f"{citation_key}:{chunk_index}",
        "attachment_key": f"attachment-{citation_key}",
        "citation_key": citation_key,
        "document_id": citation_key,
        "chunk_index": chunk_index,
        "page_start": page_start,
        "page_end": page_end if page_end is not None else page_start,
        "title": title or citation_key,
    }


def _query_row(
    citation_key: str,
    chunk_index: int,
    similarity: float,
    *,
    page_start: int = 1,
    page_end: int | None = None,
) -> tuple[str, str, dict, float]:
    return (
        f"{citation_key}:{chunk_index}",
        SUBSTANTIVE_TEXT,
        _metadata(citation_key, chunk_index, page_start=page_start, page_end=page_end),
        1 - similarity,
    )


@dataclass
class FakeVectorStore:
    source_rows: dict
    query_rows: dict[int, list[tuple[str, str, dict, float]]]
    query_calls: list[tuple[list[float], int, dict | None]]

    def get_by_citation_key(self, citation_key: str) -> dict:
        if "ids" in self.source_rows:
            return self.source_rows
        return self.source_rows.get(citation_key, {"ids": [], "documents": [], "metadatas": [], "embeddings": []})

    def query(self, embedding: list[float], n_results: int, where=None) -> dict:
        self.query_calls.append((embedding, n_results, where))
        source_index = embedding.index(1.0)
        rows = self.query_rows.get(source_index, [])
        return {
            "ids": [[row[0] for row in rows]],
            "documents": [[row[1] for row in rows]],
            "metadatas": [[row[2] for row in rows]],
            "distances": [[row[3] for row in rows]],
        }


def _source_rows(citation_key: str, count: int, dimensions: int | None = None) -> dict:
    dimensions = dimensions or count
    return {
        "ids": [f"{citation_key}:{index}" for index in range(count)],
        "documents": [SUBSTANTIVE_TEXT] * count,
        "metadatas": [_metadata(citation_key, index, page_start=index + 1) for index in range(count)],
        "embeddings": [
            [1.0 if index == dimension else 0.0 for dimension in range(dimensions)]
            for index in range(count)
        ],
    }


def test_chroma_cosine_distances_aggregate_papers_by_distinct_source_aspects(monkeypatch) -> None:
    source_count = 15
    query_rows = {}
    for index in range(source_count):
        rows = [_query_row("sourcePaper", index, 0.99)]
        if index == 0:
            rows.append(_query_row("singleMatchPaper", index, 0.97))
        if index < 8:
            rows.append(_query_row("broadMatchPaper", index, 0.85))
        query_rows[index] = rows
    store = FakeVectorStore(_source_rows("sourcePaper", source_count), query_rows, [])
    monkeypatch.setattr(settings, "similar_source_chunks", source_count)

    result = SimilaritySearchStrategy(store).search("sourcePaper")

    assert [document.citation_key for document in result.results] == [
        "broadMatchPaper",
        "singleMatchPaper",
    ]
    assert result.results[0].covered_source_chunks == 8
    assert result.results[0].total_source_chunks == source_count
    assert result.results[0].similarity_score > result.results[1].similarity_score
    assert all(call[2] == {"citation_key": {"$ne": "sourcePaper"}} for call in store.query_calls)
    assert result.queries_executed == source_count
    assert result.candidate_chunk_hits == source_count + 1 + 8


def test_adjacent_candidate_chunks_do_not_inflate_source_coverage(monkeypatch) -> None:
    source_rows = _source_rows("sourcePaper", 2)
    store = FakeVectorStore(
        source_rows,
        {
            0: [
                _query_row("candidate", index, 0.9)
                for index in (40, 41, 42)
            ]
        },
        [],
    )
    monkeypatch.setattr(settings, "similar_source_chunks", 2)

    result = SimilaritySearchStrategy(store).search("sourcePaper")

    assert len(result.results) == 1
    assert result.results[0].covered_source_chunks == 1
    assert result.results[0].coverage_ratio == 0.5
    assert len(result.results[0].overlaps) == 1


def test_candidates_without_a_covered_source_aspect_are_not_returned(monkeypatch) -> None:
    store = FakeVectorStore(
        _source_rows("sourcePaper", 1),
        {0: [_query_row("weakMatch", 0, 0.4)]},
        [],
    )
    monkeypatch.setattr(settings, "similar_source_chunks", 1)

    result = SimilaritySearchStrategy(store).search("sourcePaper")

    assert result.results == []


def test_representative_chunk_selection_preserves_distinct_topics() -> None:
    chunks = []
    for index in range(12):
        topic = 0 if index < 8 else (1 if index < 10 else 2)
        embedding = [0.0, 0.0, 0.0]
        embedding[topic] = 1.0
        chunks.append(
            SimilarityChunk(
                chunk_id=str(index),
                attachment_key="attachment",
                citation_key="paper",
                document_id="paper",
                chunk_index=index,
                page_start=index + 1,
                page_end=index + 1,
                title="Paper",
                section_type="other",
                text=SUBSTANTIVE_TEXT,
                embedding=embedding,
            )
        )

    selected = SimilaritySearchStrategy.select_representative_chunks(chunks, 3)

    assert len(selected) == 3
    assert {chunk.embedding.index(1.0) for chunk in selected} == {0, 1, 2}


def test_long_book_source_is_capped_to_configured_queries(monkeypatch) -> None:
    source = _source_rows("longBook", 800, dimensions=2)
    source["embeddings"] = [[1.0, 0.0] for _ in range(800)]
    store = FakeVectorStore(source, {}, [])
    monkeypatch.setattr(settings, "similar_source_chunks", 15)

    result = SimilaritySearchStrategy(store).search("longBook")

    assert result.usable_source_chunks == 800
    assert result.representative_source_chunks == 15
    assert result.queries_executed == 15


def test_short_article_uses_all_source_chunks_without_duplication(monkeypatch) -> None:
    store = FakeVectorStore(_source_rows("shortArticle", 3), {}, [])
    monkeypatch.setattr(settings, "similar_source_chunks", 15)

    result = SimilaritySearchStrategy(store).search("shortArticle")

    assert result.usable_source_chunks == 3
    assert result.representative_source_chunks == 3
    assert result.queries_executed == 3


def test_source_and_candidate_page_metadata_is_retained_and_rendered(monkeypatch) -> None:
    source = _source_rows("sourcePaper", 1)
    candidate_row = _query_row("candidatePaper", 4, 0.91, page_start=17, page_end=18)
    store = FakeVectorStore(source, {0: [candidate_row]}, [])
    monkeypatch.setattr(settings, "similar_source_chunks", 1)
    source["metadatas"][0]["page_start"] = 4
    source["metadatas"][0]["page_end"] = 4

    result = SimilaritySearchStrategy(store).search("sourcePaper")
    overlap = result.results[0].overlaps[0]
    output = StringIO()
    ShellRenderer(Console(file=output, force_terminal=False, width=140)).render_similar_documents(
        result, debug=True
    )

    assert overlap.source_page_start == 4
    assert overlap.candidate_page_start == 17
    assert overlap.candidate_page_end == 18
    assert "Source (PDF p. 4)" in output.getvalue()
    assert "Match (PDF pp. 17–18)" in output.getvalue()
    assert "Queries executed: 1" in output.getvalue()


def test_missing_source_citation_key_is_reported() -> None:
    store = FakeVectorStore({}, {}, [])

    with pytest.raises(CitationKeyNotFoundError, match="Cannot resolve citation key: missing"):
        SimilaritySearchStrategy(store).search("missing")


def test_unusable_source_chunks_are_excluded() -> None:
    source = _source_rows("sourcePaper", 1)
    source["documents"] = ["All rights reserved. " + "Copyright terms. " * 10]
    store = FakeVectorStore(source, {}, [])

    result = SimilaritySearchStrategy(store).search("sourcePaper")

    assert result.usable_source_chunks == 0
    assert result.results == []


def test_chroma_integration_uses_stored_embeddings_and_page_metadata(monkeypatch) -> None:
    client = chromadb.EphemeralClient()
    monkeypatch.setattr("litsearch.storage.chroma.chromadb.HttpClient", lambda **kwargs: client)
    monkeypatch.setattr(settings, "similar_source_chunks", 1)
    monkeypatch.setattr(settings, "similar_neighbors_per_source_chunk", 20)
    store = ChromaVectorStore(collection_name="paper_similarity_test")
    store.add(
        ids=["source:1", "candidate:1"],
        texts=[SUBSTANTIVE_TEXT, SUBSTANTIVE_TEXT],
        metadatas=[
            _metadata("sourcePaper", 1, page_start=4),
            _metadata("candidatePaper", 1, page_start=17, page_end=18),
        ],
        embeddings=[[1.0, 0.0], [0.9, 0.1]],
    )

    result = SimilaritySearchStrategy(store).search("sourcePaper")

    assert [document.citation_key for document in result.results] == ["candidatePaper"]
    overlap = result.results[0].overlaps[0]
    assert overlap.source_page_start == 4
    assert overlap.candidate_page_start == 17
    assert overlap.candidate_page_end == 18
    assert overlap.similarity > 0.9
