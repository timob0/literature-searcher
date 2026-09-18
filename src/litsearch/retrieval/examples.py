from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Protocol

from litsearch.config import settings
from litsearch.llm.models import ExampleAssessmentBatch, ExampleExtractionResult, ExtractedExample
from litsearch.llm.provider import LLMProvider
from litsearch.llm.errors import LLMError
from litsearch.retrieval.models import ChunkHit
from litsearch.retrieval.models import SearchFilters
from litsearch.retrieval.context import ContextExpander


@dataclass(slots=True)
class ExampleHit:
    concept: str
    example_text: str
    paper_title: str | None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    citation_key: str | None = None
    page: int | None = None
    section: str | None = None
    score: float = 0.0
    zotero_link: str = ""
    classification: str | None = None
    llm_confidence: str | None = None
    evidence_ids: list[str] = field(default_factory=list)


class RetrieverLike(Protocol):
    def retrieve(self, query: str, filters=None, top_k: int = 10) -> list[ChunkHit]:
        ...


class ExampleSearchStrategy:
    def __init__(self, retriever: RetrieverLike | None = None, llm: LLMProvider | None = None) -> None:
        self.retriever = retriever
        self.llm = llm

    def _candidate_queries(self, concept: str) -> list[str]:
        base = concept.strip()
        if not base:
            return []
        return [
            base,
            f"examples of {base}",
            f"specific {base} behaviors",
            f"practical manifestations of {base}",
            f"manager behaviours demonstrating {base}",
            f"concrete examples of {base}",
            f"{base} practices",
            f"observable {base} actions",
        ]

    def _extract_example_text(self, chunk: ChunkHit) -> str:
        text = chunk.text.strip()
        if not text:
            return ""
        for phrase in ["for example", "such as", "including", "specifically", "managers", "leaders", "supervisors", "employees", "held one-to-one", "adjusted workload", "involved employees"]:
            if phrase in text.lower():
                return text
        for sentence in text.split(". "):
            sentence = sentence.strip()
            if any(action in sentence.lower() for action in ["held", "adjusted", "involved", "encouraged", "checked", "regularly", "reported", "discussed", "supported", "managed"]):
                return sentence
        return text.split(". ")[0] if ". " in text else text

    def search(self, concept: str, top_k: int = 5, *, filters: SearchFilters | None = None) -> list[ExampleHit]:
        if self.retriever is None:
            return []

        chunks: list[ChunkHit] = []
        seen: set[tuple[str | None, int, str]] = set()
        for query in self._candidate_queries(concept):
            for chunk in self.retriever.retrieve(query, filters=filters, top_k=settings.examples_candidate_top_k):
                key = (chunk.citation_key, chunk.page_start, chunk.text[:80])
                if key in seen:
                    continue
                seen.add(key)
                chunks.append(chunk)
            chunks.sort(key=lambda chunk: chunk.score, reverse=True)
            chunks = chunks[:settings.examples_candidate_top_k]

        if self.llm is not None and settings.llm_enabled and settings.examples_llm_enabled:
            try:
                return self._extract_examples(concept, chunks)
            except LLMError:
                raise
            except Exception:
                pass
        return [self._to_hit(concept, chunk, classification="unclassified_candidate") for chunk in chunks[:top_k]]

    def _to_hit(self, concept: str, chunk: ChunkHit, *, classification: str | None = None, confidence: str | None = None) -> ExampleHit:
        return ExampleHit(
            concept=concept,
            example_text=self._extract_example_text(chunk),
            paper_title=chunk.title,
            authors=chunk.authors,
            year=chunk.year,
            citation_key=chunk.citation_key,
            page=chunk.page_start or None,
            section=chunk.section_type,
            score=chunk.score,
            zotero_link=chunk.zotero_link,
            classification=classification,
            llm_confidence=confidence,
        )

    def _classify_examples(self, concept: str, chunks: list[ChunkHit], top_k: int) -> list[ExampleHit]:
        from litsearch.llm.prompts import examples_messages

        results: list[ExampleHit] = []
        for start in range(0, min(len(chunks), 5), 5):
            batch = chunks[start:start + 5]
            evidence = [(f"P{index:02d}", chunk.text) for index, chunk in enumerate(batch, start=start + 1)]
            response = self.llm.structured_chat(examples_messages(concept, evidence), ExampleAssessmentBatch)
            by_id = {f"P{index:02d}": chunk for index, chunk in enumerate(batch, start=start + 1)}
            for assessment in response.assessments:
                chunk = by_id.get(assessment.evidence_id)
                if chunk is None or assessment.classification != "concrete_example":
                    continue
                results.append(self._to_hit(concept, chunk, classification=assessment.classification, confidence=assessment.confidence))
                if len(results) >= top_k:
                    return results
        return results

    def _extract_examples(self, concept: str, anchors: list[ChunkHit]) -> list[ExampleHit]:
        from litsearch.llm.prompts import examples_extract_messages

        related_method = getattr(self.retriever, "related_chunks", None)
        expanded: list[tuple[ChunkHit, float]] = []
        documents: dict[str, list[ChunkHit]] = {}
        for anchor in anchors:
            documents.setdefault(anchor.attachment_key, related_method(anchor) if related_method else [anchor])
            context = ContextExpander(documents[anchor.attachment_key]).expand(
                anchor,
                before=settings.examples_context_before,
                after=settings.examples_context_after,
                max_tokens=settings.examples_context_max_tokens,
                continue_structures=True,
            )
            expanded.append((self._with_text(anchor, context.text), anchor.score))
        unique: dict[tuple[str | None, str], tuple[ChunkHit, float]] = {}
        for item in expanded:
            unique.setdefault((item[0].citation_key, item[0].text[:500]), item)
        extracted: list[tuple[ChunkHit, float, ExtractedExample]] = []
        items = list(unique.values())
        for start in range(0, len(items), settings.examples_extraction_batch_evidence):
            batch = items[start:start + settings.examples_extraction_batch_evidence]
            evidence = [(f"E{index:02d}", chunk.text) for index, (chunk, _) in enumerate(batch, start=start + 1)]
            messages = examples_extract_messages(concept, evidence, max_chars=settings.examples_extraction_max_chars)
            try:
                response = self.llm.structured_chat(messages, ExampleExtractionResult, task_name="examples-extract-v2")
            except TypeError:
                response = self.llm.structured_chat(messages, ExampleExtractionResult)
            by_id = {f"E{index:02d}": item for index, item in enumerate(batch, start=start + 1)}
            for example in response.examples:
                if example.concept_match != "requested_concept" or example.confidence == "low" or not example.evidence_ids or not set(example.evidence_ids) <= set(by_id):
                    continue
                chunk, score = by_id[example.evidence_ids[0]]
                extracted.append((chunk, score, example))
        return self._deduplicate_examples(concept, extracted)

    @staticmethod
    def _with_text(anchor: ChunkHit, text: str) -> ChunkHit:
        return ChunkHit(
            chunk_id=anchor.chunk_id, attachment_key=anchor.attachment_key, parent_key=anchor.parent_key,
            citation_key=anchor.citation_key, title=anchor.title, authors=anchor.authors, year=anchor.year,
            page_start=anchor.page_start, page_end=anchor.page_end, section_type=anchor.section_type,
            section_title=anchor.section_title, text=text, score=anchor.score, zotero_link=anchor.zotero_link,
        )

    def _deduplicate_examples(self, concept: str, extracted: list[tuple[ChunkHit, float, ExtractedExample]]) -> list[ExampleHit]:
        unique: dict[str, tuple[ChunkHit, float, ExtractedExample]] = {}
        confidence_rank = {"high": 3, "medium": 2, "low": 1}
        for item in extracted:
            key = re.sub(r"\W+", " ", item[2].statement.casefold()).strip()
            previous = unique.get(key)
            if previous is None or (confidence_rank[item[2].confidence], item[1]) > (confidence_rank[previous[2].confidence], previous[1]):
                unique[key] = item
        ordered = sorted(unique.values(), key=lambda item: (confidence_rank[item[2].confidence], item[1]), reverse=True)[:settings.examples_max_results]
        return [
            ExampleHit(concept=concept, example_text=example.statement, paper_title=chunk.title, authors=chunk.authors, year=chunk.year, citation_key=chunk.citation_key, page=chunk.page_start or None, section=chunk.section_type, score=score, zotero_link=chunk.zotero_link, classification="concrete_example", llm_confidence=example.confidence, evidence_ids=example.evidence_ids)
            for chunk, score, example in ordered
        ]
