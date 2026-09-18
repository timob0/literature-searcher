from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from litsearch.config import settings
from litsearch.llm.models import DocumentSummary, SummaryEvidencePoint, SummaryFacetExtraction, SummarySection
from litsearch.llm.provider import LLMProvider
from litsearch.retrieval.context import ContextExpander
from litsearch.retrieval.models import ChunkHit, SearchFilters
from litsearch.retrieval.summary_boilerplate import is_boilerplate_for_facet
from litsearch.text_format import format_citation_with_pages

# Zotero itemType -> internal document-type label. Unknown/unlisted types fall back to "other".
_ZOTERO_ITEM_TYPE_MAP = {
    "journalArticle": "journal_article",
    "book": "book",
    "bookSection": "book_chapter",
    "conferencePaper": "conference_paper",
    "report": "report",
    "thesis": "thesis",
}

FACETS = ("topic", "problem_or_purpose", "methodology_or_approach", "finding_or_contribution", "limitation", "future_research")

# Facet-specific retrieval keywords (English + German, since the corpus is bilingual).
_FACET_QUERY_TERMS: dict[str, list[str]] = {
    "topic": ["topic", "focus", "scope", "this paper examines", "this book focuses on", "themes", "concepts", "Thema", "Schwerpunkt"],
    "problem_or_purpose": ["purpose", "aim", "objective", "research question", "problem", "this study investigates", "this book aims", "Zweck", "Ziel", "Forschungsfrage"],
    "methodology_or_approach": ["method", "methodology", "research design", "sample", "data", "analysis", "clinical research", "observational", "interviews", "survey", "experiment", "conceptual approach", "Methode", "Stichprobe"],
    "finding_or_contribution": ["results", "findings", "we found", "conclude", "shows", "demonstrates", "contribution", "argument", "proposition", "implications", "Ergebnisse", "Befunde"],
    "limitation": ["limitation", "limitations", "caution", "cannot", "restricted", "boundary", "first approximation", "generalizability", "Einschränkung", "Grenzen"],
    "future_research": ["future research", "further research", "future studies", "needs investigation", "should examine", "remains to be explored", "weitere Forschung", "zukünftige Forschung"],
}


@dataclass(slots=True)
class EvidencePassage:
    text: str
    page: int | None = None
    section: str | None = None
    citation_key: str | None = None
    paper_title: str | None = None
    zotero_link: str = ""
    score: float = 0.0
    evidence_id: str | None = None
    facet: str | None = None


@dataclass(slots=True)
class PaperSummaryResult:
    citation_key: str | None
    title: str | None
    document_type: str = "other"
    summary: DocumentSummary = field(default_factory=DocumentSummary)
    evidence_by_id: dict[str, EvidencePassage] = field(default_factory=dict)
    stage1_points: list[SummaryEvidencePoint] = field(default_factory=list)
    rejected_boilerplate: list[tuple[str, str]] = field(default_factory=list)
    insufficient_evidence: bool = False


class RetrieverLike(Protocol):
    def retrieve(self, query: str, filters=None, top_k: int = 10) -> list[ChunkHit]:
        ...


class ZoteroItemLookup(Protocol):
    def get_item_by_key(self, item_key: str) -> object | None:
        ...


class PaperSummaryStrategy:
    def __init__(self, retriever: RetrieverLike | None = None, llm: LLMProvider | None = None, zotero_client: ZoteroItemLookup | None = None) -> None:
        self.retriever = retriever
        self.llm = llm
        self.zotero_client = zotero_client

    def summarize(self, citation_key: str) -> PaperSummaryResult:
        if self.retriever is None:
            return PaperSummaryResult(citation_key=citation_key, title=None, insufficient_evidence=True)

        candidates_by_facet = self._retrieve_facet_candidates(citation_key)
        all_chunks = [chunk for chunks in candidates_by_facet.values() for chunk in chunks]
        if not all_chunks:
            return PaperSummaryResult(citation_key=citation_key, title=None, insufficient_evidence=True)

        title = next((chunk.title for chunk in all_chunks if chunk.title), None)
        parent_key = next((chunk.parent_key for chunk in all_chunks if chunk.parent_key), None)
        document_type = self._resolve_document_type(parent_key)

        expanded_by_facet, evidence_by_id, rejected = self._expand_candidates(candidates_by_facet)

        if self.llm is None or not settings.llm_enabled or not settings.paper_summary_llm_enabled:
            summary = DocumentSummary(citation_key=citation_key, title=title or "", document_type=document_type)
            return PaperSummaryResult(citation_key=citation_key, title=title, document_type=document_type, summary=summary, evidence_by_id=evidence_by_id, rejected_boilerplate=rejected, insufficient_evidence=True)

        points = self._extract_facet_evidence(title or citation_key, expanded_by_facet)
        summary = self._synthesize_summary(citation_key, title or citation_key, document_type, points, evidence_by_id)
        insufficient = not any(getattr(summary, name).points or getattr(summary, name).prose for name in ("overview", "topics", "problem_or_purpose", "methodology_or_approach", "findings_or_contributions"))

        return PaperSummaryResult(
            citation_key=citation_key,
            title=title,
            document_type=document_type,
            summary=summary,
            evidence_by_id=evidence_by_id,
            stage1_points=points,
            rejected_boilerplate=rejected,
            insufficient_evidence=insufficient,
        )

    def _resolve_document_type(self, parent_key: str | None) -> str:
        if not parent_key or self.zotero_client is None:
            return "other"
        try:
            item = self.zotero_client.get_item_by_key(parent_key)
        except Exception:
            return "other"
        item_type = getattr(item, "item_type", None) if item is not None else None
        return _ZOTERO_ITEM_TYPE_MAP.get(item_type or "", "other")

    def _retrieve_facet_candidates(self, citation_key: str) -> dict[str, list[ChunkHit]]:
        filters = SearchFilters(citation_key=citation_key)
        candidates: dict[str, list[ChunkHit]] = {}
        for facet in FACETS:
            hits: list[ChunkHit] = []
            for term in _FACET_QUERY_TERMS[facet]:
                hits.extend(self.retriever.retrieve(f"{citation_key} {term}", filters=filters, top_k=settings.summary_facet_top_k))
            deduped: dict[tuple[str | None, int, str], ChunkHit] = {}
            for hit in hits:
                key = (hit.citation_key, hit.page_start, hit.text[:80])
                existing = deduped.get(key)
                if existing is None or hit.score > existing.score:
                    deduped[key] = hit
            ranked = sorted(deduped.values(), key=lambda hit: hit.score, reverse=True)
            candidates[facet] = ranked[: settings.summary_facet_top_k]
        return candidates

    def _expand_candidates(self, candidates_by_facet: dict[str, list[ChunkHit]]) -> tuple[dict[str, list[tuple[str, ChunkHit]]], dict[str, EvidencePassage], list[tuple[str, str]]]:
        related_method = getattr(self.retriever, "related_chunks", None)
        documents: dict[str, list[ChunkHit]] = {}
        expanded_by_facet: dict[str, list[tuple[str, ChunkHit]]] = {}
        evidence_by_id: dict[str, EvidencePassage] = {}
        rejected: list[tuple[str, str]] = []
        counter = 0
        for facet, chunks in candidates_by_facet.items():
            expanded_by_facet[facet] = []
            seen_text: set[tuple[str, str]] = set()
            for chunk in chunks:
                if is_boilerplate_for_facet(chunk.text, facet):
                    rejected.append((facet, chunk.text[:120]))
                    continue
                dedup_key = (chunk.citation_key or "", chunk.text[:200])
                if dedup_key in seen_text:
                    continue
                seen_text.add(dedup_key)
                documents.setdefault(chunk.attachment_key, related_method(chunk) if related_method else [chunk])
                context = ContextExpander(documents[chunk.attachment_key]).expand(
                    chunk,
                    before=settings.summary_context_before,
                    after=settings.summary_context_after,
                    max_tokens=settings.summary_context_max_tokens,
                    continue_structures=True,
                )
                if is_boilerplate_for_facet(context.text, facet):
                    rejected.append((facet, context.text[:120]))
                    continue
                counter += 1
                evidence_id = f"E{counter:03d}"
                expanded_chunk = ChunkHit(
                    chunk_id=chunk.chunk_id, attachment_key=chunk.attachment_key, parent_key=chunk.parent_key,
                    citation_key=chunk.citation_key, title=chunk.title, authors=chunk.authors, year=chunk.year,
                    page_start=chunk.page_start, page_end=chunk.page_end, section_type=chunk.section_type,
                    section_title=chunk.section_title, text=context.text, score=chunk.score, zotero_link=chunk.zotero_link,
                )
                expanded_by_facet[facet].append((evidence_id, expanded_chunk))
                evidence_by_id[evidence_id] = EvidencePassage(
                    text=context.text, page=chunk.page_start or None, section=chunk.section_type,
                    citation_key=chunk.citation_key, paper_title=chunk.title, zotero_link=chunk.zotero_link,
                    score=chunk.score, evidence_id=evidence_id, facet=facet,
                )
        return expanded_by_facet, evidence_by_id, rejected

    def _extract_facet_evidence(self, document_title: str, expanded_by_facet: dict[str, list[tuple[str, ChunkHit]]]) -> list[SummaryEvidencePoint]:
        from litsearch.llm.prompts import summary_extract_messages

        points: list[SummaryEvidencePoint] = []
        batch_size = settings.summary_extraction_batch_evidence
        for facet, items in expanded_by_facet.items():
            for start in range(0, len(items), batch_size):
                batch = items[start:start + batch_size]
                packaged = [(evidence_id, chunk.text) for evidence_id, chunk in batch]
                if not packaged:
                    continue
                valid_ids = {evidence_id for evidence_id, _ in packaged}
                messages = summary_extract_messages(facet, document_title, packaged, max_chars=settings.summary_extraction_max_chars)
                try:
                    response = self.llm.structured_chat(messages, SummaryFacetExtraction, task_name="summary-extract-v2")
                except TypeError:
                    response = self.llm.structured_chat(messages, SummaryFacetExtraction)
                for point in response.points:
                    if point.facet != facet:
                        continue
                    if not point.evidence_ids or not set(point.evidence_ids) <= valid_ids:
                        continue
                    if not point.statement.strip():
                        continue
                    if facet in ("limitation", "future_research") and not point.explicitly_stated:
                        continue
                    points.append(point)
        return self._deduplicate_points(points)

    @staticmethod
    def _deduplicate_points(points: list[SummaryEvidencePoint]) -> list[SummaryEvidencePoint]:
        merged: dict[tuple[str, str], SummaryEvidencePoint] = {}
        order: list[tuple[str, str]] = []
        for point in points:
            key = (point.facet, re.sub(r"\W+", " ", point.statement.casefold()).strip())
            existing = merged.get(key)
            if existing is None:
                merged[key] = point
                order.append(key)
            else:
                existing.evidence_ids = list(dict.fromkeys(existing.evidence_ids + point.evidence_ids))
                if point.source_excerpt and (not existing.source_excerpt or len(point.source_excerpt) > len(existing.source_excerpt)):
                    existing.source_excerpt = point.source_excerpt
        ordered = [merged[key] for key in order]
        for index, point in enumerate(ordered, start=1):
            point.point_id = f"P{index:02d}"
        return ordered

    def _synthesize_summary(self, citation_key: str, document_title: str, document_type: str, points: list[SummaryEvidencePoint], evidence_by_id: dict[str, EvidencePassage]) -> DocumentSummary:
        from litsearch.llm.prompts import summary_synthesize_messages

        points_by_facet: dict[str, list[tuple[str, str]]] = {facet: [] for facet in FACETS}
        for point in points:
            points_by_facet.setdefault(point.facet, []).append((point.point_id, point.statement))

        if not any(points_by_facet.values()):
            return DocumentSummary(citation_key=citation_key, title=document_title, document_type=document_type)

        messages = summary_synthesize_messages(document_title, document_type, citation_key, points_by_facet)
        try:
            response = self.llm.structured_chat(messages, DocumentSummary, task_name="summary-synthesize-v2")
        except TypeError:
            response = self.llm.structured_chat(messages, DocumentSummary)

        valid_ids = {point.point_id for point in points}
        section_names = ("overview", "topics", "problem_or_purpose", "methodology_or_approach", "findings_or_contributions", "limitations", "future_research")
        references: list[list[str]] = []
        for section_name in section_names:
            section: SummarySection = getattr(response, section_name)
            section.points = [item for item in section.points if item.evidence_ids and set(item.evidence_ids) <= valid_ids]
            for item in section.points:
                item.statement = _strip_trailing_citation_bracket(item.statement)
            if section.prose:
                section.prose = _strip_trailing_citation_bracket(section.prose)
                merged_ids = list(dict.fromkeys(evidence_id for item in section.points for evidence_id in item.evidence_ids))
                if merged_ids:
                    references.append(merged_ids)
            else:
                references.extend(item.evidence_ids for item in section.points if item.evidence_ids)

        point_by_id = {point.point_id: point for point in points}
        aggregate_pages: dict[str, list[int]] = {}
        for reference_ids in references:
            pages_by_citation: dict[str, list[int]] = {}
            citation_keys: list[str] = []
            for point_id in reference_ids:
                point = point_by_id.get(point_id)
                for evidence_id in point.evidence_ids if point is not None else []:
                    source = evidence_by_id.get(evidence_id)
                    source_citation_key = source.citation_key if source else None
                    if not source_citation_key:
                        continue
                    if source_citation_key not in pages_by_citation:
                        pages_by_citation[source_citation_key] = []
                        citation_keys.append(source_citation_key)
                    aggregate_pages.setdefault(source_citation_key, [])
                    if source.page and source.page not in pages_by_citation[source_citation_key]:
                        pages_by_citation[source_citation_key].append(source.page)
                    if source.page and source.page not in aggregate_pages[source_citation_key]:
                        aggregate_pages[source_citation_key].append(source.page)
            response.citation_keys_by_reference["|".join(reference_ids)] = [
                format_citation_with_pages(source_citation_key, pages_by_citation[source_citation_key]) for source_citation_key in citation_keys
            ]
        response.citation_pages = aggregate_pages

        response.citation_key = citation_key
        response.title = document_title
        response.document_type = document_type
        return response


def _strip_trailing_citation_bracket(text: str) -> str:
    """Drop a model-added trailing "[P01, P02]"-style bracket so the renderer's own marker isn't duplicated."""
    return re.sub(r"\s*\[P\d+(?:,\s*P\d+)*\]\.?\s*$", "", text.strip()).strip()

