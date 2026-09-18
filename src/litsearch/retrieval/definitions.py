from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata
from typing import Protocol

from litsearch.config import settings
from litsearch.llm.models import (
    ConceptualEvidenceExtraction,
    ConceptualEvidencePoint,
    ConceptExplanation,
    DefinitionAssessmentBatch,
    DefinitionEvidenceBatch,
    DefinitionResult,
    ValidatedDefinition,
)
from litsearch.llm.provider import LLMProvider
from litsearch.llm.errors import LLMError
from litsearch.retrieval.models import ChunkHit, SearchFilters
from litsearch.text_format import format_citation_with_pages
from litsearch.retrieval.context import ContextExpander


@dataclass(slots=True)
class DefinitionHit:
    concept: str
    definition_text: str
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
    lexical_match: str | None = None
    candidate_score: float = 0.0


class RetrieverLike(Protocol):
    def retrieve(self, query: str, filters=None, top_k: int = 10) -> list[ChunkHit]:
        ...


class DefinitionSearchStrategy:
    def __init__(self, retriever: RetrieverLike | None = None, llm: LLMProvider | None = None) -> None:
        self.retriever = retriever
        self.llm = llm

    def _candidate_queries(self, concept: str) -> list[str]:
        base = concept.strip()
        if not base:
            return []
        concepts = [base]
        alternate = self._british_american_variant(base)
        if alternate != base.casefold():
            concepts.append(alternate)
        variants = []
        for query_concept in concepts:
            variants.extend([
                query_concept,
                f"definition of {query_concept}",
                f"how {query_concept} is defined",
                f"what is meant by {query_concept}",
                f"conceptualization of {query_concept}",
                f"conceptualisation of {query_concept}",
                f"understanding of {query_concept}",
            ])
        return list(dict.fromkeys(variants))

    @staticmethod
    def _british_american_variant(text: str) -> str:
        replacements = {
            "organisation": "organization",
            "organisations": "organizations",
            "organisational": "organizational",
            "behaviour": "behavior",
            "behaviours": "behaviors",
            "labour": "labor",
            "centre": "center",
            "catalogue": "catalog",
            "programme": "program",
        }
        variant = text.casefold()
        for british, american in replacements.items():
            variant = re.sub(rf"\b{british}\b", american, variant)
        return variant

    def _extract_definition_text(self, chunk: ChunkHit) -> str:
        sentence_candidates = []
        text = chunk.text.strip()
        if not text:
            return ""
        for sentence in text.split(". "):
            sentence = sentence.strip()
            if sentence and any(token in sentence.lower() for token in ["is the", "refers to", "defined as", "means", "understood as", "conceptualized as", "conceptualised as"]):
                sentence_candidates.append(sentence)
        if sentence_candidates:
            return sentence_candidates[0]
        return text.split(". ")[0] if ". " in text else text

    @staticmethod
    def _normalize_phrase(text: str) -> str:
        text = unicodedata.normalize("NFKC", text).casefold()
        text = re.sub(r"[\u2010-\u2015\-]+", " ", text)
        return DefinitionSearchStrategy._british_american_variant(re.sub(r"\s+", " ", text).strip())

    @classmethod
    def _same_concept(cls, requested: str, returned: str) -> bool:
        return cls._normalize_phrase(requested) == cls._normalize_phrase(returned)

    @classmethod
    def lexical_match_level(cls, concept: str, text: str) -> str:
        raw_concept = unicodedata.normalize("NFKC", concept).casefold().strip()
        raw_text = unicodedata.normalize("NFKC", text).casefold()
        if raw_concept and raw_concept in raw_text:
            return "exact_phrase"
        normalized_concept = cls._normalize_phrase(concept)
        normalized_text = cls._normalize_phrase(text)
        if normalized_concept and normalized_concept in normalized_text:
            return "normalized_phrase"
        terms = [term for term in normalized_concept.split() if len(term) > 2]
        if terms and any(re.search(rf"\b{re.escape(term)}\b", normalized_text) for term in terms):
            return "partial_term_match"
        return "no_match"

    @staticmethod
    def _reference_like(text: str) -> bool:
        lowered = text.casefold()
        doi_count = len(re.findall(r"\bdoi\s*:\s*|https?://doi\.org/", lowered))
        author_year_count = len(re.findall(r"\b[A-Z][a-z]+(?:\s+et\s+al\.)?[, ]+\(?20\d{2}\)?", text))
        year_count = len(re.findall(r"\b(?:19|20)\d{2}\b", text))
        return doi_count >= 2 or (author_year_count >= 3 and year_count >= 4)

    def _candidate_score(self, concept: str, chunk: ChunkHit) -> tuple[float, str]:
        level = self.lexical_match_level(concept, chunk.text)
        bonuses = {
            "exact_phrase": settings.define_exact_phrase_bonus,
            "normalized_phrase": settings.define_normalized_phrase_bonus,
            "partial_term_match": settings.define_partial_term_bonus,
            "no_match": 0.0,
        }
        score = chunk.score + bonuses[level]
        if chunk.section_type == "references":
            score -= settings.define_reference_section_penalty
        elif self._reference_like(chunk.text):
            score -= settings.define_reference_section_penalty
        elif chunk.section_type in {"abstract", "introduction", "literature_review", "theory"}:
            score += settings.define_section_bonus
        return score, level

    def _rank_candidates(self, concept: str, chunks: list[ChunkHit]) -> list[tuple[ChunkHit, float, str]]:
        ranked = [
            (chunk, *self._candidate_score(concept, chunk))
            for chunk in chunks
            if chunk.section_type != "references"
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:settings.define_candidate_top_k]

    def search(self, concept: str, top_k: int = 5, *, include_conceptual_discussion: bool = False, filters: SearchFilters | None = None) -> list[DefinitionHit]:
        ranked = self._retrieve_ranked(concept, filters=filters)
        if self.llm is not None and settings.llm_enabled and settings.define_llm_enabled:
            try:
                classified = self._classify_definitions(concept, ranked, top_k, include_conceptual_discussion=include_conceptual_discussion)
                if classified:
                    return classified
            except LLMError:
                raise
            except Exception:
                pass
        if self.llm is None or not settings.llm_enabled or not settings.define_llm_enabled:
            return [
                self._to_hit(concept, chunk, classification="unclassified_candidate", candidate_score=score, lexical_match=level)
                for chunk, score, level in ranked[:top_k]
            ]
        if include_conceptual_discussion:
            return [
                self._to_hit(
                    concept,
                    chunk,
                    classification="unclassified_candidate",
                    candidate_score=score,
                    lexical_match=level,
                )
                for chunk, score, level in ranked[:top_k]
            ]
        return []

    def define(self, concept: str, top_k: int = 5, *, filters: SearchFilters | None = None) -> list[DefinitionHit]:
        """Return only source-grounded definitions of the requested concept."""
        ranked = self._retrieve_ranked(concept, filters=filters)
        if not ranked or self.llm is None or not settings.llm_enabled or not settings.define_llm_enabled:
            return []
        expanded = self._expand_ranked(ranked, before=settings.define_context_before, after=settings.define_context_after, max_tokens=settings.define_context_max_tokens)
        validated = self._classify_definition_evidence(concept, expanded)
        return self._consolidate_definitions(concept, validated, top_k)

    def _expand_ranked(self, ranked: list[tuple[ChunkHit, float, str]], *, before: int, after: int, max_tokens: int) -> list[tuple[ChunkHit, float, str]]:
        expanded: list[tuple[ChunkHit, float, str]] = []
        document_chunks: dict[str, list[ChunkHit]] = {}
        related_method = getattr(self.retriever, "related_chunks", None)
        for anchor, score, lexical_match in ranked[:settings.define_candidate_top_k]:
            document_chunks.setdefault(anchor.attachment_key, related_method(anchor) if related_method else [anchor])
            context = ContextExpander(document_chunks[anchor.attachment_key]).expand(
                anchor,
                before=before,
                after=after,
                max_tokens=max_tokens,
                continue_structures=settings.define_context_continue_structures,
            )
            expanded.append((self._with_text(anchor, context.text), score, lexical_match))
        unique: dict[tuple[str | None, str], tuple[ChunkHit, float, str]] = {}
        for item in expanded:
            unique.setdefault((item[0].citation_key, self._normalize_phrase(item[0].text)[:1000]), item)
        return list(unique.values())

    @staticmethod
    def _with_text(anchor: ChunkHit, text: str) -> ChunkHit:
        return ChunkHit(
            chunk_id=anchor.chunk_id, attachment_key=anchor.attachment_key, parent_key=anchor.parent_key,
            citation_key=anchor.citation_key, title=anchor.title, authors=anchor.authors, year=anchor.year,
            doi=anchor.doi, page_start=anchor.page_start, page_end=anchor.page_end,
            section_type=anchor.section_type, section_title=anchor.section_title, text=text,
            score=anchor.score, zotero_link=anchor.zotero_link,
        )

    def _classify_definition_evidence(self, concept: str, ranked: list[tuple[ChunkHit, float, str]]) -> list[tuple[ChunkHit, float, str, object]]:
        from litsearch.llm.prompts import definition_evidence_messages

        results: list[tuple[ChunkHit, float, str, object]] = []
        for start in range(0, len(ranked), 5):
            batch = ranked[start:start + 5]
            evidence = [(f"D{index:02d}", item[0].text) for index, item in enumerate(batch, start=start + 1)]
            response = self.llm.structured_chat(
                definition_evidence_messages(concept, evidence, max_chars=settings.define_llm_evidence_max_chars),
                DefinitionEvidenceBatch,
                task_name="define-stage1",
            )
            by_id = {f"D{index:02d}": item for index, item in enumerate(batch, start=start + 1)}
            for assessment in response.assessments:
                item = by_id.get(assessment.evidence_id)
                if item is None or assessment.concept_match != "requested_concept" or assessment.classification not in {"explicit_definition", "implicit_definition"}:
                    continue
                results.append((*item, assessment))
        return results

    def _consolidate_definitions(self, concept: str, validated: list[tuple[ChunkHit, float, str, object]], top_k: int) -> list[DefinitionHit]:
        unique: dict[str, tuple[ChunkHit, float, str, object]] = {}
        confidence_rank = {"high": 3, "medium": 2, "low": 1}
        for item in validated:
            chunk, score, lexical_match, assessment = item
            text = (assessment.extracted_text or "").strip() or self._extract_definition_text(chunk)
            key = re.sub(r"\s+", " ", self._normalize_phrase(text))
            if not key:
                continue
            previous = unique.get(key)
            if previous is None or (confidence_rank[assessment.confidence], score) > (confidence_rank[previous[3].confidence], previous[1]):
                unique[key] = item
        ordered = sorted(unique.values(), key=lambda item: (confidence_rank[item[3].confidence], item[1], item[2] != "no_match"), reverse=True)
        stage_two = DefinitionResult(
            concept=concept,
            definitions=[
                ValidatedDefinition(
                    definition_id=f"D{index:02d}",
                    extracted_text=(assessment.extracted_text or chunk.text).strip(),
                    evidence_ids=[chunk.chunk_id],
                    classification=assessment.classification,
                    confidence=assessment.confidence,
                )
                for index, (chunk, _, _, assessment) in enumerate(ordered[:top_k], start=1)
            ],
        )
        return [
            self._to_hit(concept, chunk, classification=definition.classification, confidence=definition.confidence, candidate_score=score, lexical_match=lexical_match, text_override=definition.extracted_text)
            for (chunk, score, lexical_match, _), definition in zip(ordered[:top_k], stage_two.definitions)
        ]

    def explain(self, concept: str, top_k: int = 5, *, filters: SearchFilters | None = None) -> ConceptExplanation:
        """Explain a concept by extracting and synthesizing conceptual evidence."""
        ranked = self._retrieve_ranked(concept, filters=filters)[:settings.explain_anchor_top_k]
        if not ranked or self.llm is None or not settings.llm_enabled:
            from litsearch.llm.models import ExplanationOverview
            return ConceptExplanation(concept=concept, overview=ExplanationOverview(statement="", evidence_ids=[]), insufficient_evidence=True)
        expanded: list[tuple[ChunkHit, float, str]] = []
        document_chunks: dict[str, list[ChunkHit]] = {}
        for anchor, score, lexical_match in ranked:
            related_chunks = getattr(self.retriever, "related_chunks", None)
            document_chunks.setdefault(
                anchor.attachment_key,
                related_chunks(anchor) if related_chunks is not None else [anchor],
            )
            context = ContextExpander(document_chunks[anchor.attachment_key]).expand(
                anchor,
                before=settings.explain_context_before,
                after=settings.explain_context_after,
                max_tokens=settings.explain_context_max_tokens,
                continue_structures=settings.explain_context_continue_structures,
            )
            expanded_chunk = ChunkHit(
                chunk_id=anchor.chunk_id,
                attachment_key=anchor.attachment_key,
                parent_key=anchor.parent_key,
                citation_key=anchor.citation_key,
                title=anchor.title,
                authors=anchor.authors,
                year=anchor.year,
                page_start=anchor.page_start,
                page_end=anchor.page_end,
                section_type=anchor.section_type,
                section_title=anchor.section_title,
                text=context.text,
                score=anchor.score,
                zotero_link=anchor.zotero_link,
            )
            expanded.append((expanded_chunk, score, lexical_match))
        unique: dict[tuple[str | None, str], tuple[ChunkHit, float, str]] = {}
        for item in expanded:
            key = (item[0].citation_key, item[0].text[:500])
            unique.setdefault(key, item)
        try:
            points, source_by_evidence = self._extract_conceptual_points(concept, list(unique.values()))
        except LLMError:
            raise
        except Exception:
            from litsearch.llm.models import ExplanationOverview
            return ConceptExplanation(concept=concept, overview=ExplanationOverview(statement="", evidence_ids=[]), insufficient_evidence=True)
        if not points:
            from litsearch.llm.models import ExplanationOverview
            return ConceptExplanation(concept=concept, overview=ExplanationOverview(statement="", evidence_ids=[]), insufficient_evidence=True)
        return self._synthesize_concept(concept, points, source_by_evidence)

    def _extract_conceptual_points(self, concept: str, evidence: list[tuple[ChunkHit, float, str]]) -> tuple[list[ConceptualEvidencePoint], dict[str, ChunkHit]]:
        from litsearch.llm.prompts import explain_extract_messages

        points: list[ConceptualEvidencePoint] = []
        source_by_evidence: dict[str, ChunkHit] = {}
        batch_size = settings.explain_extraction_batch_evidence
        for start in range(0, len(evidence), batch_size):
            batch = evidence[start:start + batch_size]
            packaged = [(f"E{index:02d}", item[0].text) for index, item in enumerate(batch, start=start + 1)]
            source_by_evidence.update({key: item[0] for (key, _), item in zip(packaged, batch)})
            messages = explain_extract_messages(concept, packaged, max_chars=settings.explain_extraction_max_chars)
            try:
                response = self.llm.structured_chat(messages, ConceptualEvidenceExtraction, task_name="explain-extract-v2")
            except TypeError:
                response = self.llm.structured_chat(messages, ConceptualEvidenceExtraction)
            valid_ids = {key for key, _ in packaged}
            for point in response.points[:5]:
                if point.evidence_ids and set(point.evidence_ids) <= valid_ids and point.statement.strip() and point.source_excerpt.strip():
                    points.append(point)
        return self._deduplicate_conceptual_points(points), source_by_evidence

    @staticmethod
    def _deduplicate_conceptual_points(points: list[ConceptualEvidencePoint]) -> list[ConceptualEvidencePoint]:
        merged: dict[str, ConceptualEvidencePoint] = {}
        for point in points:
            key = re.sub(r"\W+", " ", point.statement.casefold()).strip()
            existing = merged.get(key)
            if existing is None:
                merged[key] = point
            else:
                existing.evidence_ids = list(dict.fromkeys(existing.evidence_ids + point.evidence_ids))
                if len(point.source_excerpt) > len(existing.source_excerpt):
                    existing.source_excerpt = point.source_excerpt
        for index, point in enumerate(merged.values(), start=1):
            point.point_id = f"P{index:02d}"
        return list(merged.values())

    def _synthesize_concept(self, concept: str, points: list[ConceptualEvidencePoint], source_by_evidence: dict[str, ChunkHit]) -> ConceptExplanation:
        from litsearch.llm.prompts import explain_synthesize_messages

        messages = explain_synthesize_messages(concept, points)
        try:
            response = self.llm.structured_chat(messages, ConceptExplanation, task_name="explain-synthesize-v2")
        except TypeError:
            response = self.llm.structured_chat(messages, ConceptExplanation)
        valid_ids = {point.point_id for point in points}
        response.points = [point for point in response.points if point.evidence_ids and set(point.evidence_ids) <= valid_ids]
        response.perspectives = [point for point in response.perspectives if point.evidence_ids and set(point.evidence_ids) <= valid_ids]
        point_by_id = {point.point_id: point for point in points}
        references = [response.overview.evidence_ids] + [point.evidence_ids for point in response.points] + [point.evidence_ids for point in response.perspectives]
        aggregate_pages: dict[str, list[int]] = {}
        for reference_ids in references:
            pages_by_citation: dict[str, list[int]] = {}
            citation_keys: list[str] = []
            for point_id in reference_ids:
                point = point_by_id.get(point_id)
                for source_id in point.evidence_ids if point is not None else []:
                    source = source_by_evidence.get(source_id)
                    citation_key = source.citation_key if source else None
                    if not citation_key:
                        continue
                    if citation_key not in pages_by_citation:
                        pages_by_citation[citation_key] = []
                        citation_keys.append(citation_key)
                    aggregate_pages.setdefault(citation_key, [])
                    for page in (source.page_start, source.page_end):
                        if page and page not in pages_by_citation[citation_key]:
                            pages_by_citation[citation_key].append(page)
                        if page and page not in aggregate_pages[citation_key]:
                            aggregate_pages[citation_key].append(page)
            response.citation_keys_by_reference["|".join(reference_ids)] = [
                format_citation_with_pages(citation_key, pages_by_citation[citation_key]) for citation_key in citation_keys
            ]
        response.citation_pages = aggregate_pages
        return response

    def _retrieve_ranked(self, concept: str, *, filters: SearchFilters | None = None) -> list[tuple[ChunkHit, float, str]]:
        if self.retriever is None:
            return []
        chunks: list[ChunkHit] = []
        seen: set[tuple[str | None, int, str]] = set()
        for query in self._candidate_queries(concept):
            for chunk in self.retriever.retrieve(query, filters=filters, top_k=settings.define_candidate_top_k):
                key = (chunk.citation_key, chunk.page_start, chunk.text[:80])
                if key not in seen:
                    seen.add(key)
                    chunks.append(chunk)
        return self._rank_candidates(concept, chunks)

    def _to_hit(self, concept: str, chunk: ChunkHit, *, classification: str | None = None, confidence: str | None = None, candidate_score: float | None = None, lexical_match: str | None = None, text_override: str | None = None) -> DefinitionHit:
        return DefinitionHit(
            concept=concept,
            definition_text=text_override if text_override is not None else self._extract_definition_text(chunk),
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
            lexical_match=lexical_match or self.lexical_match_level(concept, chunk.text),
            candidate_score=chunk.score if candidate_score is None else candidate_score,
        )

    def _classify_definitions(
        self,
        concept: str,
        ranked: list[tuple[ChunkHit, float, str]],
        top_k: int,
        *,
        include_conceptual_discussion: bool = False,
        evidence_max_chars: int = 4000,
    ) -> list[DefinitionHit]:
        from litsearch.llm.prompts import definitions_messages

        results: list[DefinitionHit] = []
        batch = ranked[:10]
        evidence = [(f"P{index:02d}", chunk.text) for index, (chunk, _, _) in enumerate(batch, start=1)]
        by_id = {f"P{index:02d}": item for index, item in enumerate(batch, start=1)}
        messages = definitions_messages(
            concept,
            evidence,
            explanation=include_conceptual_discussion,
            evidence_max_chars=evidence_max_chars,
        )
        for attempt in range(2):
            response = self.llm.structured_chat(messages, DefinitionAssessmentBatch)
            results.clear()
            accepted_classifications = {"explicit_definition", "implicit_definition"}
            if include_conceptual_discussion:
                accepted_classifications.add("conceptual_discussion")
            for assessment in response.assessments:
                item = by_id.get(assessment.evidence_id)
                if item is None or not self._same_concept(concept, assessment.requested_concept) or assessment.concept_match != "requested_concept" or assessment.classification not in accepted_classifications:
                    continue
                chunk, candidate_score, lexical_match = item
                results.append(self._to_hit(concept, chunk, classification=assessment.classification, confidence=assessment.confidence, candidate_score=candidate_score, lexical_match=lexical_match, text_override=chunk.text if include_conceptual_discussion else None))
                if len(results) >= top_k:
                    return results
            if results or attempt == 1:
                return results
            messages = messages + [{
                "role": "user",
                "content": "Correction: your evidence_id values were invalid. Return evidence_id using only the exact labels P01 through P10 shown before the passages. Do not use titles, citation keys, author names, or any other identifiers.",
            }]
        return results
