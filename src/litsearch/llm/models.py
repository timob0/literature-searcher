from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StructuredModel(BaseModel):
    """Base for LLM-facing schemas; rejects unexpected fields so a model's field-name
    mismatches surface as a validation error instead of silently defaulting to empty."""

    model_config = ConfigDict(extra="forbid")


class HealthResponse(StructuredModel):
    status: Literal["ok"]
    message: str = Field(default="")


class ExampleAssessment(StructuredModel):
    evidence_id: str
    classification: Literal["concrete_example", "possible_example", "not_example"]
    example_text: str | None = None
    confidence: Literal["high", "medium", "low"]
    reason: str = ""


class ExampleAssessmentBatch(StructuredModel):
    assessments: list[ExampleAssessment] = Field(default_factory=list)


class ExtractedExample(StructuredModel):
    example_id: str
    statement: str
    example_type: Literal["behavior", "practice", "policy", "event", "decision", "interaction", "intervention", "organizational_case", "other"]
    concept_match: Literal["requested_concept", "related_concept", "uncertain"]
    source_excerpt: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    reason: str | None = None


class ExampleExtractionResult(StructuredModel):
    concept: str
    examples: list[ExtractedExample] = Field(default_factory=list)


class DefinitionAssessment(StructuredModel):
    evidence_id: str
    requested_concept: str
    classification: Literal["explicit_definition", "implicit_definition", "conceptual_discussion", "not_definition"]
    extracted_definition: str | None = None
    confidence: Literal["high", "medium", "low"]
    concept_match: Literal["requested_concept", "different_concept", "unclear"]


class DefinitionAssessmentBatch(StructuredModel):
    assessments: list[DefinitionAssessment] = Field(default_factory=list)


class DefinitionEvidence(StructuredModel):
    evidence_id: str
    classification: Literal[
        "explicit_definition",
        "implicit_definition",
        "conceptual_characterisation",
        "not_definition",
    ]
    concept_match: Literal["requested_concept", "different_concept", "uncertain"]
    extracted_text: str | None = None
    confidence: Literal["high", "medium", "low"]
    reason: str | None = None


class DefinitionEvidenceBatch(StructuredModel):
    assessments: list[DefinitionEvidence] = Field(default_factory=list)


class ValidatedDefinition(StructuredModel):
    definition_id: str
    extracted_text: str
    evidence_ids: list[str] = Field(default_factory=list)
    classification: Literal["explicit_definition", "implicit_definition"]
    confidence: Literal["high", "medium", "low"]
    distinct_perspective: str | None = None


class DefinitionResult(StructuredModel):
    concept: str
    definitions: list[ValidatedDefinition] = Field(default_factory=list)


class ConceptualEvidencePoint(StructuredModel):
    point_id: str
    statement: str
    type: Literal["definition", "characteristic", "component", "dimension", "level", "mechanism", "boundary", "perspective", "model"]
    evidence_ids: list[str] = Field(default_factory=list)
    source_excerpt: str
    relevance: Literal["central", "supporting"]
    confidence: Literal["high", "medium", "low"]


class ConceptualEvidenceExtraction(StructuredModel):
    concept: str
    points: list[ConceptualEvidencePoint] = Field(default_factory=list)


class ConceptualExplanationPoint(StructuredModel):
    heading: str | None = None
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)


class ExplanationOverview(StructuredModel):
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)


class ConceptExplanation(StructuredModel):
    concept: str
    overview: ExplanationOverview
    points: list[ConceptualExplanationPoint] = Field(default_factory=list)
    perspectives: list[ConceptualExplanationPoint] = Field(default_factory=list)
    insufficient_evidence: bool = False
    citation_keys_by_reference: dict[str, list[str]] = Field(default_factory=dict)
    citation_pages: dict[str, list[int]] = Field(default_factory=dict)


class SummaryEvidencePoint(StructuredModel):
    point_id: str
    facet: Literal[
        "topic", "problem_or_purpose", "methodology_or_approach",
        "finding_or_contribution", "limitation", "future_research",
    ]
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)
    source_excerpt: str | None = None
    confidence: Literal["high", "medium", "low"]
    explicitly_stated: bool = True


class SummaryFacetExtraction(StructuredModel):
    facet: str
    points: list[SummaryEvidencePoint] = Field(default_factory=list)


class SummaryPoint(StructuredModel):
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)


class SummarySection(StructuredModel):
    prose: str | None = None
    points: list[SummaryPoint] = Field(default_factory=list)
    insufficient_evidence: bool = False


class DocumentSummary(StructuredModel):
    citation_key: str = ""
    title: str = ""
    document_type: str = ""
    methodology_classification: Literal[
        "quantitative_empirical", "qualitative_empirical", "mixed_methods",
        "conceptual_theoretical", "clinical_observational", "review",
        "case_based", "other", "not_stated",
    ] = "not_stated"
    overview: SummarySection = Field(default_factory=SummarySection)
    topics: SummarySection = Field(default_factory=SummarySection)
    problem_or_purpose: SummarySection = Field(default_factory=SummarySection)
    methodology_or_approach: SummarySection = Field(default_factory=SummarySection)
    findings_or_contributions: SummarySection = Field(default_factory=SummarySection)
    limitations: SummarySection = Field(default_factory=SummarySection)
    future_research: SummarySection = Field(default_factory=SummarySection)
    insufficient_evidence: bool = False
    citation_keys_by_reference: dict[str, list[str]] = Field(default_factory=dict)
    citation_pages: dict[str, list[int]] = Field(default_factory=dict)