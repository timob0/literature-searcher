from __future__ import annotations

from dataclasses import dataclass

from typer.testing import CliRunner

from litsearch.cli import app
from litsearch.retrieval.definitions import DefinitionHit, DefinitionSearchStrategy
from litsearch.retrieval.models import ChunkHit
from litsearch.retrieval.retriever import SimpleRetriever
from litsearch.llm.models import ExplanationOverview
from litsearch.shell.renderer import ShellRenderer
from rich.console import Console
from io import StringIO


@dataclass
class FakeRetriever:
    hits: list[ChunkHit]

    def retrieve(self, query: str, filters=None, top_k: int = 10):
        return self.hits


def test_definition_strategy_prefers_explanatory_sentence() -> None:
    strategy = DefinitionSearchStrategy(retriever=FakeRetriever([
        ChunkHit(
            chunk_id="chunk-1",
            attachment_key="att-1",
            parent_key="parent-1",
            citation_key="smithHealthLiteracy2024",
            title="Health Literacy Research",
            authors=["Smith"],
            year=2024,
            page_start=12,
            page_end=12,
            section_type="introduction",
            text="Health literacy is the ability to access, understand, and use health information effectively. This concept matters for patient decision-making.",
            score=0.8,
            zotero_link="zotero://open-pdf/library/items/att-1?page=12",
        ),
        ChunkHit(
            chunk_id="chunk-2",
            attachment_key="att-2",
            parent_key="parent-2",
            citation_key="doeHealthPolicy2022",
            title="Health Policy Review",
            authors=["Doe"],
            year=2022,
            page_start=8,
            page_end=8,
            section_type="discussion",
            text="The policy review discusses community engagement and local interventions.",
            score=0.6,
            zotero_link="zotero://open-pdf/library/items/att-2?page=8",
        ),
    ]))

    result = strategy.search("health literacy")
    assert result
    assert "Health literacy is" in result[0].definition_text
    assert result[0].citation_key == "smithHealthLiteracy2024"


def test_cli_define_command_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["define", "health literacy"])
    assert result.exit_code == 0
    assert "health literacy".lower() in result.output.lower()


def test_definition_llm_maps_classification_to_source_passage() -> None:
    retriever = FakeRetriever([
        ChunkHit(
            chunk_id="chunk-1", attachment_key="att-1", parent_key="parent-1",
            citation_key="smithHealthLiteracy2024", title="Health Literacy Research", page_start=12,
            text="Health literacy is the ability to access, understand, and use health information effectively.", score=0.8,
        ),
    ])

    class FakeLLM:
        model = "fake"

        def structured_chat(self, messages, response_model):
            return response_model.model_validate({"assessments": [{
                "evidence_id": "P01", "requested_concept": "health literacy", "classification": "explicit_definition",
                "extracted_definition": "invented paraphrase", "confidence": "high",
                "concept_match": "requested_concept",
            }]})

    results = DefinitionSearchStrategy(retriever=retriever, llm=FakeLLM()).search("health literacy", top_k=1)

    assert len(results) == 1
    assert results[0].definition_text.startswith("Health literacy is the ability")
    assert results[0].definition_text != "invented paraphrase"
    assert results[0].classification == "explicit_definition"


def test_definition_llm_uses_one_bounded_batch() -> None:
    chunks = [ChunkHit(
        chunk_id=f"chunk-{index}", attachment_key=f"att-{index}", parent_key=f"parent-{index}",
        citation_key=f"paper-{index}", text=f"Concept passage {index}.", score=1.0 - index / 100,
    ) for index in range(12)]

    class FakeRetriever:
        def retrieve(self, query: str, filters=None, top_k: int = 10):
            return chunks

    class FakeLLM:
        model = "fake"
        calls = 0

        def structured_chat(self, messages, response_model):
            self.calls += 1
            assert sum(len(message["content"]) for message in messages) < 25000
            return response_model.model_validate({"assessments": []})

    llm = FakeLLM()
    DefinitionSearchStrategy(retriever=FakeRetriever(), llm=llm).search("health literacy", top_k=5)

    assert llm.calls <= 2


def test_definition_returns_no_results_when_llm_returns_no_definitions() -> None:
    retriever = FakeRetriever([ChunkHit(
        chunk_id="chunk-1", attachment_key="att-1", parent_key="parent-1",
        citation_key="paper-1", text="Health literacy is the ability to use health information.", score=0.9,
    )])

    class EmptyLLM:
        model = "fake"

        def structured_chat(self, messages, response_model):
            return response_model.model_validate({"assessments": []})

    results = DefinitionSearchStrategy(retriever=retriever, llm=EmptyLLM()).search("health literacy", top_k=1)

    assert results == []


def test_definition_lexical_matching_distinguishes_hyphenation() -> None:
    assert DefinitionSearchStrategy.lexical_match_level("health-oriented leadership", "Health-Oriented Leadership is a construct.") == "exact_phrase"
    assert DefinitionSearchStrategy.lexical_match_level("health-oriented leadership", "Health oriented leadership is a construct.") == "normalized_phrase"
    assert DefinitionSearchStrategy.lexical_match_level("health-oriented leadership", "Leadership practices support employee health.") == "partial_term_match"
    assert DefinitionSearchStrategy.lexical_match_level("health-oriented leadership", "A separate concept is described here.") == "no_match"


def test_definition_matching_accepts_case_whitespace_and_line_break_variants() -> None:
    assert DefinitionSearchStrategy._same_concept("health literacy", "Health Literacy")
    assert DefinitionSearchStrategy._same_concept("health-oriented leadership", "Health\n oriented  leadership")
    assert DefinitionSearchStrategy._same_concept("organisational culture", "organizational culture")


def test_definition_retrieves_british_and_american_query_variants() -> None:
    queries = DefinitionSearchStrategy()._candidate_queries("organisational culture")

    assert "organisational culture" in queries
    assert "organizational culture" in queries
    assert "definition of organizational culture" in queries


def test_definition_accepts_normalized_llm_concept_spelling() -> None:
    retriever = FakeRetriever([ChunkHit(
        "chunk-1", "att-1", "paper-1", "paper-1", text="Health Literacy (HL) - understood as the ability to deal with health information.", score=0.9,
    )])

    class NormalizingLLM:
        model = "fake"

        def structured_chat(self, messages, response_model):
            return response_model.model_validate({"assessments": [{
                "evidence_id": "P01", "requested_concept": "Health Literacy",
                "classification": "implicit_definition", "confidence": "high",
                "concept_match": "requested_concept",
            }]})

    results = DefinitionSearchStrategy(retriever=retriever, llm=NormalizingLLM()).search("health literacy", top_k=1)

    assert len(results) == 1
    assert results[0].classification == "implicit_definition"


def test_explain_includes_conceptual_discussion_but_define_does_not() -> None:
    retriever = FakeRetriever([ChunkHit(
        "chunk-1", "att-1", "paper-1", "paper-1", text="Health literacy is discussed as a multidimensional public-health capability.", score=0.9,
    )])

    class ConceptualLLM:
        model = "fake"

        def structured_chat(self, messages, response_model):
            return response_model.model_validate({"assessments": [{
                "evidence_id": "P01", "requested_concept": "health literacy",
                "classification": "conceptual_discussion", "confidence": "medium",
                "concept_match": "requested_concept",
            }]})

    strategy = DefinitionSearchStrategy(retriever=retriever, llm=ConceptualLLM())

    assert strategy.search("health literacy") == []
    explanation = strategy.search("health literacy", include_conceptual_discussion=True)
    assert len(explanation) == 1
    assert explanation[0].classification == "conceptual_discussion"


def test_definition_rejects_non_request_local_evidence_ids() -> None:
    retriever = FakeRetriever([ChunkHit(
        "chunk-1", "att-1", "paper-1", "paper-1", text="Health literacy is discussed as a public-health capability.", score=0.9,
    )])

    class InvalidIdLLM:
        model = "fake"

        def structured_chat(self, messages, response_model):
            return response_model.model_validate({"assessments": [{
                "evidence_id": "paper-1", "requested_concept": "health literacy",
                "classification": "conceptual_discussion", "confidence": "medium",
                "concept_match": "requested_concept",
            }]})

    assert DefinitionSearchStrategy(retriever=retriever, llm=InvalidIdLLM()).search("health literacy") == []


def test_explain_falls_back_to_ranked_candidates_when_ids_remain_invalid() -> None:
    retriever = FakeRetriever([ChunkHit(
        "chunk-1", "att-1", "paper-1", "paper-1", text="Health literacy is discussed as a public-health capability.", score=0.9,
    )])

    class InvalidIdLLM:
        model = "fake"

        def structured_chat(self, messages, response_model):
            return response_model.model_validate({"assessments": [{
                "evidence_id": "wrong-id", "requested_concept": "health literacy",
                "classification": "conceptual_discussion", "confidence": "medium",
                "concept_match": "requested_concept",
            }]})

    results = DefinitionSearchStrategy(retriever=retriever, llm=InvalidIdLLM()).search(
        "health literacy", include_conceptual_discussion=True,
    )

    assert len(results) == 1
    assert results[0].classification == "unclassified_candidate"


def test_explain_falls_back_when_structured_output_is_malformed() -> None:
    retriever = FakeRetriever([ChunkHit(
        "chunk-1", "att-1", "paper-1", "paper-1", text="A long conceptual discussion.", score=0.9,
    )])

    class BrokenLLM:
        model = "fake"

        def structured_chat(self, messages, response_model):
            raise ValueError("invalid structured output")

    results = DefinitionSearchStrategy(retriever=retriever, llm=BrokenLLM()).explain("organisational culture")

    assert results.insufficient_evidence is True


def test_definition_suppresses_reference_sections_and_ranks_lexical_matches() -> None:
    chunks = [
        ChunkHit("ref", "a", "a", "ref", text="Health-oriented leadership (Smith, 2020). doi:10.1/a", score=0.99, section_type="references"),
        ChunkHit("related", "b", "b", "related", text="Health literacy is the ability to use health information.", score=0.95),
        ChunkHit("target", "c", "c", "target", text="Health-oriented leadership is a focus on leaders' responsibility for health.", score=0.80),
    ]
    strategy = DefinitionSearchStrategy(retriever=FakeRetriever(chunks))

    ranked = strategy._rank_candidates("health-oriented leadership", chunks)

    assert all(item[0].chunk_id != "ref" for item in ranked)
    assert ranked[0][0].chunk_id == "target"


def test_explain_extracts_conceptual_points_and_synthesizes_without_raw_chunks() -> None:
    chunk = ChunkHit(
        "culture:1:hash", "culture", "paper-1", "culture2024",
        text="A long organizational anecdote about a meeting. Culture guides and constrains behaviour through shared norms.", score=0.8,
    )

    class ConceptLLM:
        model = "fake"
        calls = []

        def structured_chat(self, messages, response_model, **kwargs):
            self.calls.append((response_model.__name__, messages))
            if response_model.__name__ == "ConceptualEvidenceExtraction":
                return response_model.model_validate({"concept": "organisational culture", "points": [{
                    "point_id": "raw", "statement": "Culture guides and constrains behaviour through shared norms.",
                    "type": "characteristic", "evidence_ids": ["E01"],
                    "source_excerpt": "Culture guides and constrains behaviour through shared norms.",
                    "relevance": "central", "confidence": "high",
                }]})
            return response_model.model_validate({
                "concept": "organisational culture",
                "overview": {"statement": "Organisational culture consists of shared norms that guide behaviour.", "evidence_ids": ["P01"]},
                "points": [{"heading": "Shared norms", "statement": "Culture guides behaviour through shared norms.", "evidence_ids": ["P01"]}],
                "perspectives": [],
            })

    llm = ConceptLLM()
    result = DefinitionSearchStrategy(retriever=FakeRetriever([chunk]), llm=llm).explain("organisational culture")

    assert result.overview.statement.startswith("Organisational culture")
    assert len(result.points) == 1
    assert result.points[0].evidence_ids == ["P01"]
    assert "long organizational anecdote" not in llm.calls[-1][1][1]["content"]


def test_explain_merges_duplicate_conceptual_points_and_preserves_list() -> None:
    chunks = [
        ChunkHit("culture:1:hash", "culture", "paper-1", "culture2024", text="Culture can be analysed at three levels: artifacts, espoused values, and basic underlying assumptions.", score=0.9),
        ChunkHit("culture:2:hash", "culture", "paper-1", "culture2024", text="Culture consists of shared assumptions that guide behaviour.", score=0.8),
    ]

    class ConceptLLM:
        model = "fake"

        def structured_chat(self, messages, response_model, **kwargs):
            if response_model.__name__ == "ConceptualEvidenceExtraction":
                return response_model.model_validate({"concept": "organisational culture", "points": [
                    {"point_id": "a", "statement": "Culture is based on shared assumptions.", "type": "characteristic", "evidence_ids": ["E01"], "source_excerpt": "shared assumptions", "relevance": "central", "confidence": "high"},
                    {"point_id": "b", "statement": "Culture is based on shared assumptions.", "type": "characteristic", "evidence_ids": ["E02"], "source_excerpt": "shared assumptions", "relevance": "supporting", "confidence": "high"},
                    {"point_id": "c", "statement": "Culture has levels: artifacts, espoused values, and basic underlying assumptions.", "type": "level", "evidence_ids": ["E01"], "source_excerpt": "artifacts, espoused values, and basic underlying assumptions", "relevance": "central", "confidence": "high"},
                ]})
            return response_model.model_validate({"concept": "organisational culture", "overview": {"statement": "Culture is shared and layered.", "evidence_ids": ["P01", "P02"]}, "points": [{"statement": "Culture is based on shared assumptions.", "evidence_ids": ["P01", "P02"]}, {"statement": "Culture has three levels.", "evidence_ids": ["P02"]}], "perspectives": []})

    result = DefinitionSearchStrategy(retriever=FakeRetriever(chunks), llm=ConceptLLM()).explain("organisational culture")

    assert len(result.points) == 2
    assert result.points[0].evidence_ids == ["P01", "P02"]


def test_explain_renderer_preserves_internal_ids_with_citation_keys() -> None:
    output = StringIO()
    renderer = ShellRenderer(Console(file=output, force_terminal=False, width=200))
    explanation = type("Explanation", (), {
        "concept": "culture",
        "insufficient_evidence": False,
        "overview": ExplanationOverview(statement="Culture is shared.", evidence_ids=["P01", "P05"]),
        "points": [type("Point", (), {"heading": "Shared assumptions", "statement": "Culture has shared assumptions.", "evidence_ids": ["P01"]})()],
        "perspectives": [],
        "citation_keys_by_reference": {
            "P01|P05": ["scheinOrganizationalCultureLeadership2010", "hofstedeCulturesOrganizationsSoftware2010"],
            "P01": ["scheinOrganizationalCultureLeadership2010"],
        },
    })()

    renderer.render_explanation(explanation)

    rendered = output.getvalue()
    assert "[P01, P05 | scheinOrganizationalCultureLeadership2010; hofstedeCulturesOrganizationsSoftware2010]" in rendered
    assert "[P01 | scheinOrganizationalCultureLeadership2010]" in rendered
    assert "Sources" in rendered


def test_definition_returns_no_llm_validated_results_for_different_concept() -> None:
    retriever = FakeRetriever([ChunkHit(
        "health-literacy", "a", "a", "healthLiteracy", text="Health literacy is the ability to use health information.", score=0.9,
    )])

    class DifferentConceptLLM:
        model = "fake"

        def structured_chat(self, messages, response_model):
            return response_model.model_validate({"assessments": [{
                "evidence_id": "P01", "requested_concept": "health-oriented leadership",
                "classification": "explicit_definition", "confidence": "high",
                "concept_match": "different_concept",
            }]})

    results = DefinitionSearchStrategy(retriever=retriever, llm=DifferentConceptLLM()).search("health-oriented leadership")

    assert results == []


def test_define_stage_one_rejects_related_concept_and_keeps_definition() -> None:
    chunks = [
        ChunkHit("target", "att-1", "paper-1", "target2024", text="Health-oriented leadership is defined as leadership behaviour that takes employee health into account.", score=0.8),
        ChunkHit("related", "att-2", "paper-2", "literacy2020", text="Health literacy refers to the ability to access and understand health information.", score=0.95),
    ]

    class FakeLLM:
        model = "fake"

        def structured_chat(self, messages, response_model, **kwargs):
            return response_model.model_validate({"assessments": [
                {"evidence_id": "D01", "classification": "explicit_definition", "concept_match": "requested_concept", "extracted_text": chunks[0].text, "confidence": "high"},
                {"evidence_id": "D02", "classification": "explicit_definition", "concept_match": "different_concept", "extracted_text": chunks[1].text, "confidence": "high"},
            ]})

    results = DefinitionSearchStrategy(retriever=FakeRetriever(chunks), llm=FakeLLM()).define("health-oriented leadership")

    assert len(results) == 1
    assert results[0].citation_key == "target2024"
    assert results[0].classification == "explicit_definition"


def test_define_excludes_conceptual_discussion_and_deduplicates_overlap() -> None:
    chunks = [
        ChunkHit("one", "att-1", "paper-1", "paper2024", text="Supportive leadership is characterised by listening to employees and considering their needs.", score=0.9),
        ChunkHit("two", "att-1", "paper-1", "paper2024", text="Supportive leadership is characterised by listening to employees and considering their needs.", score=0.8),
    ]

    class FakeLLM:
        model = "fake"

        def structured_chat(self, messages, response_model, **kwargs):
            return response_model.model_validate({"assessments": [
                {"evidence_id": "D01", "classification": "conceptual_characterisation", "concept_match": "requested_concept", "extracted_text": chunks[0].text, "confidence": "high"},
                {"evidence_id": "D02", "classification": "conceptual_characterisation", "concept_match": "requested_concept", "extracted_text": chunks[1].text, "confidence": "high"},
            ]})

    results = DefinitionSearchStrategy(retriever=FakeRetriever(chunks), llm=FakeLLM()).define("supportive leadership")

    assert results == []


def test_define_returns_zero_when_only_related_literature_is_retrieved() -> None:
    chunk = ChunkHit("one", "att-1", "paper-1", "literacy2020", text="Health literacy refers to the ability to access and understand health information.", score=0.95)

    class FakeLLM:
        model = "fake"

        def structured_chat(self, messages, response_model, **kwargs):
            return response_model.model_validate({"assessments": [{
                "evidence_id": "D01", "classification": "explicit_definition", "concept_match": "different_concept", "extracted_text": chunk.text, "confidence": "high",
            }]})

    assert DefinitionSearchStrategy(retriever=FakeRetriever([chunk]), llm=FakeLLM()).define("health-oriented leadership") == []


def test_retriever_removes_embedded_null_bytes_from_chroma_results() -> None:
    result = {
        "documents": [["Definition with\x00 embedded byte"]],
        "metadatas": [[{"chunk_id": "att:0:hash", "attachment_key": "att", "parent_key": "paper", "citation_key": "paper2024", "title": "Title\x00"}]],
        "distances": [[0.1]],
    }

    hits = SimpleRetriever(embedding_provider=object(), vector_store=object())._hits_from_result(result)

    assert hits[0].text == "Definition with embedded byte"
    assert hits[0].title == "Title"
