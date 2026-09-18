from __future__ import annotations

from dataclasses import dataclass

from typer.testing import CliRunner

from litsearch.cli import app
from litsearch.retrieval.examples import ExampleHit, ExampleSearchStrategy
from litsearch.retrieval.models import ChunkHit
from litsearch.shell.renderer import ShellRenderer
from rich.console import Console
from io import StringIO


@dataclass
class FakeRetriever:
    hits: list[ChunkHit]

    def retrieve(self, query: str, filters=None, top_k: int = 10):
        return self.hits


def test_example_strategy_prefers_actionable_examples() -> None:
    strategy = ExampleSearchStrategy(retriever=FakeRetriever([
        ChunkHit(
            chunk_id="chunk-1",
            attachment_key="att-1",
            parent_key="parent-1",
            citation_key="mullerSupportiveLeadership2024",
            title="Supportive Leadership in Practice",
            authors=["Muller"],
            year=2024,
            page_start=18,
            page_end=18,
            section_type="results",
            text="Managers regularly held one-to-one meetings with employees, adjusted workloads, and involved staff in decision-making. This represented supportive leadership practice.",
            score=0.86,
            zotero_link="zotero://open-pdf/library/items/att-1?page=18",
        ),
        ChunkHit(
            chunk_id="chunk-2",
            attachment_key="att-2",
            parent_key="parent-2",
            citation_key="smithLeadershipWellbeing2021",
            title="Leadership and Wellbeing",
            authors=["Smith"],
            year=2021,
            page_start=6,
            page_end=6,
            section_type="discussion",
            text="Supportive leadership was positively associated with employee wellbeing and organizational outcomes.",
            score=0.68,
            zotero_link="zotero://open-pdf/library/items/att-2?page=6",
        ),
    ]))

    result = strategy.search("supportive leadership practices")
    assert result
    assert "Managers regularly" in result[0].example_text
    assert result[0].citation_key == "mullerSupportiveLeadership2024"


def test_cli_examples_command_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["examples", "supportive leadership practices"])
    assert result.exit_code == 0
    assert "supportive leadership practices".lower() in result.output.lower()


class FakeLLM:
    model = "fake"

    def __init__(self, response):
        self.response = response
        self.messages = []

    def is_available(self):
        return True

    def structured_chat(self, messages, response_model):
        self.messages.append(messages)
        return response_model.model_validate(self.response)


def test_examples_maps_valid_llm_evidence_ids_to_source_hits() -> None:
    retriever = FakeRetriever([
        ChunkHit(
            chunk_id="chunk-1", attachment_key="att-1", parent_key="parent-1",
            citation_key="smithLeadership2024", title="Leadership Study", page_start=8,
            text="Managers redistributed responsibilities when staff reported overload.", score=0.9,
        ),
        ChunkHit(
            chunk_id="chunk-2", attachment_key="att-2", parent_key="parent-2",
            citation_key="jonesLeadership2021", title="Leadership Review", page_start=12,
            text="Supportive leadership is associated with employee wellbeing.", score=0.8,
        ),
    ])
    llm = FakeLLM({"concept": "supportive leadership", "examples": [
        {"example_id": "E01", "statement": "Managers redistributed responsibilities.", "example_type": "practice", "concept_match": "requested_concept", "source_excerpt": "Managers redistributed responsibilities", "evidence_ids": ["E01"], "confidence": "high"},
    ]})

    results = ExampleSearchStrategy(retriever=retriever, llm=llm).search("supportive leadership", top_k=5)

    assert len(results) == 1
    assert results[0].citation_key == "smithLeadership2024"
    assert results[0].classification == "concrete_example"
    assert results[0].llm_confidence == "high"
    assert results[0].page == 8


def test_examples_falls_back_when_llm_fails() -> None:
    retriever = FakeRetriever([ChunkHit(
        chunk_id="chunk-1", attachment_key="att-1", parent_key="parent-1",
        citation_key="smithLeadership2024", title="Leadership Study", page_start=8,
        text="Managers redistributed responsibilities when staff reported overload.", score=0.9,
    )])

    class FailingLLM(FakeLLM):
        def structured_chat(self, messages, response_model):
            raise RuntimeError("Ollama unavailable")

    results = ExampleSearchStrategy(retriever=retriever, llm=FailingLLM({})).search("supportive leadership", top_k=1)

    assert len(results) == 1
    assert results[0].classification == "unclassified_candidate"


def test_examples_extracts_multiple_concrete_examples_from_one_block() -> None:
    chunk = ChunkHit(
        "culture:1:hash", "culture", "paper-1", "schein2010",
        title="Culture Study", text="An organization espoused teamwork but rewarded competition. It claimed safety while cost pressure encouraged unsafe practice.", score=0.9,
    )

    class ExtractingLLM:
        model = "fake"

        def structured_chat(self, messages, response_model, **kwargs):
            return response_model.model_validate({"concept": "espoused values", "examples": [
                {"example_id": "a", "statement": "The organization espoused teamwork but rewarded competition.", "example_type": "practice", "concept_match": "requested_concept", "source_excerpt": "espoused teamwork but rewarded competition", "evidence_ids": ["E01"], "confidence": "high"},
                {"example_id": "b", "statement": "The organization espoused safety while cost pressure encouraged unsafe practice.", "example_type": "practice", "concept_match": "requested_concept", "source_excerpt": "claimed safety while cost pressure encouraged unsafe practice", "evidence_ids": ["E01"], "confidence": "high"},
            ]})

    results = ExampleSearchStrategy(retriever=FakeRetriever([chunk]), llm=ExtractingLLM()).search("espoused values", top_k=5)

    assert len(results) == 2
    assert all(len(result.example_text) < 120 for result in results)
    assert all(result.evidence_ids == ["E01"] for result in results)


def test_examples_renderer_shows_citation_key_and_page() -> None:
    output = StringIO()
    renderer = ShellRenderer(Console(file=output, force_terminal=False, width=160))
    hit = ExampleHit(
        concept="espoused values", example_text="Teamwork was espoused while competition was rewarded.",
        paper_title="Culture Study", citation_key="schein2010", page=12, classification="concrete_example",
        llm_confidence="high", evidence_ids=["E03"],
    )

    renderer.render_examples("espoused values", [hit])

    assert "[schein2010, p. 12]" in output.getvalue()


def test_examples_caps_merged_candidate_pool() -> None:
    chunks = [ChunkHit(
        f"chunk-{index}", f"att-{index}", f"paper-{index}", f"paper-{index}",
        text=f"Concrete practice {index}.", score=1.0 - index / 100,
    ) for index in range(40)]

    class CountingLLM:
        model = "fake"
        calls = 0

        def structured_chat(self, messages, response_model, **kwargs):
            self.calls += 1
            return response_model.model_validate({"concept": "concept", "examples": []})

    llm = CountingLLM()
    ExampleSearchStrategy(retriever=FakeRetriever(chunks), llm=llm).search("concept", top_k=5)

    assert llm.calls <= 10
