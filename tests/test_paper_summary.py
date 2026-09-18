from __future__ import annotations

import re
from dataclasses import dataclass
from io import StringIO

from rich.console import Console
from typer.testing import CliRunner

from litsearch.cli import app
from litsearch.retrieval.models import ChunkHit
from litsearch.retrieval.paper_summary import PaperSummaryResult, PaperSummaryStrategy
from litsearch.retrieval.summary_boilerplate import classify_boilerplate, is_boilerplate_for_facet
from litsearch.shell.renderer import ShellRenderer


TOC_TEXT = "Part One: Foundations\nChapter 1 The Concept of Organizational Culture\nChapter 2 Levels of Culture\nChapter 3 Culture and Leadership\nChapter 4 Assessing Culture"
DISCLAIMER_TEXT = "This publication is designed to provide accurate information. It is sold with the understanding that the publisher is not engaged in rendering professional advice. The advice and strategies contained herein may not be suitable for your situation. You should consult a competent professional where appropriate."
MARKETING_TEXT = "Praise for Organizational Culture and Leadership: \"Regarded as one of the most influential management books of all time.\" -- Harvard Business Review \"A landmark achievement.\" -- Warren Bennis, USC"
BIOGRAPHY_TEXT = "About the Author. Edgar H. Schein is a professor emeritus at the MIT Sloan School of Management. He has written numerous books and articles on organizational culture and has consulted widely."
METHODOLOGY_TEXT = "I have always written from this scholar/practitioner perspective. Clinical Research argues that practical experiences where we help organizations solve problems provide opportunities to observe and inquire. I rely more on careful observation, group interviews, and focused inquiry with informants."
LIMITATION_TEXT = "These dimensions reflect my own cultural understanding and should be viewed only as a first approximation."


@dataclass
class FakeRetriever:
    hits: list[ChunkHit]

    def retrieve(self, query: str, filters=None, top_k: int = 10):
        return list(self.hits)

    def related_chunks(self, anchor: ChunkHit) -> list[ChunkHit]:
        return [chunk for chunk in self.hits if chunk.attachment_key == anchor.attachment_key] or [anchor]


def _chunk(text: str, *, page: int = 1, citation_key: str = "scheinOrganizationalCultureLeadership2010") -> ChunkHit:
    return ChunkHit(
        chunk_id=f"att-1:{page}:hash", attachment_key="att-1", parent_key="parent-1",
        citation_key=citation_key, title="Organizational Culture and Leadership, 4th Edition",
        page_start=page, page_end=page, text=text, score=0.9,
    )


class ScriptedSummaryLLM:
    model = "fake"

    def structured_chat(self, messages, response_model, **kwargs):
        content = messages[-1]["content"]
        if response_model.__name__ == "SummaryFacetExtraction":
            return self._extract(content, response_model)
        return self._synthesize(content, response_model)

    @staticmethod
    def _extract(content, response_model):
        evidence_ids = re.findall(r"\[(E\d+)\]", content)
        if "Facet: TOPICS" in content:
            return response_model.model_validate({"facet": "topic", "points": [
                {"point_id": "tmp", "facet": "topic", "statement": "The book examines levels of organizational culture and leadership's role in shaping it.", "evidence_ids": evidence_ids, "confidence": "high", "explicitly_stated": True},
            ]})
        if "Facet: PROBLEM / PURPOSE" in content:
            return response_model.model_validate({"facet": "problem_or_purpose", "points": [
                {"point_id": "tmp", "facet": "problem_or_purpose", "statement": "The book aims to clarify culture and its relationship with leadership.", "evidence_ids": evidence_ids, "confidence": "high", "explicitly_stated": True},
            ]})
        if "Facet: APPROACH / METHODOLOGY" in content:
            if "clinical" not in content.casefold() and "observation" not in content.casefold():
                return response_model.model_validate({"facet": "methodology_or_approach", "points": []})
            return response_model.model_validate({"facet": "methodology_or_approach", "points": [
                {"point_id": "tmp", "facet": "methodology_or_approach", "statement": "The author draws on a clinical, observational scholar-practitioner approach based on consulting experience, observation, group interviews and focused inquiry.", "evidence_ids": evidence_ids, "confidence": "high", "explicitly_stated": True},
            ]})
        if "Facet: FINDINGS / CONTRIBUTIONS" in content:
            return response_model.model_validate({"facet": "finding_or_contribution", "points": [
                {"point_id": "tmp", "facet": "finding_or_contribution", "statement": "Culture operates at multiple levels that leaders can diagnose and shape.", "evidence_ids": evidence_ids, "confidence": "high", "explicitly_stated": True},
            ]})
        if "Facet: LIMITATIONS" in content:
            if "first approximation" in content:
                return response_model.model_validate({"facet": "limitation", "points": [
                    {"point_id": "tmp", "facet": "limitation", "statement": "The proposed dimensions reflect the author's own cultural understanding and are only a first approximation.", "evidence_ids": evidence_ids, "confidence": "high", "explicitly_stated": True},
                ]})
            return response_model.model_validate({"facet": "limitation", "points": []})
        return response_model.model_validate({"facet": "future_research", "points": []})

    @staticmethod
    def _synthesize(content, response_model):
        point_ids = list(dict.fromkeys(re.findall(r"\[(P\d+)\]", content)))

        def has(label: str) -> bool:
            return f"{label}:\n" in content

        def section(statement: str, present: bool) -> dict:
            return {
                "points": [{"statement": statement, "evidence_ids": point_ids}] if present else [],
                "insufficient_evidence": not present,
            }

        methodology_classification = "clinical_observational" if "clinical" in content.casefold() else "not_stated"
        return response_model.model_validate({
            "overview": {"prose": "Schein presents organizational culture as a system of shared assumptions shaped through leadership.", "points": [{"statement": "overview", "evidence_ids": point_ids}] if point_ids else [], "insufficient_evidence": not point_ids},
            "topics": section("Levels and dimensions of organizational culture; the relationship between culture and leadership.", has("TOPICS")),
            "problem_or_purpose": {"prose": "The book aims to clarify culture and its relationship with leadership." if has("PROBLEM / PURPOSE") else None, "points": [], "insufficient_evidence": not has("PROBLEM / PURPOSE")},
            "methodology_or_approach": {"prose": "The author uses a clinical and observational scholar-practitioner approach." if has("APPROACH / METHODOLOGY") else None, "points": [], "insufficient_evidence": not has("APPROACH / METHODOLOGY")},
            "methodology_classification": methodology_classification,
            "findings_or_contributions": section("Culture operates at multiple levels that leaders can diagnose and shape.", has("FINDINGS / CONTRIBUTIONS")),
            "limitations": section("The proposed dimensions are a first approximation grounded in the author's own understanding.", has("LIMITATIONS") and "first approximation" in content),
            "future_research": section("", False),
        })


def test_boilerplate_classifier_rejects_disclaimer_and_marketing_and_biography() -> None:
    assert classify_boilerplate(DISCLAIMER_TEXT) == "disclaimer"
    assert classify_boilerplate(MARKETING_TEXT) == "marketing"
    assert classify_boilerplate(BIOGRAPHY_TEXT) == "biography"
    assert classify_boilerplate(LIMITATION_TEXT) is None
    assert classify_boilerplate(METHODOLOGY_TEXT) is None


def test_toc_allowed_for_topics_but_rejected_for_other_facets() -> None:
    assert classify_boilerplate(TOC_TEXT) == "toc"
    assert is_boilerplate_for_facet(TOC_TEXT, "topic") is False
    assert is_boilerplate_for_facet(TOC_TEXT, "finding_or_contribution") is True


def test_summarize_rejects_disclaimer_marketing_and_biography_before_llm() -> None:
    chunks = [
        _chunk(DISCLAIMER_TEXT, page=1),
        _chunk(MARKETING_TEXT, page=20),
        _chunk(BIOGRAPHY_TEXT, page=260),
        _chunk(METHODOLOGY_TEXT, page=100),
        _chunk(LIMITATION_TEXT, page=180),
    ]
    strategy = PaperSummaryStrategy(retriever=FakeRetriever(chunks), llm=ScriptedSummaryLLM())
    result = strategy.summarize("scheinOrganizationalCultureLeadership2010")

    rejected_categories = {facet for facet, _ in result.rejected_boilerplate}
    assert rejected_categories  # something was rejected before ever reaching the LLM
    assert not any(point.facet == "limitation" and "may not be suitable" in point.statement for point in result.stage1_points)
    assert all(DISCLAIMER_TEXT[:40] not in (point.source_excerpt or "") for point in result.stage1_points)


def test_summarize_extracts_clinical_observational_methodology_without_biography() -> None:
    chunks = [_chunk(METHODOLOGY_TEXT, page=100), _chunk(BIOGRAPHY_TEXT, page=260)]
    strategy = PaperSummaryStrategy(retriever=FakeRetriever(chunks), llm=ScriptedSummaryLLM())
    result = strategy.summarize("scheinOrganizationalCultureLeadership2010")

    assert result.summary.methodology_classification == "clinical_observational"
    assert "About the Author" not in (result.summary.methodology_or_approach.prose or "")
    assert "clinical" in (result.summary.methodology_or_approach.prose or "").casefold()


def test_summarize_keeps_explicit_limitation() -> None:
    chunks = [_chunk(LIMITATION_TEXT, page=21)]
    strategy = PaperSummaryStrategy(retriever=FakeRetriever(chunks), llm=ScriptedSummaryLLM())
    result = strategy.summarize("scheinOrganizationalCultureLeadership2010")

    assert result.summary.limitations.points
    assert not result.summary.limitations.insufficient_evidence


def test_summarize_reports_absent_future_research_without_hallucination() -> None:
    chunks = [_chunk(METHODOLOGY_TEXT, page=100)]
    strategy = PaperSummaryStrategy(retriever=FakeRetriever(chunks), llm=ScriptedSummaryLLM())
    result = strategy.summarize("scheinOrganizationalCultureLeadership2010")

    assert result.summary.future_research.insufficient_evidence
    assert not result.summary.future_research.points


def test_toc_dump_never_rendered_but_topics_are_concise() -> None:
    chunks = [_chunk(TOC_TEXT, page=1), _chunk(METHODOLOGY_TEXT, page=100)]
    strategy = PaperSummaryStrategy(retriever=FakeRetriever(chunks), llm=ScriptedSummaryLLM())
    result = strategy.summarize("scheinOrganizationalCultureLeadership2010")

    output = StringIO()
    renderer = ShellRenderer(Console(file=output, force_terminal=False, width=160))
    renderer.render_summary(result)
    rendered = output.getvalue()

    assert "Chapter 1" not in rendered
    assert "Chapter 2" not in rendered
    assert "organizational culture" in rendered.casefold()
    assert "levels and dimensions" in rendered.casefold()


def test_render_summary_does_not_print_raw_evidence_chunks() -> None:
    chunks = [_chunk(METHODOLOGY_TEXT, page=100), _chunk(LIMITATION_TEXT, page=180)]
    strategy = PaperSummaryStrategy(retriever=FakeRetriever(chunks), llm=ScriptedSummaryLLM())
    result = strategy.summarize("scheinOrganizationalCultureLeadership2010")

    output = StringIO()
    renderer = ShellRenderer(Console(file=output, force_terminal=False, width=160))
    renderer.render_summary(result)
    rendered = output.getvalue()

    assert METHODOLOGY_TEXT not in rendered
    assert LIMITATION_TEXT not in rendered
    assert len(rendered.split()) < 400


def test_summarize_without_llm_returns_insufficient_evidence() -> None:
    chunks = [_chunk(METHODOLOGY_TEXT, page=100)]
    strategy = PaperSummaryStrategy(retriever=FakeRetriever(chunks), llm=None)
    result = strategy.summarize("scheinOrganizationalCultureLeadership2010")

    assert result.insufficient_evidence
    assert isinstance(result, PaperSummaryResult)


def test_cli_summarize_paper_command_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["summarize-paper", "smithMethodStudy2024"])
    assert result.exit_code == 0
    assert "smithMethodStudy2024".lower() in result.output.lower()

