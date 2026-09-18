from __future__ import annotations

from collections.abc import Iterable

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from litsearch.text_format import format_citation_with_pages


class ShellRenderer:
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def render_banner(
        self,
        embedding_available: bool,
        chroma_available: bool,
        ollama_available: bool = False,
        llm_model: str | None = None,
        llm_provider: str = "ollama",
        llm_endpoint: str | None = None,
    ) -> None:
        embedding = "connected" if embedding_available else "unavailable"
        chroma = "connected" if chroma_available else "unavailable"
        provider_status = "connected" if ollama_available else "unavailable"
        model = llm_model or "disabled"
        endpoint = f"\nEndpoint:   {llm_endpoint}" if llm_endpoint else ""
        self.console.print(
            Panel(
                f"Embedding: {embedding}\nChroma:    {chroma}\nLLM:       {llm_provider} ({provider_status})\nModel:     {model}{endpoint}\n\nType 'help' for commands.",
                title="Academic Literature Search",
                border_style="blue",
            )
        )

    def render_hits(self, heading: str, hits: Iterable[object], text_field: str) -> None:
        hits = list(reversed(list(hits)))
        self.console.print(f"[bold]{heading}[/bold]")
        if not hits:
            self.console.print("No results were found.")
            return
        for index, hit in enumerate(hits, start=1):
            title = getattr(hit, "paper_title", None) or getattr(hit, "title", None) or "Unknown paper"
            citation = getattr(hit, "citation_key", None) or "No citation key"
            page = getattr(hit, "page", None) or getattr(hit, "page_start", None)
            score = getattr(hit, "score", None)
            self.console.print(f"\n[cyan][{index}][/cyan] {title}")
            self.console.print(f"    Citation: {citation}")
            if page:
                self.console.print(f"    Page: {page}")
            if score is not None:
                self.console.print(f"    Score: {score:.2f}")
            classification = getattr(hit, "classification", None)
            if classification:
                self.console.print(f"    Classification: {classification}")
            self.console.print(f'\n    "{getattr(hit, text_field)}"')

    def render_definitions(self, concept: str, hits: Iterable[object]) -> None:
        hits = list(reversed(list(hits)))
        self.console.print(f"[bold]DEFINE results for: {concept}[/bold]")
        self.console.print(f"Definitions found: {len(hits)}")
        if not hits:
            self.console.print(f'No validated definition found for "{concept}".')
            return
        for index, hit in enumerate(hits, start=1):
            self.console.print(f"\n[cyan][{index}][/cyan] {getattr(hit, 'paper_title', None) or 'Unknown paper'}")
            self.console.print(f"    Citation: {getattr(hit, 'citation_key', None) or 'No citation key'}")
            if getattr(hit, "page", None):
                self.console.print(f"    Page: {hit.page}")
            self.console.print(f"    Classification: {hit.classification}")
            self.console.print(f"    Confidence: {hit.llm_confidence}")
            self.console.print(f"    Score: {hit.score:.2f}")
            self.console.print(f'\n    "{hit.definition_text}"')

    def render_examples(self, concept: str, hits: Iterable[object]) -> None:
        hits = list(reversed(list(hits)))
        self.console.print(f"[bold]EXAMPLES: {concept}[/bold]")
        self.console.print(f"Examples found: {len(hits)}")
        for index, hit in enumerate(hits, start=1):
            citation = getattr(hit, "citation_key", None) or "citation key unavailable"
            page = getattr(hit, "page", None)
            location = f"{citation}, p. {page}" if page else citation
            self.console.print(f"\n{index}. {hit.example_text}")
            self.console.print(escape(f"   [{location}]"))
            self.console.print(f"   Confidence: {getattr(hit, 'llm_confidence', None) or 'unknown'}")

    def render_document_hits(self, heading: str, hits: Iterable[object], *, full: bool = False, query: str | None = None) -> None:
        hits = list(reversed(list(hits)))
        self.console.print(f"[bold]{heading}[/bold]")
        if not hits:
            self.console.print("No matching documents were found.")
            return
        for index, document in enumerate(hits, start=1):
            representative = document.representative
            title = document.title or "Unknown paper"
            citation = document.citation_key or "No citation key"
            self.console.print(f"\n[cyan][{index}][/cyan] {title}")
            self.console.print(f"    Citation: {citation}")
            self.console.print(f"    Matching chunks: {document.chunk_count}")
            self.console.print(f"    Best score: {document.score:.2f}")
            chunks = document.chunks if full else [representative]
            for chunk in chunks:
                page = f" (p. {chunk.page_start})" if chunk.page_start else ""
                text = Text(chunk.text, style="dark_green")
                if query:
                    text.highlight_words([query], style="bold yellow", case_sensitive=False)
                self.console.print("\n    \"", end="")
                self.console.print(text, end="")
                self.console.print(f'"{page}')

    def render_summary(self, result: object) -> None:
        summary = getattr(result, "summary", None) or result
        title = getattr(result, "title", None) or getattr(summary, "title", None) or "Unknown document"
        citation_key = getattr(result, "citation_key", None) or getattr(summary, "citation_key", None) or "unknown"
        self.console.print(f"[bold]{escape(title)}[/bold]")
        self.console.print(citation_key)
        if getattr(result, "insufficient_evidence", False) and not self._summary_has_content(summary):
            self.console.print("\nNo summary evidence was found for this document.")
            return
        self._render_summary_section("OVERVIEW", summary.overview, summary, prose_only=True)
        self._render_summary_section("TOPICS", summary.topics, summary)
        self._render_summary_section("PURPOSE", summary.problem_or_purpose, summary)
        self._render_summary_section("APPROACH / METHODOLOGY", summary.methodology_or_approach, summary)
        self._render_summary_section("KEY FINDINGS / CONTRIBUTIONS", summary.findings_or_contributions, summary)
        self._render_summary_section("LIMITATIONS", summary.limitations, summary)
        self._render_summary_section("FUTURE RESEARCH / DIRECTIONS", summary.future_research, summary)

    @staticmethod
    def _summary_has_content(summary: object) -> bool:
        names = ("overview", "topics", "problem_or_purpose", "methodology_or_approach", "findings_or_contributions", "limitations", "future_research")
        return any(getattr(getattr(summary, name), "prose", None) or getattr(getattr(summary, name), "points", None) for name in names)

    def _render_summary_section(self, heading: str, section: object, summary: object, *, prose_only: bool = False) -> None:
        self.console.print(f"\n[bold cyan]{heading}[/bold cyan]")
        prose = getattr(section, "prose", None)
        points = list(getattr(section, "points", None) or [])
        if getattr(section, "insufficient_evidence", False) or (not prose and not points):
            self.console.print("Not explicitly stated in the retrieved evidence.")
            return
        if prose:
            self.console.print(prose)
            merged_ids = list(dict.fromkeys(evidence_id for point in points for evidence_id in point.evidence_ids))
            if merged_ids:
                self.console.print(escape(self._summary_provenance(summary, merged_ids)))
            return
        if prose_only:
            return
        for point in points:
            provenance = self._summary_provenance(summary, point.evidence_ids)
            self.console.print(escape(f"\u2022 {point.statement} {provenance}"))

    @staticmethod
    def _summary_provenance(summary: object, evidence_ids: list[str]) -> str:
        citation_keys_by_reference = getattr(summary, "citation_keys_by_reference", None) or {}
        citation_keys = citation_keys_by_reference.get("|".join(evidence_ids), [])
        internal = ", ".join(evidence_ids) or "evidence unavailable"
        citations = "; ".join(citation_keys) or "citation key unavailable"
        return f"[{internal} | {citations}]"

    def render_explanation(self, explanation: object) -> None:
        concept = getattr(explanation, "concept", "")
        self.console.print(f"[bold]EXPLAIN: {concept}[/bold]")
        if getattr(explanation, "insufficient_evidence", False):
            self.console.print("No sufficient conceptual evidence was found.")
            return
        overview = explanation.overview
        overview_provenance = self._explanation_provenance(explanation, overview.evidence_ids)
        self.console.print(f"\n[bold cyan]Overview[/bold cyan]\n{overview.statement}")
        self.console.print(overview_provenance)
        self.console.print("\n[bold cyan]Key aspects[/bold cyan]")
        for index, point in enumerate(explanation.points, start=1):
            heading = f"{point.heading}: " if point.heading else ""
            provenance = self._explanation_provenance(explanation, point.evidence_ids)
            self.console.print(f"\n{index}. {heading}{point.statement}")
            self.console.print(f"   {provenance}")
        if explanation.perspectives:
            self.console.print("\n[bold cyan]Different perspectives[/bold cyan]")
            for point in explanation.perspectives:
                provenance = self._explanation_provenance(explanation, point.evidence_ids)
                self.console.print(f"\n- {point.statement}")
                self.console.print(f"  {provenance}")
        source_keys = []
        citation_pages = getattr(explanation, "citation_pages", None)
        if citation_pages:
            for key, pages in citation_pages.items():
                label = format_citation_with_pages(key, pages)
                if label not in source_keys:
                    source_keys.append(label)
        else:
            for keys in explanation.citation_keys_by_reference.values():
                for key in keys:
                    if key not in source_keys:
                        source_keys.append(key)
        if source_keys:
            self.console.print("\n[bold cyan]Sources[/bold cyan]")
            for key in source_keys:
                self.console.print(key)

    @staticmethod
    def _explanation_provenance(explanation: object, evidence_ids: list[str]) -> str:
        citation_keys = explanation.citation_keys_by_reference.get("|".join(evidence_ids), [])
        internal = ", ".join(evidence_ids) or "evidence unavailable"
        citations = "; ".join(citation_keys) or "citation key unavailable"
        return f"[{internal} | {citations}]"

    def error(self, message: str) -> None:
        self.console.print(f"[red]{message}[/red]")