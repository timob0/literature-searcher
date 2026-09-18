from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from types import SimpleNamespace

from rich.console import Console

from litsearch.shell.app import LiteratureShell
from litsearch.shell.renderer import ShellRenderer
from litsearch.shell.state import ShellServices


@dataclass
class FakeRetriever:
    calls: list[tuple[str, int]] = field(default_factory=list)
    embedding_provider: object | None = object()
    vector_store: object | None = object()

    def retrieve(self, query: str, filters=None, top_k: int = 10):
        self.calls.append((query, top_k))
        return []


@dataclass
class FakeStrategy:
    calls: list[tuple[str, int | None]] = field(default_factory=list)

    def search(self, query: str, top_k: int = 5):
        self.calls.append((query, top_k))
        return []

    def explain(self, query: str, top_k: int = 5):
        self.calls.append((query, top_k))
        return []


@dataclass
class FakeSummaryStrategy:
    calls: list[str] = field(default_factory=list)

    def summarize(self, citation_key: str):
        self.calls.append(citation_key)
        return type("Summary", (), {"paper_title": None, "citation_key": citation_key, "sections": []})()


class FakeRenderer:
    def render_banner(self, **kwargs):
        pass

    def render_hits(self, *args, **kwargs):
        pass

    def render_document_hits(self, *args, **kwargs):
        pass

    def render_summary(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def test_shell_reuses_one_retriever_across_strategies() -> None:
    retriever = FakeRetriever()
    definitions = FakeStrategy()
    examples = FakeStrategy()
    paper_summary = FakeSummaryStrategy()
    services = ShellServices(retriever, definitions, examples, paper_summary)
    shell = LiteratureShell(services=services, renderer=FakeRenderer())

    shell.do_search("employee wellbeing")
    shell.do_define("health literacy")
    shell.do_examples("supportive leadership")
    shell.do_summarize("smithMethodStudy2024")

    assert retriever.calls == [("employee wellbeing", 15)]
    assert definitions.calls == [("health literacy", 5)]
    assert examples.calls == [("supportive leadership", 5)]
    assert paper_summary.calls == ["smithMethodStudy2024"]


def test_shell_exit_requests_cmd2_loop_shutdown() -> None:
    services = ShellServices(FakeRetriever(), FakeStrategy(), FakeStrategy(), FakeSummaryStrategy())
    shell = LiteratureShell(services=services, renderer=FakeRenderer())

    assert shell.do_exit("") is True


def test_shell_dispatches_repeated_define_commands() -> None:
    definitions = FakeStrategy()
    services = ShellServices(FakeRetriever(), definitions, FakeStrategy(), FakeSummaryStrategy())
    shell = LiteratureShell(services=services, renderer=FakeRenderer())

    shell.onecmd("define health oriented leadership")
    shell.onecmd("define health oriented leadership")

    assert definitions.calls == [
        ("health oriented leadership", 5),
        ("health oriented leadership", 5),
    ]


def test_shell_find_uses_raw_semantic_retrieval() -> None:
    retriever = FakeRetriever()
    definitions = FakeStrategy()
    services = ShellServices(retriever, definitions, FakeStrategy(), FakeSummaryStrategy())
    shell = LiteratureShell(services=services, renderer=FakeRenderer())

    shell.do_find("health literacy")

    assert retriever.calls == [("health literacy", 50)]
    assert definitions.calls == []


def test_shell_author_filter_supports_setting_and_inline_forms() -> None:
    retriever = FakeRetriever()
    services = ShellServices(retriever, FakeStrategy(), FakeStrategy(), FakeSummaryStrategy())
    shell = LiteratureShell(services=services, renderer=FakeRenderer())

    shell.onecmd("set author Smith,Jones")
    shell.do_find("culture")
    shell.do_find("culture --author Taylor")

    assert retriever.calls == [("culture", 50), ("culture", 50)]


def test_shell_source_setting_is_session_local() -> None:
    services = ShellServices(FakeRetriever(), FakeStrategy(), FakeStrategy(), FakeSummaryStrategy())
    shell = LiteratureShell(services=services, renderer=FakeRenderer())

    shell.onecmd("set sources schein2010,hofstede2010")

    assert shell.source_filter == "schein2010,hofstede2010"


def test_shell_sources_accepts_space_separated_completion_selections() -> None:
    services = ShellServices(FakeRetriever(), FakeStrategy(), FakeStrategy(), FakeSummaryStrategy())
    shell = LiteratureShell(services=services, renderer=FakeRenderer())

    shell.onecmd("set sources schein2010 hofstede2010 schein2010")

    assert shell.source_filter == "schein2010,hofstede2010"


def test_shell_source_completion_uses_seen_citation_keys() -> None:
    services = ShellServices(FakeRetriever(), FakeStrategy(), FakeStrategy(), FakeSummaryStrategy())
    shell = LiteratureShell(services=services, renderer=FakeRenderer())
    shell.source_candidates = ["scheinOrganizationalCultureLeadership2010", "smithLeadership2024"]

    completions = shell.complete_set("schein", "set sources schein", 13, 19)

    assert completions.to_strings() == ("scheinOrganizationalCultureLeadership2010",)


def test_shell_clear_sources_and_author_unsets_filters() -> None:
    services = ShellServices(FakeRetriever(), FakeStrategy(), FakeStrategy(), FakeSummaryStrategy())
    shell = LiteratureShell(services=services, renderer=FakeRenderer())
    shell.source_filter = "schein2010"
    shell.author_filter = "Smith"

    shell.do_clear("sources")
    shell.do_clear("author")

    assert shell.source_filter is None
    assert shell.author_filter is None


def test_shell_open_requires_sources() -> None:
    services = ShellServices(FakeRetriever(), FakeStrategy(), FakeStrategy(), FakeSummaryStrategy())
    renderer = FakeRenderer()
    renderer.console = type("Console", (), {"print": lambda self, *args, **kwargs: None})()
    shell = LiteratureShell(services=services, renderer=renderer)

    shell.do_open("")

    assert shell.source_filter is None


def test_shell_show_displays_target_and_full_session_setup() -> None:
    from rich.console import Console
    from io import StringIO

    output = StringIO()
    renderer = FakeRenderer()
    renderer.console = Console(file=output, force_terminal=False, width=160)
    services = ShellServices(FakeRetriever(), FakeStrategy(), FakeStrategy(), FakeSummaryStrategy())
    shell = LiteratureShell(services=services, renderer=renderer)
    shell.author_filter = "Smith"
    shell.source_filter = "schein2010"

    shell.do_show("sources")
    shell.do_show("")

    rendered = output.getvalue()
    assert "sources: schein2010" in rendered
    assert "author: Smith" in rendered
    assert "llm-provider:" in rendered
    assert "llm-model:" in rendered


def test_shell_find_groups_documents_and_supports_full_option() -> None:
    from litsearch.retrieval.models import ChunkHit

    retriever = FakeRetriever()
    retriever.retrieve = lambda query, filters=None, top_k=10: [
        ChunkHit("a1", "att-a", "paper-a", "paper-a", title="Paper A", page_start=1, text="strong", score=0.9),
        ChunkHit("a2", "att-a", "paper-a", "paper-a", title="Paper A", page_start=2, text="also relevant", score=0.7),
        ChunkHit("b1", "att-b", "paper-b", "paper-b", title="Paper B", page_start=3, text="best B", score=0.8),
    ]
    services = ShellServices(retriever, FakeStrategy(), FakeStrategy(), FakeSummaryStrategy())
    shell = LiteratureShell(services=services, renderer=FakeRenderer())

    grouped = shell._group_find_hits(retriever.retrieve("query", top_k=50))

    assert [document.citation_key for document in grouped] == ["paper-a", "paper-b"]
    assert grouped[0].chunk_count == 2
    assert grouped[0].representative.text == "strong"


def test_shell_explain_requests_conceptual_discussions() -> None:
    class ExplainStrategy(FakeStrategy):
        def __init__(self):
            super().__init__()
            self.include_conceptual_discussion = False

        def explain(self, query: str, top_k: int = 5):
            self.calls.append((query, top_k))
            self.include_conceptual_discussion = True
            return []

    definitions = ExplainStrategy()
    services = ShellServices(FakeRetriever(), definitions, FakeStrategy(), FakeSummaryStrategy())
    shell = LiteratureShell(services=services, renderer=FakeRenderer())

    shell.do_explain("health literacy")

    assert definitions.calls == [("health literacy", 5)]
    assert definitions.include_conceptual_discussion is True


def test_renderer_prints_highest_ranked_hit_last() -> None:
    output = StringIO()
    renderer = ShellRenderer(Console(file=output, force_terminal=False))
    hits = [
        SimpleNamespace(title="Highest", citation_key="high", text="high text", score=0.9),
        SimpleNamespace(title="Lowest", citation_key="low", text="low text", score=0.2),
    ]

    renderer.render_hits("SEARCH", hits, "text")

    rendered = output.getvalue()
    assert rendered.index("Lowest") < rendered.index("Highest")


def test_renderer_highlights_find_query_in_document_chunks() -> None:
    from litsearch.shell.state import DocumentSearchHit
    from litsearch.retrieval.models import ChunkHit

    output = StringIO()
    renderer = ShellRenderer(Console(file=output, force_terminal=False, width=160))
    chunk = ChunkHit("chunk", "att", "paper", "paper", text="Organizational culture shapes behavior.", score=0.9)
    document = DocumentSearchHit("paper", "Culture", [], None, "", [chunk])

    renderer.render_document_hits("FIND", [document], query="organizational culture")

    rendered = output.getvalue()
    assert "Organizational culture shapes behavior." in rendered
    assert "organizational culture" not in rendered