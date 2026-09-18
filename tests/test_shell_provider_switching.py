from __future__ import annotations

from litsearch.config import settings
from litsearch.shell.app import LiteratureShell
from litsearch.shell.state import ShellServices
from tests.test_shell import FakeRenderer, FakeRetriever, FakeStrategy, FakeSummaryStrategy


def test_shell_provider_switch_reuses_retriever(monkeypatch) -> None:
    services = ShellServices(FakeRetriever(), FakeStrategy(), FakeStrategy(), FakeSummaryStrategy())
    shell = LiteratureShell(services=services, renderer=FakeRenderer())
    original_retriever = shell.services.retriever
    monkeypatch.setattr(settings, "llm_provider", "ollama")

    shell.onecmd("set llm-provider copilot")

    assert shell.services.retriever is original_retriever
    assert shell.services.llm.provider_name == "copilot"