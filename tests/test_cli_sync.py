from __future__ import annotations

import pytest
from typer.testing import CliRunner

from litsearch.cli import app
from litsearch.zotero.client import ZoteroClient


@pytest.fixture(autouse=True)
def _force_zotero_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    # These CLI commands talk to whatever Zotero/Chroma backend is reachable on the
    # machine running the tests, with `rebuild` wiping the real vector store and
    # manifest. Force the "backend unavailable" short-circuit so the test only
    # exercises CLI plumbing and never touches a real index.
    monkeypatch.setattr(ZoteroClient, "is_available", lambda self: False)


def test_cli_sync_reports_backend_status() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["sync"])
    assert result.exit_code == 0
    assert "Sync" in result.output or "backend" in result.output.lower()


def test_cli_rebuild_reports_backend_status() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["rebuild"])
    assert result.exit_code == 0
    assert "Rebuild" in result.output or "backend" in result.output.lower()
