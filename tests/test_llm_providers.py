from __future__ import annotations

import json
from types import SimpleNamespace

from litsearch.config import Settings
from litsearch.llm.copilot_cli import GitHubCopilotCLIProvider
from litsearch.llm.domain import LLMMessage
from litsearch.llm.factory import create_llm_provider
from litsearch.llm.models import HealthResponse
from litsearch.llm.ollama import OllamaLLMProvider


def test_factory_selects_all_supported_providers() -> None:
    for name in ("ollama", "openai", "copilot"):
        provider = create_llm_provider(Settings(llm_provider=name))
        assert provider.provider_name == name
    provider = create_llm_provider(Settings(llm_provider="openai_compatible", openai_compatible_model="local"))
    assert provider.provider_name == "openai_compatible"
    assert provider.model_name == "local"


def test_ollama_provider_uses_configured_remote_endpoint() -> None:
    provider = OllamaLLMProvider(host="http://192.168.1.50:11434", model="remote-model")

    assert provider.host == "http://192.168.1.50:11434"
    assert provider.model_name == "remote-model"


def test_ollama_structured_output_is_validated() -> None:
    provider = OllamaLLMProvider(host="http://test", model="test")
    provider.client = SimpleNamespace(chat=lambda **kwargs: {"message": {"content": json.dumps({"status": "ok", "message": "ready"})}})

    result = provider.structured_chat([LLMMessage(role="user", content="status")], HealthResponse)

    assert result.status == "ok"


def test_copilot_command_is_restricted_and_omits_empty_model(monkeypatch) -> None:
    provider = GitHubCopilotCLIProvider(executable="copilot", model="")
    monkeypatch.setattr("shutil.which", lambda executable: "/usr/local/bin/copilot")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout='{"status":"ok","message":"ready"}', stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    provider.structured_chat([{"role": "user", "content": "status"}], HealthResponse)

    command = captured["command"]
    assert "-p" in command
    assert "-s" in command
    assert "--no-ask-user" in command
    assert captured["kwargs"]["stdin"] is not None
    assert captured["kwargs"]["env"]["CI"] == "1"
    assert "--model" not in command
    assert not any(flag in command for flag in ("--allow-all", "--allow-all-tools", "--yolo"))