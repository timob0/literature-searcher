from __future__ import annotations

from litsearch.config import Settings
from litsearch.llm.copilot_cli import GitHubCopilotCLIProvider
from litsearch.llm.errors import LLMConfigurationError
from litsearch.llm.ollama import OllamaLLMProvider
from litsearch.llm.openai_compatible import OpenAICompatibleLLMProvider
from litsearch.llm.openai_provider import OpenAILLMProvider
from litsearch.llm.provider import LLMProvider


def create_llm_provider(config: Settings) -> LLMProvider:
    if config.llm_provider == "ollama":
        return OllamaLLMProvider(
            host=config.ollama_base_url or config.ollama_host,
            model=config.ollama_model,
            timeout=config.ollama_timeout_seconds,
            api_key=config.ollama_api_key,
        )
    if config.llm_provider == "openai":
        return OpenAILLMProvider(api_key=config.openai_api_key, model=config.openai_model)
    if config.llm_provider == "openai_compatible":
        return OpenAICompatibleLLMProvider(base_url=config.openai_compatible_base_url, api_key=config.openai_compatible_api_key, model=config.openai_compatible_model)
    if config.llm_provider == "copilot":
        return GitHubCopilotCLIProvider(executable=config.copilot_cli_executable, model=config.copilot_model)
    raise LLMConfigurationError(f"Unsupported LLM_PROVIDER: {config.llm_provider}")