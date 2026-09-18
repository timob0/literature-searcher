from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from litsearch.config import settings
from litsearch.llm.domain import LLMHealthStatus, LLMMessage
from litsearch.llm.errors import LLMAuthenticationError, LLMConfigurationError, LLMStructuredOutputError, LLMTimeoutError, LLMUnavailableError
from litsearch.llm.json_utils import strip_json_fences

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class GitHubCopilotCLIProvider:
    def __init__(self, *, executable: str | None = None, model: str | None = None) -> None:
        self.executable = executable or settings.copilot_cli_executable
        self.model = model if model is not None else settings.copilot_model

    @property
    def provider_name(self) -> str:
        return "copilot"

    @property
    def model_name(self) -> str:
        return self.model or "configured default"

    def health_check(self) -> LLMHealthStatus:
        path = shutil.which(self.executable)
        return LLMHealthStatus(
            available=path is not None,
            provider=self.provider_name,
            model=self.model_name,
            endpoint=path,
            error=None if path else f"Executable not found: {self.executable}",
        )

    def is_available(self) -> bool:
        return self.health_check().available

    @staticmethod
    def _serialize(messages: list[LLMMessage | dict[str, str]], schema: dict) -> str:
        body = "\n\n".join(f"{message['role'].upper()}: {message['content']}" for message in [item.model_dump() if isinstance(item, LLMMessage) else item for item in messages])
        return f"{body}\n\nReturn ONLY valid JSON matching this schema. Do not include Markdown or commentary.\nJSON schema:\n{json.dumps(schema)}"

    @staticmethod
    def _clean_json(output: str) -> str:
        return strip_json_fences(output)

    def structured_chat(self, messages: list[LLMMessage | dict[str, str]], response_model: type[ResponseModel], *, task_name: str | None = None) -> ResponseModel:
        path = shutil.which(self.executable)
        if path is None:
            raise LLMUnavailableError(f"Copilot CLI executable not found: {self.executable}")
        schema = response_model.model_json_schema()
        prompt = self._serialize(messages, schema)
        validation_error = ""
        for attempt in range(2):
            if attempt:
                prompt += f"\n\nYour previous response did not conform to the required JSON schema. Validation error: {validation_error}\nReturn ONLY corrected JSON."
            command = [path, "-p", prompt, "-s", "--no-ask-user"]
            if self.model:
                command.extend(["--model", self.model])
            try:
                environment = os.environ.copy()
                environment["CI"] = "1"
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    timeout=settings.copilot_timeout_seconds,
                    check=False,
                    env=environment,
                )
            except subprocess.TimeoutExpired as exc:
                raise LLMTimeoutError("Copilot CLI request timed out") from exc
            if completed.returncode != 0:
                error = (completed.stderr or completed.stdout).strip()
                if "auth" in error.casefold() or "login" in error.casefold():
                    raise LLMAuthenticationError(error)
                raise LLMUnavailableError(error or f"Copilot CLI exited with code {completed.returncode}")
            output = completed.stdout.strip()
            lowered_output = output.casefold()
            if "cannot find github copilot cli" in lowered_output or "install github copilot cli" in lowered_output:
                raise LLMUnavailableError(
                    "GitHub Copilot CLI is not installed. Install it separately and authenticate it before using the copilot provider."
                )
            try:
                return response_model.model_validate_json(self._clean_json(output))
            except (ValidationError, ValueError) as exc:
                validation_error = str(exc)
                if attempt == 1:
                    raise LLMStructuredOutputError(validation_error) from exc
        raise LLMStructuredOutputError("Copilot CLI failed to produce structured output")