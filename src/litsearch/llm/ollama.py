from __future__ import annotations

from typing import TypeVar

from ollama import Client
from pydantic import BaseModel, ValidationError

from litsearch.config import settings
from litsearch.llm.domain import LLMHealthStatus, LLMMessage
from litsearch.llm.errors import LLMModelNotFoundError, LLMStructuredOutputError, LLMTimeoutError, LLMUnavailableError
from litsearch.llm.json_utils import strip_json_fences


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class OllamaLLMProvider:
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        api_key: str | None = None,
    ) -> None:
        self.host = host or settings.ollama_host
        self.model = model or settings.ollama_model
        self.timeout = timeout or settings.ollama_timeout_seconds
        self.api_key = api_key if api_key is not None else settings.ollama_api_key
        headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else None
        self.client = Client(host=self.host, timeout=self.timeout, headers=headers)

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self.model

    def health_check(self) -> LLMHealthStatus:
        try:
            self.client.list()
            return LLMHealthStatus(available=True, provider=self.provider_name, model=self.model, endpoint=self.host)
        except Exception as exc:
            return LLMHealthStatus(available=False, provider=self.provider_name, model=self.model, endpoint=self.host, error=str(exc))

    def is_available(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception:
            return False

    def structured_chat(
        self,
        messages: list[LLMMessage | dict[str, str]],
        response_model: type[ResponseModel],
        *,
        task_name: str | None = None,
    ) -> ResponseModel:
        request_messages = list(messages)
        schema = response_model.model_json_schema()
        for attempt in range(2):
            try:
                serialized = [message.model_dump() if isinstance(message, LLMMessage) else message for message in request_messages]
                response = self.client.chat(
                    model=self.model,
                    messages=serialized,
                    format=schema,
                    options={
                        "temperature": settings.ollama_temperature,
                        "num_ctx": settings.ollama_context_length,
                        "num_predict": settings.ollama_max_output_tokens,
                    },
                    think=settings.ollama_thinking,
                )
                content = strip_json_fences(response["message"]["content"])
                return response_model.model_validate_json(content)
            except ValidationError as exc:
                if attempt == 1:
                    raise LLMStructuredOutputError(f"Ollama returned invalid structured output: {exc}") from exc
                request_messages.append({
                    "role": "user",
                    "content": f"Your previous response did not match the required JSON schema. Return corrected JSON only, with no markdown, using exactly these field names. Validation error: {exc}. Schema: {schema}",
                })
            except TimeoutError as exc:
                raise LLMTimeoutError(f"Ollama request timed out: {exc}") from exc
            except KeyError as exc:
                raise LLMStructuredOutputError(f"Ollama response was missing {exc}") from exc
            except Exception as exc:
                message = str(exc).casefold()
                if "not found" in message or "model" in message and "not" in message:
                    raise LLMModelNotFoundError(str(exc)) from exc
                raise LLMUnavailableError(f"Ollama request failed: {exc}") from exc
        raise RuntimeError("Ollama structured request failed")