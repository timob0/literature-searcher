from __future__ import annotations

import json
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from litsearch.config import settings
from litsearch.llm.domain import LLMHealthStatus, LLMMessage
from litsearch.llm.errors import LLMAuthenticationError, LLMModelNotFoundError, LLMStructuredOutputError, LLMTimeoutError, LLMUnavailableError
from litsearch.llm.json_utils import strip_json_fences

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class OpenAICompatibleLLMProvider:
    def __init__(self, *, base_url: str | None = None, api_key: str | None = None, model: str | None = None) -> None:
        self.endpoint = base_url or settings.openai_compatible_base_url
        self.model = model or settings.openai_compatible_model
        self.client = OpenAI(api_key=api_key or settings.openai_compatible_api_key or "unused", base_url=self.endpoint, timeout=settings.openai_compatible_timeout_seconds)

    @property
    def provider_name(self) -> str:
        return "openai_compatible"

    @property
    def model_name(self) -> str:
        return self.model

    def health_check(self) -> LLMHealthStatus:
        if not self.model:
            return LLMHealthStatus(available=False, provider=self.provider_name, endpoint=self.endpoint, error="OPENAI_COMPATIBLE_MODEL is not configured")
        try:
            self.client.models.list()
            return LLMHealthStatus(available=True, provider=self.provider_name, model=self.model, endpoint=self.endpoint)
        except Exception as exc:
            return LLMHealthStatus(available=False, provider=self.provider_name, model=self.model, endpoint=self.endpoint, error=str(exc))

    def is_available(self) -> bool:
        return self.health_check().available

    def structured_chat(self, messages: list[LLMMessage | dict[str, str]], response_model: type[ResponseModel], *, task_name: str | None = None) -> ResponseModel:
        serialized = [message.model_dump() if isinstance(message, LLMMessage) else message for message in messages]
        schema = response_model.model_json_schema()
        correction = None
        native = settings.openai_compatible_native_structured_output
        for attempt in range(2):
            try:
                prompt_messages = serialized + ([{"role": "user", "content": correction}] if correction else [])
                if not native:
                    prompt_messages = prompt_messages + [{"role": "user", "content": f"Return JSON only matching this schema: {json.dumps(schema)}"}]
                kwargs = {
                    "model": self.model,
                    "messages": prompt_messages,
                    "temperature": settings.ollama_temperature,
                    "max_tokens": settings.openai_max_output_tokens,
                }
                if native:
                    kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": response_model.__name__, "schema": schema}}
                try:
                    response = self.client.chat.completions.create(**kwargs)
                except Exception:
                    if not native:
                        raise
                    native = False
                    prompt_messages = serialized + [{"role": "user", "content": f"Return JSON only matching this schema: {json.dumps(schema)}"}]
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=prompt_messages,
                        temperature=settings.ollama_temperature,
                        max_tokens=settings.openai_max_output_tokens,
                    )
                content = response.choices[0].message.content or ""
                return response_model.model_validate_json(strip_json_fences(content))
            except ValidationError as exc:
                if attempt == 1:
                    raise LLMStructuredOutputError(str(exc)) from exc
                correction = f"Return corrected JSON only. Validation error: {exc}. Schema: {json.dumps(schema)}"
            except TimeoutError as exc:
                raise LLMTimeoutError(str(exc)) from exc
            except Exception as exc:
                message = str(exc).casefold()
                if "auth" in message or "api key" in message:
                    raise LLMAuthenticationError(str(exc)) from exc
                if "not found" in message:
                    raise LLMModelNotFoundError(str(exc)) from exc
                raise LLMUnavailableError(str(exc)) from exc
        raise LLMStructuredOutputError("Compatible provider failed to produce structured output")