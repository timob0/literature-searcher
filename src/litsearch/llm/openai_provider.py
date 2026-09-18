from __future__ import annotations

import time
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from litsearch.config import settings
from litsearch.llm.domain import LLMHealthStatus, LLMMessage
from litsearch.llm.errors import LLMAuthenticationError, LLMModelNotFoundError, LLMStructuredOutputError, LLMTimeoutError, LLMUnavailableError

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class OpenAILLMProvider:
    def __init__(self, *, api_key: str | None = None, model: str | None = None) -> None:
        self.model = model or settings.openai_model
        self.client = OpenAI(api_key=api_key or settings.openai_api_key, timeout=settings.openai_timeout_seconds)

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self.model

    def health_check(self) -> LLMHealthStatus:
        if not settings.openai_api_key:
            return LLMHealthStatus(available=False, provider=self.provider_name, model=self.model, error="OPENAI_API_KEY is not configured")
        try:
            self.client.models.retrieve(self.model)
            return LLMHealthStatus(available=True, provider=self.provider_name, model=self.model)
        except Exception as exc:
            return LLMHealthStatus(available=False, provider=self.provider_name, model=self.model, error=str(exc))

    def is_available(self) -> bool:
        return self.health_check().available

    def structured_chat(self, messages: list[LLMMessage | dict[str, str]], response_model: type[ResponseModel], *, task_name: str | None = None) -> ResponseModel:
        serialized = [message.model_dump() if isinstance(message, LLMMessage) else message for message in messages]
        try:
            response = self.client.responses.parse(
                model=self.model,
                input=serialized,
                text_format=response_model,
                reasoning={"effort": settings.openai_reasoning_effort},
                max_output_tokens=settings.openai_max_output_tokens,
            )
            parsed = getattr(response, "output_parsed", None)
            if isinstance(parsed, response_model):
                return parsed
            raise LLMStructuredOutputError("OpenAI returned no validated structured output")
        except LLMStructuredOutputError:
            raise
        except TimeoutError as exc:
            raise LLMTimeoutError(f"OpenAI request timed out: {exc}") from exc
        except Exception as exc:
            message = str(exc).casefold()
            if "auth" in message or "api key" in message:
                raise LLMAuthenticationError(str(exc)) from exc
            if "not found" in message:
                raise LLMModelNotFoundError(str(exc)) from exc
            raise LLMUnavailableError(f"OpenAI request failed: {exc}") from exc