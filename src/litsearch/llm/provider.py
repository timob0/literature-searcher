from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

from litsearch.llm.domain import LLMHealthStatus, LLMMessage


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class LLMProvider(Protocol):
    @property
    def provider_name(self) -> str:
        ...

    @property
    def model_name(self) -> str:
        ...

    def health_check(self) -> LLMHealthStatus:
        ...

    def structured_chat(
        self,
        messages: list[LLMMessage | dict[str, str]],
        response_model: type[ResponseModel],
        *,
        task_name: str | None = None,
    ) -> ResponseModel:
        ...

    def is_available(self) -> bool:
        ...