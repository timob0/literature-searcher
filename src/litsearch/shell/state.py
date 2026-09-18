from __future__ import annotations

from dataclasses import dataclass

from litsearch.config import settings
from litsearch.llm.factory import create_llm_provider
from litsearch.llm.provider import LLMProvider
from litsearch.retrieval.definitions import DefinitionSearchStrategy
from litsearch.retrieval.examples import ExampleSearchStrategy
from litsearch.retrieval.paper_summary import PaperSummaryStrategy
from litsearch.retrieval.retriever import SimpleRetriever
from litsearch.zotero.client import ZoteroClient


@dataclass(slots=True)
class DocumentSearchHit:
    citation_key: str | None
    title: str | None
    authors: list[str]
    year: int | None
    zotero_link: str
    chunks: list[object]

    @property
    def representative(self):
        return max(self.chunks, key=lambda chunk: chunk.score)

    @property
    def score(self) -> float:
        return self.representative.score

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


@dataclass(slots=True)
class ShellServices:
    retriever: SimpleRetriever
    definitions: DefinitionSearchStrategy
    examples: ExampleSearchStrategy
    paper_summary: PaperSummaryStrategy
    llm: LLMProvider | None = None

    @classmethod
    def create(cls) -> "ShellServices":
        retriever = SimpleRetriever()
        try:
            llm = create_llm_provider(settings) if settings.llm_enabled else None
        except Exception:
            llm = None
        try:
            candidate_client = ZoteroClient()
            zotero_client = candidate_client if candidate_client.is_available() else None
        except Exception:
            zotero_client = None
        return cls(
            retriever=retriever,
            definitions=DefinitionSearchStrategy(retriever=retriever, llm=llm),
            examples=ExampleSearchStrategy(retriever=retriever, llm=llm),
            paper_summary=PaperSummaryStrategy(retriever=retriever, llm=llm, zotero_client=zotero_client),
            llm=llm,
        )

    def with_llm(self, llm: LLMProvider | None) -> "ShellServices":
        self.llm = llm
        self.definitions.llm = llm
        self.examples.llm = llm
        self.paper_summary.llm = llm
        return self