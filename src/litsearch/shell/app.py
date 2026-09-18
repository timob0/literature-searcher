from __future__ import annotations

import cmd2
import shlex
from pathlib import Path

from litsearch.config import settings
from litsearch.llm.factory import create_llm_provider
from litsearch.llm.errors import LLMConfigurationError
from litsearch.shell.renderer import ShellRenderer
from litsearch.shell.state import ShellServices
from litsearch.shell.state import DocumentSearchHit
from litsearch.shell.opener import open_pdf
from litsearch.indexing.manifest import IndexManifest
from litsearch.retrieval.models import SearchFilters


class LiteratureShell(cmd2.Cmd):
    """Persistent cmd2 adapter over the existing retrieval strategies."""

    def __init__(self, services: ShellServices | None = None, renderer: ShellRenderer | None = None) -> None:
        super().__init__(allow_cli_args=False)
        self.services = services or ShellServices.create()
        self.renderer = renderer or ShellRenderer()
        self.prompt = "litsearch> "
        self.author_filter: str | None = None
        self.source_filter: str | None = None
        self.source_candidates: list[str] = []
        self._last_summary = None
        health = self.services.llm.health_check() if self.services.llm is not None else None
        self.renderer.render_banner(
            embedding_available=self.services.retriever.embedding_provider is not None,
            chroma_available=self.services.retriever.vector_store is not None,
            ollama_available=health.available if health is not None else False,
            llm_model=self.services.llm.model_name if self.services.llm is not None else None,
            llm_provider=health.provider if health is not None else settings.llm_provider,
            llm_endpoint=health.endpoint if health is not None else None,
        )

    def _require_query(self, arg: str) -> str | None:
        query = arg.strip()
        if not query:
            self.renderer.error("A query is required.")
            return None
        return query

    def do_set(self, arg: str) -> None:
        """set llm-provider|llm-model|llm-endpoint <value> for this session."""
        parts = shlex.split(arg)
        if parts and parts[0] == "author":
            self.author_filter = parts[1] if len(parts) > 1 else None
            self.renderer.error("Usage: set author author1,author2") if len(parts) > 2 else None
            return
        if parts and parts[0] == "sources":
            sources = [source.strip() for part in parts[1:] for source in part.split(",") if source.strip()]
            if not sources:
                self.renderer.error("Usage: set sources citationKey1 citationKey2")
                return
            self.source_filter = ",".join(dict.fromkeys(sources))
            return
        if not parts or parts[0] not in {"llm-provider", "llm-model", "llm-endpoint"}:
            return super().do_set(arg)
        if len(parts) != 2:
            self.renderer.error("Usage: set llm-provider|llm-model|llm-endpoint <value>")
            return
        setting_name, value = parts
        if setting_name == "llm-provider":
            settings.llm_provider = value
        elif setting_name == "llm-model":
            if settings.llm_provider == "ollama":
                settings.ollama_model = value
            elif settings.llm_provider == "openai":
                settings.openai_model = value
            elif settings.llm_provider == "openai_compatible":
                settings.openai_compatible_model = value
            elif settings.llm_provider == "copilot":
                settings.copilot_model = value
        elif setting_name == "llm-endpoint":
            if settings.llm_provider == "ollama":
                settings.ollama_host = value
                settings.ollama_base_url = value
            elif settings.llm_provider == "openai_compatible":
                settings.openai_compatible_base_url = value
            else:
                self.renderer.error(f"Endpoint switching is not supported for {settings.llm_provider}.")
                return
        try:
            self.services.with_llm(create_llm_provider(settings))
            console = getattr(self.renderer, "console", None)
            if console is not None:
                console.print(f"LLM provider: {settings.llm_provider} ({self.services.llm.model_name})")
        except LLMConfigurationError as exc:
            self.renderer.error(str(exc))

    def complete_set(self, text: str, line: str, begidx: int, endidx: int) -> list[str]:
        """Complete session parameters, especially citation keys for set sources."""
        try:
            parts = shlex.split(line[:begidx])
        except ValueError:
            parts = line[:begidx].split()
        if len(parts) >= 2 and parts[1] == "sources":
            prefix = text.rsplit(",", 1)[-1].casefold()
            candidates = [candidate for candidate in self.source_candidates if candidate.casefold().startswith(prefix)]
            return self.basic_complete(text, line, begidx, endidx, candidates)
        parameters = ("author", "sources", "llm-provider", "llm-model", "llm-endpoint")
        return self.basic_complete(text, line, begidx, endidx, parameters)

    def _remember_sources(self, hits: object) -> None:
        values: list[str] = []
        for hit in hits if isinstance(hits, (list, tuple)) else []:
            citation_key = getattr(hit, "citation_key", None)
            if citation_key:
                values.append(citation_key)
        for citation_key in values:
            if citation_key not in self.source_candidates:
                self.source_candidates.append(citation_key)

    def do_clear(self, arg: str) -> None:
        """clear sources|author\nClear a session-local retrieval filter."""
        setting = arg.strip().casefold()
        if setting == "sources":
            self.source_filter = None
            self.renderer.error("Cleared session source filter.")
            return
        if setting == "author":
            self.author_filter = None
            self.renderer.error("Cleared session author filter.")
            return
        return super().do_clear(arg)

    def do_show(self, arg: str) -> None:
        """show [parameter]\nDisplay one session parameter or the complete session setup."""
        parameter = arg.strip().casefold()
        values = {
            "author": self.author_filter or "(none)",
            "sources": self.source_filter or "(none)",
            "llm-provider": settings.llm_provider,
            "llm-model": self.services.llm.model_name if self.services.llm is not None else "(disabled)",
            "llm-endpoint": getattr(self.services.llm.health_check(), "endpoint", None) if self.services.llm is not None else "(none)",
        }
        if parameter:
            if parameter not in values:
                self.renderer.error(f"Unknown session parameter: {parameter}")
                return
            self.renderer.console.print(f"{parameter}: {values[parameter]}")
            return
        self.renderer.console.print("[bold]Current session setup[/bold]")
        for name, value in values.items():
            self.renderer.console.print(f"{name}: {value}")
        self.renderer.console.print(f"llm-enabled: {settings.llm_enabled}")
        self.renderer.console.print(f"define-llm-enabled: {settings.define_llm_enabled}")
        self.renderer.console.print(f"examples-llm-enabled: {settings.examples_llm_enabled}")
        self.renderer.console.print(f"paper-summary-llm-enabled: {settings.paper_summary_llm_enabled}")

    def do_open(self, arg: str) -> None:
        """open\nOpen PDFs for the session's configured citation-key sources."""
        if not self.source_filter:
            self.renderer.error("No sources are set. Use `set sources citationKey1,citationKey2` first.")
            return
        requested = [source.strip() for source in self.source_filter.split(",") if source.strip()]
        entries = {
            entry.citation_key: entry
            for entry in IndexManifest().all_entries()
            if entry.citation_key in requested and entry.index_status == "indexed"
        }
        for citation_key in requested:
            entry = entries.get(citation_key)
            if entry is None:
                self.renderer.error(f"Source not found in the index manifest: {citation_key}")
                continue
            path = entry.pdf_path
            if not path or path.startswith("abstract:"):
                self.renderer.error(f"No local PDF is available for: {citation_key}")
                continue
            if not Path(path).expanduser().exists():
                self.renderer.error(f"PDF file is missing for {citation_key}: {path}")
                continue
            try:
                open_pdf(path)
                self.renderer.console.print(f"Opened: {citation_key}")
            except Exception as exc:
                self.renderer.error(f"Could not open {citation_key}: {exc}")

    def do_search(self, arg: str) -> None:
        """search <query>\nSearch the indexed literature semantically."""
        query = self._require_query(arg)
        if query is None:
            return
        try:
            results = self.services.retriever.retrieve(query, top_k=settings.display_top_papers)
            self._remember_sources(results)
            self.renderer.render_hits(f"SEARCH results for: {query}", results, "text")
        except Exception as exc:
            self.renderer.error(f"Search failed: {exc}")

    def do_find(self, arg: str) -> None:
        """find [--full] <concept>\nGroup semantic Chroma hits by document without LLM analysis."""
        try:
            parts = shlex.split(arg)
        except ValueError as exc:
            self.renderer.error(f"Find failed: {exc}")
            return
        full = "--full" in parts
        author, parts = self._extract_author(parts)
        query = " ".join(part for part in parts if part != "--full")
        concept = self._require_query(query)
        if concept is None:
            return
        try:
            chunks = self.services.retriever.retrieve(concept, filters=SearchFilters(author=author or self.author_filter), top_k=settings.retrieval_top_k)
            self._remember_sources(chunks)
            results = self._group_find_hits(chunks)
            self.renderer.render_document_hits(f"FIND results for: {concept}", results, full=full, query=concept)
        except Exception as exc:
            self.renderer.error(f"Find failed: {exc}")

    @staticmethod
    def _group_find_hits(chunks: list[object]) -> list[DocumentSearchHit]:
        grouped: dict[str, list[object]] = {}
        for chunk in chunks:
            key = getattr(chunk, "citation_key", None) or getattr(chunk, "parent_key", None) or getattr(chunk, "attachment_key", "")
            grouped.setdefault(key, []).append(chunk)
        documents = [
            DocumentSearchHit(
                citation_key=getattr(group[0], "citation_key", None),
                title=getattr(group[0], "title", None),
                authors=getattr(group[0], "authors", []),
                year=getattr(group[0], "year", None),
                zotero_link=getattr(max(group, key=lambda chunk: chunk.score), "zotero_link", ""),
                chunks=sorted(group, key=lambda chunk: chunk.score, reverse=True),
            )
            for group in grouped.values()
        ]
        documents.sort(key=lambda document: (document.score, document.chunk_count), reverse=True)
        return documents[:settings.display_top_papers]

    def do_define(self, arg: str) -> None:
        """define <concept>\nFind definitions of a concept."""
        parts = shlex.split(arg)
        author, query_parts = self._extract_author(parts)
        concept = self._require_query(" ".join(query_parts))
        if concept is None:
            return
        try:
            define = getattr(self.services.definitions, "define", None)
            filters = SearchFilters(author=author or self.author_filter, citation_key=self.source_filter)
            if define is not None:
                try:
                    results = define(concept, top_k=5, filters=filters)
                except TypeError:
                    results = define(concept, top_k=5)
            else:
                try:
                    results = self.services.definitions.search(concept, top_k=5, filters=filters)
                except TypeError:
                    results = self.services.definitions.search(concept, top_k=5)
            render_definitions = getattr(self.renderer, "render_definitions", None)
            if render_definitions is not None:
                self._remember_sources(results)
                render_definitions(concept, results)
            else:
                self.renderer.render_hits(f"DEFINE results for: {concept}", results, "definition_text")
        except Exception as exc:
            self.renderer.error(f"Definition search failed: {exc}")

    def do_explain(self, arg: str) -> None:
        """explain <concept>\nExplain a concept using expanded neighboring chunks."""
        parts = shlex.split(arg)
        author, query_parts = self._extract_author(parts)
        concept = self._require_query(" ".join(query_parts))
        if concept is None:
            return
        try:
            try:
                result = self.services.definitions.explain(concept, top_k=5, filters=SearchFilters(author=author or self.author_filter, citation_key=self.source_filter))
            except TypeError:
                result = self.services.definitions.explain(concept, top_k=5)
            render_explanation = getattr(self.renderer, "render_explanation", None)
            if render_explanation is not None:
                render_explanation(result)
            else:
                self.renderer.render_hits(f"EXPLAIN results for: {concept}", result, "definition_text")
        except Exception as exc:
            self.renderer.error(f"Explanation search failed: {exc}")

    @staticmethod
    def _extract_author(parts: list[str]) -> tuple[str | None, list[str]]:
        if "--author" not in parts:
            return None, parts
        index = parts.index("--author")
        if index + 1 >= len(parts):
            return None, [part for part in parts if part != "--author"]
        author = parts[index + 1]
        return author, parts[:index] + parts[index + 2:]

    def do_examples(self, arg: str) -> None:
        """examples <concept>\nFind practical examples of a concept."""
        concept = self._require_query(arg)
        if concept is None:
            return
        try:
            try:
                results = self.services.examples.search(concept, top_k=5, filters=SearchFilters(citation_key=self.source_filter))
            except TypeError:
                results = self.services.examples.search(concept, top_k=5)
            render_examples = getattr(self.renderer, "render_examples", None)
            if render_examples is not None:
                self._remember_sources(results)
                render_examples(concept, results)
            else:
                self.renderer.render_hits(f"EXAMPLES results for: {concept}", results, "example_text")
        except Exception as exc:
            self.renderer.error(f"Example search failed: {exc}")

    def do_summarize(self, arg: str) -> None:
        """summarize <citation-key>\nBuild a concise, evidence-grounded document summary."""
        citation_key = self._require_query(arg)
        if citation_key is None:
            return
        try:
            result = self.services.paper_summary.summarize(citation_key)
            self._last_summary = result
            self.renderer.render_summary(result)
        except Exception as exc:
            self.renderer.error(f"Paper summary failed: {exc}")

    def do_evidence(self, arg: str) -> None:
        """evidence <point-id>\nShow the source passage behind a summarize point ID (e.g. evidence P03)."""
        point_id = arg.strip()
        result = getattr(self, "_last_summary", None)
        if not point_id or result is None:
            self.renderer.error("Run summarize first, then use: evidence <point-id>")
            return
        point = next((p for p in result.stage1_points if p.point_id == point_id), None)
        if point is None:
            self.renderer.error(f"No evidence found for point ID {point_id}. Run summarize first.")
            return
        console = self.renderer.console
        console.print(f"[bold]{point_id}[/bold] ({point.facet}, {point.confidence} confidence)")
        console.print(point.statement)
        for evidence_id in point.evidence_ids:
            passage = result.evidence_by_id.get(evidence_id)
            if passage is None:
                continue
            page = f" (p. {passage.page})" if passage.page else ""
            console.print(f"\n[{evidence_id}]{page}")
            console.print(passage.text)


    def do_exit(self, _: str) -> bool:
        """exit\nLeave the interactive shell."""
        return True

    do_quit = do_exit
    do_q = do_exit


def run_shell() -> None:
    LiteratureShell().cmdloop()