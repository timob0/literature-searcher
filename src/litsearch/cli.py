from __future__ import annotations

from pathlib import Path
import shutil
import sys

import typer

from litsearch.config import settings
from litsearch.indexing.incremental import IncrementalIndexer
from litsearch.indexing.indexer import PDFIndexer
from litsearch.indexing.manifest import IndexManifest
from litsearch.indexing.incremental import INDEX_PIPELINE_VERSION
from litsearch.indexing.audit import IndexAudit
from litsearch.indexing.sync_service import ZoteroSyncService
from litsearch.llm.factory import create_llm_provider
from litsearch.retrieval.definitions import DefinitionSearchStrategy
from litsearch.retrieval.examples import ExampleSearchStrategy
from litsearch.retrieval.paper_summary import PaperSummaryStrategy
from litsearch.retrieval.retriever import SimpleRetriever
from litsearch.storage.chroma import ChromaVectorStore
from litsearch.zotero.client import ZoteroClient

app = typer.Typer(add_completion=False)


@app.command()
def shell() -> None:
    """Launch the persistent interactive research shell."""
    from litsearch.shell.app import run_shell

    run_shell()


@app.command()
def status() -> None:
    typer.echo("Status: academic literature search is ready.")
    typer.echo(f"Index pipeline version: {INDEX_PIPELINE_VERSION}")
    try:
        manifest = IndexManifest()
        indexed_documents = sum(entry.index_status == "indexed" for entry in manifest.all_entries())
        typer.echo(f"Indexed documents: {indexed_documents}")
    except Exception as exc:
        typer.echo(f"Indexed documents: unavailable ({exc})")

    try:
        vector_store = ChromaVectorStore()
        typer.echo(f"Indexed chunks: {vector_store.count()}")
    except Exception as exc:
        typer.echo(f"Indexed chunks: unavailable ({exc})")

    try:
        zotero = ZoteroClient()
        if not zotero.is_available():
            typer.echo("Zotero collections: unavailable (start Zotero with the local API enabled)")
        else:
            collections = zotero.get_collections()
            typer.echo("Zotero collections available for query:")
            if collections:
                for key, name in collections:
                    typer.echo(f"  {name} ({key})")
            else:
                typer.echo("  None")
    except Exception as exc:
        typer.echo(f"Zotero collections: unavailable ({exc})")

    if not settings.llm_enabled:
        typer.echo("LLM provider: disabled")
    else:
        try:
            llm = create_llm_provider(settings)
            health = llm.health_check()
            typer.echo(f"LLM provider: {health.provider}")
            typer.echo(f"LLM model: {health.model}")
            if health.endpoint:
                typer.echo(f"LLM endpoint: {health.endpoint}")
            typer.echo(f"LLM status: {'connected' if health.available else 'unavailable'}")
            if health.error:
                typer.echo(f"LLM detail: {health.error}")
        except Exception as exc:
            typer.echo(f"LLM provider: unavailable ({exc})")


@app.command("audit-index")
def audit_index() -> None:
    """Check indexed chunks for page, size, and manifest consistency."""
    try:
        audit = IndexAudit.run(IndexManifest(), ChromaVectorStore())
    except Exception as exc:
        typer.echo(f"Index audit unavailable: {exc}")
        raise typer.Exit(code=1)
    typer.echo("Index audit")
    typer.echo(f"Documents: {audit.documents}")
    typer.echo(f"Chunks: {audit.chunks}")
    typer.echo(f"Oversized chunks: {audit.oversized_chunks}")
    typer.echo(f"Invalid page ranges: {audit.invalid_page_ranges}")
    typer.echo(f"All-page-1 anomalies: {audit.all_page_one_anomalies}")
    typer.echo(f"Manifest page mismatches: {audit.manifest_page_mismatches}")
    typer.echo(f"Potential duplicate chunks: {audit.potential_duplicate_chunks}")
    typer.echo(f"Index pipeline version: {INDEX_PIPELINE_VERSION}")
    typer.echo(f"Status: {'PASS' if audit.passed else 'FAIL'}")
    if not audit.passed:
        raise typer.Exit(code=1)


@app.command("documents")
def documents(
    index_status: str | None = typer.Option(None, "--status", help="Only show entries with this manifest status."),
    show_path: bool = typer.Option(False, "--path", help="Show the full PDF storage path."),
    show_collections: bool = typer.Option(False, "--collection", help="Show the Zotero collections containing each PDF."),
) -> None:
    """List source PDFs recorded in the local indexing manifest."""
    entries = IndexManifest().all_entries()
    if index_status:
        entries = [entry for entry in entries if entry.index_status == index_status]

    typer.echo(f"Indexed source PDFs: {len(entries)}")
    if not entries:
        typer.echo("No matching source PDFs found.")
        return

    zotero = ZoteroClient() if show_collections else None
    for entry in entries:
        citation = entry.citation_key or "no citation key"
        chunks = entry.number_of_chunks if entry.number_of_chunks is not None else 0
        document_name = entry.pdf_path if show_path else Path(entry.pdf_path).name
        details = f"{entry.index_status}; {chunks} chunks; {citation}"
        if zotero is not None:
            try:
                collections = zotero.get_item_collections(entry.parent_key)
                details += f"; collections: {', '.join(collections) if collections else 'none'}"
            except Exception:
                details += "; collections: unavailable"
        typer.echo(f"- {document_name} [{details}]")


def _print_sync_summary(prefix: str, summary: object) -> None:
    typer.echo(f"{prefix}: discovered {summary.pdfs_discovered} PDFs")
    typer.echo(f"{prefix}: indexed {summary.new_pdfs_indexed} new, reindexed {summary.changed_pdfs_reindexed}")
    typer.echo(f"{prefix}: skipped {summary.unchanged_pdfs_skipped} unchanged, removed {summary.deleted_pdfs_removed}")
    typer.echo(f"{prefix}: failed {summary.failed_pdfs}, chunks created {summary.chunks_created}")
    typer.echo(f"{prefix}: abstract fallbacks {summary.abstract_sources_indexed}, skipped without citation key {summary.skipped_without_citation}")


def _read_collection_names(collections_file: Path) -> list[str]:
    if not collections_file.exists():
        return []
    return [
        line.strip()
        for line in collections_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _sync_library(*, limit: int | None, collection_id: str | None, collections_file: Path | None = None, rebuild_first: bool = False) -> None:
    prefix = "Rebuild" if rebuild_first else "Sync"
    zotero = ZoteroClient()
    if not zotero.is_available():
        typer.echo(f"{prefix}: Zotero is unavailable. Start Zotero with the local API enabled.")
        return

    vector_store = ChromaVectorStore()
    manifest = IndexManifest()
    if rebuild_first:
        vector_store.reset()
        for entry in manifest.all_entries():
            manifest.delete_entry(entry.attachment_key)
        typer.echo("Rebuild: cleared the vector collection and indexing manifest.")

    collection_ids = None
    if collections_file is not None:
        collection_names = _read_collection_names(collections_file)
        if collection_names:
            collection_ids = zotero.get_collection_ids_for_names(collection_names)
            typer.echo(f"{prefix}: expanded {len(collection_names)} configured collections to {len(collection_ids)} collections including descendants.")

    typer.echo(f"{prefix}: collecting candidate documents from Zotero.")
    sys.stdout.flush()
    def show_collection_progress(position: int, total: int | None, collection_key: str) -> None:
        progress_total = total if total is not None else "?"
        typer.echo(f"{prefix}: Zotero page {position} of {progress_total} queried ({collection_key})")
        sys.stdout.flush()

    candidate_items = (
        zotero.get_library_items_for_collections(collection_ids, limit=limit, progress=show_collection_progress)
        if collection_ids is not None
        else zotero.get_library_items(limit=limit, collection_id=collection_id)
    )
    candidate_count = len(candidate_items)
    typer.echo(f"{prefix}: found {candidate_count} candidate documents; loading the embedding model.")
    sys.stdout.flush()
    if candidate_count == 0:
        typer.echo(f"{prefix}: 0 of 0 documents processed.")

    indexer = PDFIndexer(vector_store=vector_store)
    def show_progress(position: int, total: int, item: object) -> None:
        label = getattr(item, "citation_key", None) or getattr(item, "item_key", "unknown")
        typer.echo(f"{prefix}: {position} of {total} documents processed ({label})")

    summary = ZoteroSyncService(
        client=zotero,
        indexer=indexer,
        incremental=IncrementalIndexer(manifest),
    ).sync(limit=limit, collection_id=collection_id, collection_ids=collection_ids, progress=show_progress, items=candidate_items)
    _print_sync_summary("Rebuild" if rebuild_first else "Sync", summary)


@app.command()
def sync(
    limit: int = typer.Option(0, min=0, help="Maximum Zotero parent items to inspect; 0 means all matching items."),
    collection_id: str | None = typer.Option(None, help="Optional Zotero collection key."),
    collections_file: Path = typer.Option(Path(settings.zotero_collection_file), "--collections-file", help="Line-based file of Zotero collection names to include recursively."),
) -> None:
    typer.echo("Sync: indexing PDFs from the local Zotero library.")
    try:
        _sync_library(limit=limit or None, collection_id=collection_id, collections_file=collections_file)
    except Exception as exc:
        typer.echo(f"Sync: backend unavailable: {exc}")
        typer.echo("Sync: ensure Zotero, Chroma, and the local embedding model are available.")


@app.command()
def rebuild(
    limit: int = typer.Option(0, min=0, help="Maximum Zotero parent items to inspect; 0 means all matching items."),
    collection_id: str | None = typer.Option(None, help="Optional Zotero collection key."),
    collections_file: Path = typer.Option(Path(settings.zotero_collection_file), "--collections-file", help="Line-based file of Zotero collection names to include recursively."),
) -> None:
    typer.echo("Rebuild: reindexing the local Zotero library.")
    try:
        _sync_library(limit=limit or None, collection_id=collection_id, collections_file=collections_file, rebuild_first=True)
    except Exception as exc:
        typer.echo(f"Rebuild: backend unavailable: {exc}")
        typer.echo("Rebuild: ensure Zotero, Chroma, and the local embedding model are available.")


@app.command()
def purge(
    confirm: bool = typer.Option(False, "--yes", help="Confirm deletion of the local vector collection and manifest."),
) -> None:
    """Clear all indexed chunks and manifest records."""
    if not confirm:
        typer.echo("Purge not performed. Re-run with --yes to clear the local index.")
        return
    vector_store_error = None
    try:
        vector_store = ChromaVectorStore()
        vector_store.reset()
    except Exception as exc:
        vector_store_error = str(exc)
    manifest = IndexManifest()
    entries = manifest.all_entries()
    for entry in entries:
        manifest.delete_entry(entry.attachment_key)
    storage_path = Path(settings.chroma_path).expanduser()
    shutil.rmtree(storage_path, ignore_errors=True)
    storage_path.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Purge: removed {len(entries)} manifest records and deleted {storage_path}.")
    if vector_store_error:
        typer.echo(f"Purge: Chroma reset warning: {vector_store_error}")
    typer.echo("Purge: restart Chroma before indexing again.")


@app.command()
def define(concept: str) -> None:
    retriever = SimpleRetriever()
    strategy = DefinitionSearchStrategy(retriever=retriever)
    typer.echo(f"DEFINE results for: {concept}")

    if retriever.embedding_provider is None or retriever.vector_store is None:
        typer.echo("No retrieval backend is currently available. Start Chroma and ensure the embedding model is accessible.")
        raise typer.Exit(code=0)

    results = strategy.define(concept, top_k=3)
    if not results:
        typer.echo("No definition results were found.")
        raise typer.Exit(code=0)
    for idx, hit in enumerate(results, start=1):
        typer.echo(f"{idx}. {hit.paper_title or 'Unknown paper'}")
        typer.echo(f"   {hit.definition_text}")
        if hit.citation_key:
            typer.echo(f"   Citation: {hit.citation_key}")
        if hit.page is not None:
            typer.echo(f"   Page: {hit.page}")


@app.command()
def examples(concept: str) -> None:
    retriever = SimpleRetriever()
    strategy = ExampleSearchStrategy(retriever=retriever)
    typer.echo(f"EXAMPLES results for: {concept}")

    if retriever.embedding_provider is None or retriever.vector_store is None:
        typer.echo("No retrieval backend is currently available. Start Chroma and ensure the embedding model is accessible.")
        raise typer.Exit(code=0)

    results = strategy.search(concept, top_k=3)
    if not results:
        typer.echo("No example results were found.")
        raise typer.Exit(code=0)
    for idx, hit in enumerate(results, start=1):
        typer.echo(f"{idx}. {hit.paper_title or 'Unknown paper'}")
        typer.echo(f"   {hit.example_text}")
        if hit.citation_key:
            typer.echo(f"   Citation: {hit.citation_key}")
        if hit.page is not None:
            typer.echo(f"   Page: {hit.page}")


@app.command()
def summarize_paper(citation_key: str) -> None:
    retriever = SimpleRetriever()
    strategy = PaperSummaryStrategy(retriever=retriever)
    typer.echo(f"SUMMARIZE_PAPER for: {citation_key}")

    if retriever.embedding_provider is None or retriever.vector_store is None:
        typer.echo("No retrieval backend is currently available. Start Chroma and ensure the embedding model is accessible.")
        raise typer.Exit(code=0)

    result = strategy.summarize(citation_key)
    if result.insufficient_evidence and not result.stage1_points:
        typer.echo("No summary evidence was found.")
        raise typer.Exit(code=0)
    summary = result.summary
    for name, heading in (
        ("overview", "OVERVIEW"), ("topics", "TOPICS"), ("problem_or_purpose", "PURPOSE"),
        ("methodology_or_approach", "APPROACH / METHODOLOGY"), ("findings_or_contributions", "KEY FINDINGS / CONTRIBUTIONS"),
        ("limitations", "LIMITATIONS"), ("future_research", "FUTURE RESEARCH / DIRECTIONS"),
    ):
        section = getattr(summary, name)
        typer.echo(heading)
        if section.prose:
            typer.echo(f"  {section.prose}")
        for point in section.points:
            typer.echo(f"  - {point.statement}")
        if not section.prose and not section.points:
            typer.echo("  Not explicitly stated in the retrieved evidence.")


if __name__ == "__main__":
    app()
