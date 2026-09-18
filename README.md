# Academic Literature Search

This project implements a local academic evidence-retrieval workflow over Zotero PDF literature.

## Purpose

The system indexes Zotero PDF attachments, extracts text page-by-page, chunks paragraphs into semantically meaningful units, embeds them locally with BGE-M3, and stores them in Chroma for retrieval. The goal is to support academic search and evidence retrieval without an LLM dependency.

## Initial vertical slice

This repository includes the first working slice:

- Zotero parent item resolution
- PDF attachment discovery
- Better BibTeX citation key lookup when available
- PDF extraction
- paragraph/sentence preservation
- chunking
- local BGE-M3 embedding
- Chroma insertion
- semantic query by chunk
- exact paper and page provenance
- Zotero deep links

PDF page provenance uses the 1-based physical PDF page ordinal throughout
extraction, chunk metadata, Chroma, retrieval, and Zotero navigation. It is
not the printed page label: a physical PDF page 37 may display printed page
21. For a cross-page chunk, `page_start` is the first physical page and
`page_end` is the last; navigation opens `page_start`.

## Quick start

1. Create the environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e .[dev]
   ```

2. Start Chroma locally:

   ```bash
   chroma run --path ./data/chroma --host 127.0.0.1 --port 8000
   ```

3. Configure the app:

   ```bash
   cp .env.example .env
   ```

4. Run the first test slice:

   ```bash
   pytest -q
   ```

5. Start and stop the local Chroma backend:

   ```bash
   ./startup.sh
   ./shutdown.sh
   ```

6. Explore the CLI:

   ```bash
   ./litsearch.sh
   ./litsearch.sh status
   ./litsearch.sh documents
   ./litsearch.sh documents --status indexed
   ./litsearch.sh documents --status indexed --path
   ./litsearch.sh documents --status indexed --collection
   ./litsearch.sh sync --limit 20
   ./litsearch.sh sync --limit 0
   ./litsearch.sh sync --collections-file data/zotero_collections.txt
   ./litsearch.sh rebuild --collections-file data/zotero_collections.txt
   ./litsearch.sh audit-index
   ./litsearch.sh purge --yes
   ./litsearch.sh define "health literacy"
   python -m litsearch.cli status
   python -m litsearch.cli sync
   python -m litsearch.cli define "health literacy"
   python -m litsearch.cli examples "supportive leadership"
   python -m litsearch.cli summarize-paper smithMethodStudy2024
   ./litsearch.sh shell
   ```

## Interactive Research Shell

Run `litsearch shell` (or `./litsearch.sh shell`) for a persistent terminal session. The embedding model and Chroma connection are initialized once when the shell starts.

```text
litsearch> define health literacy
litsearch> examples supportive leadership
litsearch> search employee wellbeing
litsearch> summarize smithMethodStudy2024
litsearch> help
litsearch> exit
```

Phase 1 supports `search`, `find`, `define`, `explain`, `examples`, `summarize`, `help`, and `exit` (with `quit` and `q` aliases). `find` returns direct semantic Chroma hits without LLM analysis, grouped by document. By default it shows the highest-scoring matching chunk for each document; use `find --full <concept>` to show all retrieved matching chunks. `define` returns validated definitions, while `explain` also includes validated conceptual discussions. Normal Typer commands remain available, and shell settings are session-local.

The shell supports an optional author filter for `find`, `define`, and `explain`:

```text
set author Smith,Jones
find organisational culture
define organisational culture --author Taylor
explain organisational culture --author Taylor
```

Author filtering uses author metadata stored in Chroma. Existing records created before author metadata was indexed have empty author fields, so run one rebuild after upgrading before using this filter.

Restrict analysis to selected Zotero sources with a session-local citation-key filter:

```text
set sources scheinOrganizationalCultureLeadership2010,hofstedeCulturesOrganizationsSoftware2010
define organisational culture
explain organisational culture
examples organisational culture
```

Source keys may also be separated by spaces when selected through the tab-completion picker.

The source filter applies to `define`, `explain`, and `examples`. It uses the exact citation keys stored in Chroma.

Open the selected source PDFs with the system viewer:

```text
set sources scheinOrganizationalCultureLeadership2010
open
```

Multiple comma-separated sources are opened in one command. The command uses indexed manifest paths; page navigation is only available when a page locator is available, while source-level opening uses the system default PDF viewer.

Clear session filters with:

```text
clear sources
clear author
```

Inspect session parameters with:

```text
show sources
show llm-provider
show
```

The full `show` output includes active filters, LLM provider/model/endpoint, and per-task LLM enablement without displaying API keys.

`define` uses a strict two-stage pipeline: semantic candidates are expanded with neighboring chunks, Qwen/provider Stage 1 keeps only explicit or implicit definitions of the requested concept, and Stage 2 deduplicates and ranks the validated source definitions. It does not fill the result limit with related passages.

`explain` expands each semantic anchor with adjacent chunks from the same indexed document before sending evidence to Ollama. Defaults are one chunk before and after, with a 2,500-token context limit; configure `EXPLAIN_CONTEXT_BEFORE`, `EXPLAIN_CONTEXT_AFTER`, `EXPLAIN_CONTEXT_MAX_TOKENS`, or `EXPLAIN_CONTEXT_CONTINUE_STRUCTURES` to tune it. Expansion is local and does not change the Chroma index.

`explain` then performs two provider-independent LLM stages: conceptual evidence extraction (definitions, characteristics, components, dimensions, mechanisms, and perspectives) followed by cross-source synthesis. Raw expanded chunks are not printed as the explanation; every synthesized point retains evidence IDs.

## Local LLM Analysis

The optional local analysis layer connects to the existing Ollama daemon and defaults to `qwen3.5:latest`. BGE-M3 and Chroma remain responsible for finding candidate evidence; Qwen will classify, extract, or synthesize only retrieved passages in later phases. Zotero remains authoritative for citation and page provenance. Academic content is sent only to the local Ollama service.

Configure with `LLM_ENABLED`, `OLLAMA_HOST`, `OLLAMA_MODEL`, `OLLAMA_API_KEY`, `OLLAMA_CONTEXT_LENGTH`, `OLLAMA_TEMPERATURE`, `OLLAMA_THINKING`, `OLLAMA_TIMEOUT_SECONDS`, and the per-task `DEFINE_LLM_ENABLED`, `EXAMPLES_LLM_ENABLED`, and `PAPER_SUMMARY_LLM_ENABLED` settings. If Ollama is unavailable, the application continues to support embedding-only retrieval. `OLLAMA_API_KEY`, when set, is sent as a `Bearer` token in the `Authorization` header, which is required for Ollama Cloud (`https://ollama.com`) models.

## LLM Providers

The active backend is selected with `LLM_PROVIDER`: `ollama`, `openai`, `openai_compatible`, or `copilot`. Retrieval, context expansion, prompts, evidence IDs, and provenance remain provider-independent.

```text
# Local Ollama
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3.5:latest

# Remote Ollama on a trusted LAN
LLM_PROVIDER=ollama
OLLAMA_HOST=http://192.168.1.50:11434
OLLAMA_MODEL=<remote-model>

# Ollama Cloud
LLM_PROVIDER=ollama
OLLAMA_HOST=https://ollama.com
OLLAMA_MODEL=<cloud-model>
OLLAMA_API_KEY=<key>

# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=<key>
OPENAI_MODEL=gpt-5.6-luna

# Generic OpenAI-compatible server
LLM_PROVIDER=openai_compatible
OPENAI_COMPATIBLE_BASE_URL=http://192.168.1.60:8000/v1
OPENAI_COMPATIBLE_MODEL=<model>

# GitHub Copilot CLI
LLM_PROVIDER=copilot
COPILOT_CLI_EXECUTABLE=copilot
COPILOT_MODEL=
```

In the shell, provider changes are session-local:

```text
set llm-provider ollama
set llm-model qwen3.5:latest
set llm-endpoint http://192.168.1.50:11434
```

Only retrieved and context-expanded evidence is sent to the selected provider. Zotero PDFs, Chroma, and local filesystem paths are not uploaded as a corpus. Remote Ollama must separately be configured to listen on the LAN; do not expose an unauthenticated endpoint beyond a trusted LAN or protected VPN/reverse proxy. The Copilot CLI must already be installed and authenticated, and this application grants it no tools or repository access.

`sync` discovers PDF child attachments from the local Zotero API and incrementally indexes new or changed files. The page-aware ingestion pipeline is versioned; an ingestion-version change forces existing PDFs to be reindexed even when their files are unchanged. Use `audit-index` before a rebuild to check page ranges, chunk sizes, manifest counts, stale metadata, and duplicate text. Use `rebuild` to clear the local collection and manifest and reindex from scratch. Do not start a full rebuild until the audit and focused tests pass. The retrieval commands support the four explicit modes: `SEARCH` through the retriever API, `DEFINE`, `EXAMPLES`, and `SUMMARIZE_PAPER`.

The file `data/zotero_collections.txt` accepts one Zotero collection name per line. Matching collections and all nested subcollections are included during `sync` or `rebuild`; blank lines and lines beginning with `#` are ignored. The `purge` command requires `--yes` and clears the Chroma collection, manifest, and configured `CHROMA_PATH` storage directory. Stop the Chroma backend first, then restart it with `./startup.sh` after purging.

For `sync` and `rebuild`, `--limit 0` means all matching documents. Zotero discovery uses `ZOTERO_WORKERS` (default `8`) for parallel attachment lookups. PDF extraction, embedding, and Chroma writes remain serialized because the local embedding model and vector-store client are shared; this avoids multiplying model memory usage and unsafe concurrent writes.

Sync and rebuild print progress for each candidate document. Items without a Zotero citation key are skipped. If a citation-keyed item has no usable PDF attachment, its Zotero abstract is indexed instead.

## Directory layout

See the project package structure under `src/litsearch`.
