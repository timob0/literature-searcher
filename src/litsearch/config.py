from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # Resolve .env relative to the repo root so settings load regardless of the invoking cwd.
    model_config = SettingsConfigDict(env_file=str(_REPO_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore")

    zotero_api_url: str = Field(default="http://localhost:23119/api/")
    zotero_storage_dir: str = Field(default="~/Zotero/storage")
    zotero_collection_file: str = Field(default="data/zotero_collections.txt")
    zotero_workers: int = Field(default=8)
    extraction_workers: int = Field(default=4)
    chroma_host: str = Field(default="localhost")
    chroma_port: int = Field(default=8000)
    chroma_collection: str = Field(default="literature_chunks")
    chroma_path: str = Field(default="data/chroma")
    embedding_model: str = Field(default="BAAI/bge-m3")
    embedding_device: str = Field(default="cpu")
    embedding_batch_size: int = Field(default=32)
    embedding_normalize: bool = Field(default=True)
    torch_num_threads: int | None = Field(default=None)
    chunk_target_tokens: int = Field(default=500)
    chunk_min_tokens: int = Field(default=200)
    chunk_max_tokens: int = Field(default=800)
    chunk_overlap_tokens: int = Field(default=75)
    retrieval_top_k: int = Field(default=50)
    display_top_papers: int = Field(default=15)
    define_candidate_top_k: int = Field(default=30)
    define_exact_phrase_bonus: float = Field(default=0.35)
    define_normalized_phrase_bonus: float = Field(default=0.25)
    define_partial_term_bonus: float = Field(default=0.10)
    define_section_bonus: float = Field(default=0.05)
    define_reference_section_penalty: float = Field(default=1.0)
    define_context_before: int = Field(default=1)
    define_context_after: int = Field(default=1)
    define_context_max_tokens: int = Field(default=1800)
    define_context_continue_structures: bool = Field(default=True)
    define_llm_evidence_max_chars: int = Field(default=2600)
    explain_context_before: int = Field(default=1)
    explain_context_after: int = Field(default=1)
    explain_context_max_tokens: int = Field(default=2500)
    explain_context_continue_structures: bool = Field(default=True)
    explain_anchor_top_k: int = Field(default=5)
    explain_llm_evidence_max_chars: int = Field(default=2200)
    explain_extraction_batch_evidence: int = Field(default=5)
    explain_extraction_max_chars: int = Field(default=2200)
    reranker_enabled: bool = Field(default=False)
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    manifest_path: str = Field(default="data/index_manifest.sqlite")
    passage_store_path: str = Field(default="data/passage_store.sqlite")
    better_bibtex_enabled: bool = Field(default=True)
    llm_enabled: bool = Field(default=True)
    llm_provider: str = Field(default="ollama")
    ollama_host: str = Field(default="http://localhost:11434")
    ollama_base_url: str = Field(default="")
    ollama_api_key: str = Field(default="")
    ollama_model: str = Field(default="qwen3.5:latest")
    ollama_context_length: int = Field(default=8192)
    ollama_temperature: float = Field(default=0.0)
    ollama_thinking: bool = Field(default=False)
    ollama_timeout_seconds: float = Field(default=120.0)
    ollama_max_output_tokens: int = Field(default=4096)
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-5.6-luna")
    openai_reasoning_effort: str = Field(default="none")
    openai_timeout_seconds: float = Field(default=120.0)
    openai_max_output_tokens: int = Field(default=4000)
    openai_compatible_base_url: str = Field(default="http://localhost:8000/v1")
    openai_compatible_api_key: str = Field(default="")
    openai_compatible_model: str = Field(default="")
    openai_compatible_timeout_seconds: float = Field(default=300.0)
    openai_compatible_native_structured_output: bool = Field(default=False)
    copilot_cli_executable: str = Field(default="copilot")
    copilot_model: str = Field(default="")
    copilot_timeout_seconds: float = Field(default=180.0)
    define_llm_enabled: bool = Field(default=True)
    examples_llm_enabled: bool = Field(default=True)
    examples_candidate_top_k: int = Field(default=30)
    examples_context_before: int = Field(default=1)
    examples_context_after: int = Field(default=1)
    examples_context_max_tokens: int = Field(default=1800)
    examples_extraction_batch_evidence: int = Field(default=3)
    examples_extraction_max_chars: int = Field(default=6000)
    examples_max_results: int = Field(default=10)
    paper_summary_llm_enabled: bool = Field(default=True)
    summary_facet_top_k: int = Field(default=10)
    summary_context_before: int = Field(default=1)
    summary_context_after: int = Field(default=1)
    summary_context_max_tokens: int = Field(default=1800)
    summary_extraction_batch_evidence: int = Field(default=3)
    summary_extraction_max_chars: int = Field(default=2200)
    summary_target_min_words: int = Field(default=300)
    summary_target_max_words: int = Field(default=600)

    @property
    def data_dir(self) -> Path:
        return Path(self.manifest_path).parent


settings = Settings()

# Must happen before torch is imported anywhere (config is the first project import in every
# entry point), otherwise torch's native thread pools are already sized and this has no effect.
if settings.torch_num_threads:
    os.environ.setdefault("OMP_NUM_THREADS", str(settings.torch_num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(settings.torch_num_threads))
