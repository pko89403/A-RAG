from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings loaded from environment variables.

    Uses OpenAI-compatible endpoint via `ChatOpenAI(base_url=...)`
    (Works with Azure OpenAI `/openai/v1` endpoints too)
    """

    # LLM (OpenAI-compatible endpoint, often Azure OpenAI "openai/v1" endpoints)
    openai_endpoint: str | None
    openai_api_key: str | None
    openai_model_name: str | None
    openai_api_version: str | None
    openai_reasoning_effort: str | None
    openai_temperature: float | None

    azure_search_endpoint: str | None
    azure_search_api_key: str | None
    azure_search_api_version: str

    history_max_turns: int

    research_paper_index_name: str
    trace_tool_calls: bool
    trace_skills: bool
    skills_sources: tuple[str, ...]


def load_env(dotenv_path: str | Path | None = None) -> None:
    """Load environment variables from a local `.env` file.

    - If `dotenv_path` is None, uses `<repo_root>/.env` when present.
    - Does not override already-set environment variables.
    """
    if dotenv_path is None:
        repo_root = Path(__file__).resolve().parents[1]
        dotenv_path = repo_root / ".env"

    path = Path(dotenv_path)
    if not path.exists():
        return
    load_dotenv(dotenv_path=path, override=False)


def load_settings() -> Settings:
    """Load settings from `os.environ`.

    Notes:
    - `.env` loading is handled automatically.
    - Skills sources (`DEEPAGENTS_SKILLS`) are normalized to POSIX-style paths like `/skills/project/`.
    """
    load_env()
    temp = os.environ.get("OPENAI_TEMPERATURE")
    skills_sources = _parse_csv(os.environ.get("DEEPAGENTS_SKILLS", "skills/project"))
    azure_search_endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
    azure_search_api_key = os.environ.get("AZURE_SEARCH_API_KEY")
    max_turns = os.environ.get("HISTORY_MAX_TURNS", "5")
    return Settings(
        openai_endpoint=os.environ.get("OPENAI_ENDPOINT"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        openai_model_name=os.environ.get("OPENAI_MODELNAME"),
        openai_api_version=os.environ.get("OPENAI_API_VERSION"),
        openai_reasoning_effort=os.environ.get("OPENAI_REASONING_EFFORT"),
        openai_temperature=float(temp) if temp else None,
        azure_search_endpoint=azure_search_endpoint,
        azure_search_api_key=azure_search_api_key,
        azure_search_api_version=os.environ.get(
            "AZURE_SEARCH_API_VERSION", "2023-11-01"
        ),
        history_max_turns=int(max_turns) if max_turns.isdigit() else 5,
        research_paper_index_name=os.environ.get(
            "AZURE_SEARCH_API_RESEARCH_PAPER_INDEX",
            "research-paper-index-v1",
        ),
        trace_tool_calls=os.environ.get("DEEPAGENTS_TRACE_TOOLS", "false") == "true",
        trace_skills=os.environ.get("DEEPAGENTS_TRACE_SKILLS", "false") == "true",
        skills_sources=tuple(_normalize_skill_source(s) for s in skills_sources),
    )


def _parse_csv(value: str) -> list[str]:
    """Parse a comma-separated list from an env var."""
    return [p.strip() for p in value.split(",") if p.strip()]


def _normalize_skill_source(source: str) -> str:
    """Normalize skill source strings to the DeepAgents-friendly format.

    DeepAgents skills are configured as directory "sources" relative to the backend root.
    We normalize to:
    - forward slashes
    - leading slash
    - trailing slash
    Example: `skills/project` -> `/skills/project/`
    """
    s = source.replace("\\", "/").strip()
    if not s.startswith("/"):
        s = "/" + s
    if not s.endswith("/"):
        s = s + "/"
    return s
