from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING
from typing_extensions import TypedDict

from deepagents import create_deep_agent
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend
from langchain_core.globals import set_debug
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from paper_analysis_deepagents.config import Settings
from paper_analysis_deepagents.llm import build_embed_query, build_llm
from paper_analysis_deepagents.tools.azure_search import AzureSearchConfig
from paper_analysis_deepagents.tools.research_paper_search import (
    make_chunk_read,
    make_cite_sources,
    make_contextual_search,
    make_keyword_search,
    make_semantic_hybrid_search,
)
from paper_analysis_deepagents.tools.research_paper_think import (
    research_paper_think_tool,
)

logging.basicConfig(level=logging.INFO)
set_debug(False)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["LANGCHAIN_VERBOSE"] = "false"

DEFAULT_INDEX_NAME = "research-paper-index-v1"
RESEARCH_PAPER_SUMMARY_AGENT_PROMPT_ROUTE = "/memory/RESEARCH_PAPER_SUMMARY_AGENTS.md"
RESEARCH_PAPER_SUMMARY_PERSONA_PROMPT = (
    "당신은 연구 논문 분석 및 요약을 전문으로 하는 AI 리서치 어시스턴트다. "
    "논문의 핵심 기여, 방법론, 실험 결과, 한계점을 정확하게 파악하고 설명한다. "
    "질문 의도를 끝까지 파악해 성심성의껏 친절하게 응답한다. "
    "핵심 결론을 먼저 제시한 뒤 근거와 맥락을 충분히 설명해, 사용자가 바로 이해하고 활용할 수 있도록 비교적 자세하고 긴 답변을 제공한다. "
    "모든 답변은 한국어로 작성한다."
)
RESEARCH_PAPER_SUMMARY_SYSTEM_PROMPT = RESEARCH_PAPER_SUMMARY_PERSONA_PROMPT


class LoggingFilesystemBackend(FilesystemBackend):
    """Filesystem backend that logs skill reads (/skills/.../SKILL.md)."""

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:  # type: ignore[override]
        try:
            if "/skills/" in str(file_path) and str(file_path).endswith("SKILL.md"):
                logger.info("[SKILL] read SKILL.md: %s", file_path)
        except Exception:
            # Best effort logging; do not break normal flow.
            pass
        return super().read(file_path, offset=offset, limit=limit)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:  # type: ignore[override]
        try:
            if "/skills/" in str(file_path) and str(file_path).endswith("SKILL.md"):
                logger.info("[SKILL] read SKILL.md: %s", file_path)
        except Exception:
            # Best effort logging; do not break normal flow.
            pass
        return await super().aread(file_path, offset=offset, limit=limit)


class ResearchPaperSummaryContextSchema(TypedDict, total=False):
    """Runtime configurable values for research paper summary agent."""

    thread_id: str
    session_scope: str
    request_id: str


if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.store.base import BaseStore


def build_research_paper_summary_agent(
    settings: Settings,
    llm: BaseChatModel | None = None,
    *,
    checkpointer: "BaseCheckpointSaver | None" = None,
    store: "BaseStore | None" = None,
):
    """Build a minimal DeepAgents RAG agent for research paper Q&A."""
    model = llm or build_llm(settings)
    repo_root = Path(__file__).resolve().parents[1]
    memory_root = repo_root / "memory"
    skills_root = repo_root / "skills"
    index_name = settings.research_paper_index_name or DEFAULT_INDEX_NAME

    if not settings.azure_search_endpoint or not settings.azure_search_api_key:
        raise ValueError(
            "AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY must be configured"
        )

    # MemoryMiddleware reads through backend virtual paths.
    # Keep memory on a dedicated route so it does not depend on "/workspace/" mapping.
    # 운영 규칙(agent_prompt)은 memory 문서로 분리 관리한다.
    memory_files = [RESEARCH_PAPER_SUMMARY_AGENT_PROMPT_ROUTE]

    search_config = AzureSearchConfig(
        endpoint=settings.azure_search_endpoint,
        api_key=settings.azure_search_api_key,
        api_version=settings.azure_search_api_version,
    )

    # Use provided checkpointer or default to in-memory MemorySaver
    agent_checkpointer = checkpointer if checkpointer is not None else MemorySaver()

    # chunk_read and cite_sources must share the same store so citations can be
    # selected only from chunks read in the current turn.
    agent_store = store if store is not None else InMemoryStore()
    embed_query = build_embed_query(settings)

    return create_deep_agent(
        model=model,
        tools=[
            make_semantic_hybrid_search(
                search_config,
                index_name=index_name,
                embed_query=embed_query,
            ),
            make_contextual_search(
                search_config,
                index_name=index_name,
                embed_query=embed_query,
            ),
            make_keyword_search(search_config, index_name=index_name),
            research_paper_think_tool,
            make_chunk_read(
                search_config,
                index_name=index_name,
                store=agent_store,
                workspace_root=repo_root,
                path_only=True,
            ),
            make_cite_sources(store=agent_store),
        ],
        system_prompt=RESEARCH_PAPER_SUMMARY_SYSTEM_PROMPT,
        memory=memory_files,
        context_schema=ResearchPaperSummaryContextSchema,
        checkpointer=agent_checkpointer,
        store=agent_store,
        backend=lambda runtime: CompositeBackend(
            default=StateBackend(runtime),
            routes={
                "/memories/": StoreBackend(runtime),
                "/memory/": FilesystemBackend(root_dir=memory_root, virtual_mode=True),
                "/skills/": LoggingFilesystemBackend(
                    root_dir=skills_root, virtual_mode=True
                ),
                "/workspace/": FilesystemBackend(root_dir=repo_root, virtual_mode=True),
            },
        ),
        skills=["/skills"],
        name="research_paper_summary_agent",
        debug=True,
    )
