from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Type

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from paper_analysis_deepagents.tools.azure_search import (
    AzureSearchConfig,
    get_index_schema,
    search,
)

EXPECTED_RESEARCH_PAPER_FIELDS: tuple[str, ...] = (
    "id",
    "source_file",
    "type",
    "page_number",
    "title",
    "content",
    "embedding",
    "page_blob_url",
    "figure_blob_urls",
    "table_blob_urls",
    "image_descriptions",
    "table_descriptions",
    "original_caption",
    "row_count",
    "column_count",
    "node_id",
    "start_at",
    "owner",
    "process",
)

# Search/highlight fields for non-node documents.
# Base filter is always `type ne 'node'`, so node_* fields are excluded on purpose.
KEYWORD_TEXT_FIELDS: tuple[str, ...] = (
    "title",
    "content",
    "image_descriptions",
    "table_descriptions",
    "original_caption",
)
BASE_FILTER = "type ne 'node'"
DEFAULT_CHUNK_FILE_TTL_SECONDS = 600


class KeywordSearchInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "정확 문자열 매칭용 질의(고유명사/수치/약어). "
            "예: 'Transformer', 'BERT fine-tuning'"
        ),
    )
    top_k: int = Field(
        5,
        ge=1,
        le=50,
        description="반환할 최대 결과 수 (초기 탐색은 5~10 권장)",
    )
    filter: str | None = Field(
        default=None,
        description=(
            "선택적 Azure Search OData 필터. "
            "예: process eq '논문' and owner eq 'research-team'"
        ),
    )


class KeywordSearchTool(BaseTool):
    """Azure AI Search full-text keyword search tool."""

    name: str = "keyword_search"
    description: str = (
        "keyword_search (정확 키워드 검색)\n"
        "주 사용처: 고유명사/약어/수치/논문명처럼 정확 매칭이 중요한 질의.\n"
        "작동: Azure AI Search BM25 계열 키워드 검색(대소문자 비구분).\n"
        "운영: 질의어와 텍스트 필드의 정확 일치 신호를 우선 반영한다.\n"
        "filter 사용 가능: source_file, type, page_number, start_at, owner, process\n"
        "입력 JSON:\n"
        '{ "query": "string", "top_k": 5, "filter": "optional OData filter" }\n'
        "출력 JSON 배열:\n"
        '[ { "id": "string", "snippet": "string" } ]\n'
        "운영 규칙:\n"
        "- 내부 기본 필터로 `type ne 'node'`가 항상 AND 결합됨.\n"
        "- 이 도구는 점진적 정보 노출을 위해 id+snippet만 제공함.\n"
        "- 결과는 relevance 점수 기준 상위 top_k를 반환함."
    )
    args_schema: Type[BaseModel] = KeywordSearchInput

    config: AzureSearchConfig
    index_name: str

    def _run(
        self, query: str, top_k: int = 5, filter: str | None = None
    ) -> list[dict[str, Any]]:
        return run_keyword_search(
            config=self.config,
            index_name=self.index_name,
            query=query,
            top_k=top_k,
            filter=filter,
        )


def make_keyword_search(
    config: AzureSearchConfig, *, index_name: str
) -> KeywordSearchTool:
    """Factory for BaseTool-based full-text keyword search."""
    return KeywordSearchTool(config=config, index_name=index_name)


class SemanticHybridSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="의미 기반 하이브리드 검색용 자연어 질의",
    )
    top_k: int = Field(
        5,
        ge=1,
        le=50,
        description="반환할 최대 결과 수 (초기 탐색은 8~12 권장)",
    )
    filter: str | None = Field(
        default=None,
        description=(
            "선택적 Azure Search OData 필터. "
            "예: process eq '논문' and start_at ge '2024'"
        ),
    )


class SemanticHybridSearchTool(BaseTool):
    """Azure AI Search vector+keyword hybrid search tool."""

    name: str = "semantic_hybrid_search"
    description: str = (
        "semantic_hybrid_search (의미 + 키워드 하이브리드)\n"
        "주 사용처: 의도/개념 유사성 확장이 필요한 질의.\n"
        "작동: 질의 임베딩 + 키워드 검색 결합(벡터 필드 `embedding`).\n"
        "운영: 의미 유사도와 키워드 일치도를 함께 반영한다.\n"
        "filter 사용 가능: source_file, type, page_number, start_at, owner, process\n"
        "입력 JSON:\n"
        '{ "query": "string", "top_k": 5, "filter": "optional OData filter" }\n'
        "출력 JSON 배열:\n"
        '[ { "id": "string", "snippet": "string" } ]\n'
        "운영 규칙:\n"
        "- 내부 기본 필터로 `type ne 'node'`가 항상 AND 결합됨.\n"
        "- 이 도구는 점진적 정보 노출을 위해 id+snippet만 제공함.\n"
        "- 결과는 relevance 점수 기준 상위 top_k를 반환함."
    )
    args_schema: Type[BaseModel] = SemanticHybridSearchInput

    config: AzureSearchConfig
    index_name: str
    embed_query: Callable[[str], list[float]]

    def _run(
        self, query: str, top_k: int = 5, filter: str | None = None
    ) -> list[dict[str, Any]]:
        return run_semantic_hybrid_search(
            config=self.config,
            index_name=self.index_name,
            embed_query=self.embed_query,
            query=query,
            top_k=top_k,
            filter=filter,
        )


def make_semantic_hybrid_search(
    config: AzureSearchConfig,
    *,
    index_name: str,
    embed_query: Callable[[str], list[float]],
) -> SemanticHybridSearchTool:
    """Factory for BaseTool-based semantic hybrid search."""
    return SemanticHybridSearchTool(
        config=config,
        index_name=index_name,
        embed_query=embed_query,
    )


class ContextualSearchInput(BaseModel):
    query: str = Field(..., description="페이지 단위 맥락 복원을 위한 자연어 질의")
    top_k: int = Field(5, ge=1, le=50, description="최종 반환 결과 수")
    filter: str | None = Field(
        default=None,
        description="선택적 OData 필터 (내부 타입 allowlist와 AND 결합)",
    )


class ContextualSearchTool(BaseTool):
    """Hybrid search with page-centric replacement for contextual retrieval."""

    name: str = "contextual_search"
    description: str = (
        "contextual_search (맥락 중심 하이브리드 검색)\n"
        "주 사용처: 페이지 단위 맥락 복원.\n"
        "작동:\n"
        "- 대상 타입을 page_chunk/figure_image/table_image로 제한한 하이브리드 검색.\n"
        "- 상위 figure/table(20%)는 부모 page_chunk로 승격해 맥락 단절 완화.\n"
        "- 내부 후보를 넓게 수집 후 중복 제거, 점수 기준 top_k 반환.\n"
        "운영: 페이지 중심 후보를 우선 재구성한다.\n"
        "입력 JSON:\n"
        '{ "query": "string", "top_k": 5, "filter": "optional OData filter" }\n'
        "출력 JSON 배열:\n"
        '[ { "id": "string", "snippet": "string" } ]\n'
        "운영 규칙:\n"
        "- 노드(type='node')는 검색 대상에서 제외됨.\n"
        "- 이 도구는 점진적 정보 노출을 위해 id+snippet만 제공함.\n"
        "- 결과는 relevance 점수 기준 상위 top_k를 반환함."
    )
    args_schema: Type[BaseModel] = ContextualSearchInput

    config: AzureSearchConfig
    index_name: str
    embed_query: Callable[[str], list[float]]

    def _run(
        self,
        query: str,
        top_k: int = 5,
        filter: str | None = None,
    ) -> list[dict[str, Any]]:
        return run_contextual_search(
            config=self.config,
            index_name=self.index_name,
            embed_query=self.embed_query,
            query=query,
            top_k=top_k,
            filter=filter,
        )


def make_contextual_search(
    config: AzureSearchConfig,
    *,
    index_name: str,
    embed_query: Callable[[str], list[float]],
) -> ContextualSearchTool:
    return ContextualSearchTool(
        config=config,
        index_name=index_name,
        embed_query=embed_query,
    )


class ChunkReadInput(BaseModel):
    ids: list[str] = Field(
        ...,
        min_length=1,
        description="이전 검색 도구 결과에서 선택한 문서 ID 목록",
    )
    adjacent_window: int = Field(
        default=0,
        ge=0,
        le=2,
        description="인접 청크 확장 폭. 0이면 확장 없음, 1이면 ±1 청크 추가(연속 문맥 확인용)",
    )


class CiteSourcesInput(BaseModel):
    ids: list[str] = Field(
        ...,
        min_length=1,
        description="이번 턴 최종 답변에서 실제 인용한 문서 ID 목록",
    )
    reason: str | None = Field(
        default=None,
        description="인용 근거 선택 사유(선택)",
    )


class ChunkReadTool(BaseTool):
    """Read full payload by id with duplicate-read tracking."""

    name: str = "chunk_read"
    description: str = (
        "chunk_read (청크 정독)\n"
        "주 사용처: 검색 결과(id+snippet)에서 관련성이 높은 후보를 선택해 원문/메타데이터를 깊게 읽는 핵심 도구.\n"
        "입력 JSON:\n"
        '{ "ids": ["string"], "adjacent_window": 0 }\n'
        "출력 JSON 객체:\n"
        '{ "requested_ids": [...], "resolved_ids": [...], "results": [ { "id": "string", "already_read": true|false, '
        '"notice": "optional", "content": "string", "metadata": { "source_file": "...", "page_number": 1, '
        '"page_blob_url": "optional", "figure_blob_urls": ["optional"], "table_blob_urls": ["optional"], ... } } ] }\n'
        "동작:\n"
        "- 동일 id 중복 읽기 추적은 현재 턴(request_id) 범위에서만 적용.\n"
        "- 이미 읽은 id 재요청 시:\n"
        '  { "id": "...", "already_read": true, "notice": "This chunk has been read before" }\n'
        "- 최초 읽기 시 full content + metadata 반환.\n"
        "- thread_id가 있으면 원문을 `/workspace/threads/{thread_id}/chunks/{hash}.json` 경로에 저장.\n"
        "- 저장된 chunk 파일은 TTL 600초(10분) 경과 시 자동 정리.\n"
        "- 이 에이전트는 path-only 모드로 설정될 수 있으므로, path가 오면 read_file(path)로 원문을 읽어야 함.\n"
        "- adjacent_window>0 이면 page/table/figure 번호 기반 인접 청크를 함께 조회.\n"
        "- 출력이 큰 경우 FilesystemMiddleware가 자동 오프로딩 가능.\n"
        "- 최종 답변 본문에는 id/source_file/page_number 및 도구명을 그대로 노출하지 말고 내용 중심으로 근거를 요약할 것."
    )
    args_schema: Type[BaseModel] = ChunkReadInput

    config: AzureSearchConfig
    index_name: str
    store: BaseStore | None = None
    workspace_root: str | None = None
    path_only: bool = False
    chunk_file_ttl_seconds: int | None = None

    def _run(
        self,
        ids: list[str],
        adjacent_window: int = 0,
        config: RunnableConfig = None,
    ) -> dict[str, Any]:
        requested_ids = _normalize_ids(ids)
        if not requested_ids:
            return {
                "requested_ids": [],
                "resolved_ids": [],
                "results": [],
                "notice": "ids is required",
            }

        resolved_ids = (
            _expand_adjacent_chunk_ids(requested_ids, adjacent_window)
            if adjacent_window > 0
            else requested_ids
        )
        thread_id = _extract_thread_id(config)
        request_id = _extract_request_id(config)
        results = [
            self._read_single_chunk(
                doc_id=doc_id,
                thread_id=thread_id,
                request_id=request_id,
            )
            for doc_id in resolved_ids
        ]
        return {
            "requested_ids": requested_ids,
            "resolved_ids": resolved_ids,
            "adjacent_window": adjacent_window,
            "results": results,
        }

    def _read_single_chunk(
        self,
        *,
        doc_id: str,
        thread_id: str | None,
        request_id: str | None,
    ) -> dict[str, Any]:
        if thread_id and self.workspace_root:
            self._cleanup_expired_chunk_files(thread_id=thread_id)

        workspace_path = (
            _build_chunk_workspace_path(thread_id, doc_id)
            if thread_id and self.workspace_root
            else None
        )
        if self.store is not None and thread_id and request_id:
            namespace = _turn_read_store_namespace(thread_id, request_id)
            existing = self.store.get(namespace, doc_id)
            if existing is not None:
                out: dict[str, Any] = {
                    "id": doc_id,
                    "already_read": True,
                    "notice": "This chunk has been read before",
                }
                value = existing.value if hasattr(existing, "value") else None
                if isinstance(value, dict):
                    path = value.get("path")
                    if isinstance(path, str) and path.strip():
                        out["path"] = path
                if isinstance(
                    out.get("path"), str
                ) and not self._is_workspace_path_available(str(out["path"])):
                    # File may have expired by TTL cleanup. Reset duplicate marker and re-fetch.
                    self.store.delete(namespace, doc_id)
                else:
                    if "path" not in out and isinstance(workspace_path, str):
                        out["path"] = workspace_path
                    self._record_turn_read(
                        doc_id=doc_id,
                        thread_id=thread_id,
                        request_id=request_id,
                        path=out.get("path")
                        if isinstance(out.get("path"), str)
                        else None,
                    )
                    return out
                if "path" not in out and isinstance(workspace_path, str):
                    out["path"] = workspace_path

        safe_id = doc_id.replace("'", "''")
        payload = search(
            config=self.config,
            index_name=self.index_name,
            query="*",
            top=1,
            filter=f"id eq '{safe_id}'",
            select=(
                "id,source_file,type,page_number,title,content,node_id,"
                "process,owner,start_at,image_descriptions,"
                "table_descriptions,original_caption,row_count,column_count,page_blob_url,"
                "figure_blob_urls,table_blob_urls"
            ),
        )
        rows = payload.get("value", [])
        if not isinstance(rows, list) or not rows:
            return {
                "id": doc_id,
                "already_read": False,
                "notice": "Chunk not found",
            }

        row = rows[0] if isinstance(rows[0], dict) else {}
        read_result = _chunk_row_to_read_result(row)
        persisted_path = self._persist_chunk_result_file(
            thread_id=thread_id, read_result=read_result
        )
        if persisted_path:
            read_result["path"] = persisted_path
        if self.path_only and persisted_path:
            # Force file-mediated context usage: callers must read the stored file.
            read_result.pop("content", None)
            read_result["notice"] = (
                "Open the stored file path with read_file for full original content"
            )
        self._record_turn_read(
            doc_id=doc_id,
            thread_id=thread_id,
            request_id=request_id,
            path=persisted_path,
        )
        return read_result

    def _record_turn_read(
        self,
        *,
        doc_id: str,
        thread_id: str | None,
        request_id: str | None,
        path: str | None,
    ) -> None:
        if self.store is None or not thread_id or not request_id:
            return
        self.store.put(
            _turn_read_store_namespace(thread_id, request_id),
            doc_id,
            {
                "id": doc_id,
                "read_at": datetime.now(timezone.utc).isoformat(),
                "path": path or "",
            },
        )

    def _cleanup_expired_chunk_files(self, *, thread_id: str) -> None:
        ttl = self.chunk_file_ttl_seconds
        if ttl is None or ttl <= 0 or not self.workspace_root:
            return
        safe_thread = _sanitize_path_component(thread_id)
        thread_chunks_workspace = f"/workspace/threads/{safe_thread}/chunks"
        thread_chunks_real = _resolve_workspace_real_path(
            self.workspace_root, thread_chunks_workspace
        )
        if thread_chunks_real is None or not thread_chunks_real.exists():
            return
        cutoff_ts = datetime.now(timezone.utc).timestamp() - ttl
        for path in thread_chunks_real.glob("*.json"):
            if not path.is_file():
                continue
            if _chunk_file_is_expired(path, cutoff_ts):
                try:
                    path.unlink()
                except Exception:
                    continue

    def _is_workspace_path_available(self, workspace_path: str) -> bool:
        if not self.workspace_root:
            return False
        real_path = _resolve_workspace_real_path(self.workspace_root, workspace_path)
        return real_path is not None and real_path.exists()

    def _persist_chunk_result_file(
        self, *, thread_id: str | None, read_result: dict[str, Any]
    ) -> str | None:
        if not thread_id or not self.workspace_root:
            return None
        doc_id = str(read_result.get("id") or "").strip()
        if not doc_id:
            return None
        workspace_path = _build_chunk_workspace_path(thread_id, doc_id)
        real_path = _resolve_workspace_real_path(self.workspace_root, workspace_path)
        if real_path is None:
            return None

        payload = {
            "id": doc_id,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "content": read_result.get("content"),
            "metadata": read_result.get("metadata"),
        }
        try:
            real_path.parent.mkdir(parents=True, exist_ok=True)
            real_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return workspace_path
        except Exception:
            return None


def make_chunk_read(
    config: AzureSearchConfig,
    *,
    index_name: str,
    store: BaseStore | None = None,
    workspace_root: str | Path | None = None,
    path_only: bool = False,
    chunk_file_ttl_seconds: int | None = None,
) -> ChunkReadTool:
    """Factory for chunk_read tool."""
    root_value = str(workspace_root) if workspace_root is not None else None
    ttl_seconds = (
        chunk_file_ttl_seconds
        if chunk_file_ttl_seconds is not None
        else DEFAULT_CHUNK_FILE_TTL_SECONDS
    )
    return ChunkReadTool(
        config=config,
        index_name=index_name,
        store=store,
        workspace_root=root_value,
        path_only=path_only,
        chunk_file_ttl_seconds=ttl_seconds,
    )


class CiteSourcesTool(BaseTool):
    """Record explicit citation sources for the current turn."""

    name: str = "cite_sources"
    description: str = (
        "cite_sources (인용 근거 지정)\n"
        "주 사용처: 최종 답변 직전, 실제 사용한 근거 ID를 명시적으로 확정.\n"
        "입력 JSON:\n"
        '{ "ids": ["string"], "reason": "optional string" }\n'
        "출력 JSON 객체:\n"
        '{ "success": true|false, "ids": ["string"], "reason": "optional", "notice": "optional" }\n'
        "운영 규칙:\n"
        "- 현재 턴(thread_id + request_id)에서 chunk_read로 읽은 ID만 지정 가능.\n"
        "- type이 figure_image/table_image인 경우 부분 인용이 아닌 객체 전체 인용으로 취급.\n"
        "- 미충족 시 success=false와 notice 반환.\n"
        "- 사용자에게 제시한 핵심 결론과 직접 연결된 최소 근거만 선택할 것.\n"
        "- 지정한 ids/reason은 내부 인용 선택용이며, 사용자 본문에 원문 식별자나 내부 구조를 그대로 출력하지 말 것."
    )
    args_schema: Type[BaseModel] = CiteSourcesInput

    store: BaseStore | None = None

    def _run(
        self,
        ids: list[str],
        reason: str | None = None,
        config: RunnableConfig = None,
    ) -> dict[str, Any]:
        normalized_ids = _normalize_ids(ids)
        if not normalized_ids:
            return {"success": False, "notice": "ids is required", "requested_ids": []}
        thread_id = _extract_thread_id(config)
        request_id = _extract_request_id(config)
        if self.store is None or not thread_id or not request_id:
            return {
                "success": False,
                "notice": "thread_id and request_id are required",
                "requested_ids": normalized_ids,
            }
        missing_ids = [
            doc_id
            for doc_id in normalized_ids
            if self.store.get(_turn_read_store_namespace(thread_id, request_id), doc_id)
            is None
        ]
        if missing_ids:
            return {
                "success": False,
                "notice": "Source must be read in this turn before citation",
                "requested_ids": normalized_ids,
                "missing_ids": missing_ids,
            }

        normalized_reason = (
            reason.strip() if isinstance(reason, str) and reason.strip() else ""
        )
        payload = {
            "ids": normalized_ids,
            "reason": normalized_reason,
            "cited_at": datetime.now(timezone.utc).isoformat(),
        }
        self.store.put(
            _citation_choice_store_namespace(thread_id, request_id),
            "citations",
            payload,
        )
        out: dict[str, Any] = {"success": True, "ids": normalized_ids}
        if normalized_reason:
            out["reason"] = normalized_reason
        out["notice"] = "Citations recorded"
        return out


def make_cite_sources(*, store: BaseStore | None = None) -> CiteSourcesTool:
    """Factory for cite_sources tool."""
    return CiteSourcesTool(store=store)


def fetch_research_paper_index_schema(
    config: AzureSearchConfig, *, index_name: str
) -> dict[str, Any]:
    """Fetch index schema for research paper documents."""
    return get_index_schema(config, index_name=index_name)


def validate_research_paper_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Validate whether index fields cover expected research paper schema."""
    fields = schema.get("fields", []) if isinstance(schema, dict) else []
    actual_field_names = [
        f.get("name") for f in fields if isinstance(f, dict) and f.get("name")
    ]
    actual_set = set(actual_field_names)
    expected_set = set(EXPECTED_RESEARCH_PAPER_FIELDS)

    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    return {
        "expected_field_count": len(EXPECTED_RESEARCH_PAPER_FIELDS),
        "actual_field_count": len(actual_field_names),
        "missing_fields": missing,
        "extra_fields": extra,
        "is_compatible": len(missing) == 0,
        "actual_fields": actual_field_names,
    }


def extract_search_field_capabilities(schema: dict[str, Any]) -> dict[str, Any]:
    """Extract user-facing field capabilities from Azure index schema."""
    fields = schema.get("fields", []) if isinstance(schema, dict) else []
    filterable_fields: list[str] = []
    facetable_fields: list[str] = []
    searchable_fields: list[str] = []
    sortable_fields: list[str] = []

    for field in fields:
        if not isinstance(field, dict):
            continue
        name = field.get("name")
        if not isinstance(name, str) or not name:
            continue
        if field.get("filterable") is True:
            filterable_fields.append(name)
        if field.get("facetable") is True:
            facetable_fields.append(name)
        if field.get("searchable") is True:
            searchable_fields.append(name)
        if field.get("sortable") is True:
            sortable_fields.append(name)

    filter_examples: list[str] = []
    if "process" in filterable_fields:
        filter_examples.append("process eq '논문'")
    if "page_number" in filterable_fields:
        filter_examples.append("page_number ge 1 and page_number le 20")
    if "start_at" in filterable_fields:
        filter_examples.append("start_at ge '2023' and start_at le '2025'")
    if "type" in filterable_fields:
        filter_examples.append("type ne 'node'")

    return {
        "filterable_fields": sorted(filterable_fields),
        "facetable_fields": sorted(facetable_fields),
        "searchable_fields": sorted(searchable_fields),
        "sortable_fields": sorted(sortable_fields),
        "filter_examples": filter_examples,
    }


def run_keyword_search(
    *,
    config: AzureSearchConfig,
    index_name: str,
    query: str,
    top_k: int = 5,
    filter: str | None = None,
) -> list[dict[str, Any]]:
    effective_filter = _merge_with_base_filter(filter)

    payload = search(
        config=config,
        index_name=index_name,
        query=query,
        top=top_k,
        filter=effective_filter,
        select="id,source_file,title,content,image_descriptions,table_descriptions,original_caption",
        highlight_fields=list(KEYWORD_TEXT_FIELDS),
        highlight_pre_tag="<em>",
        highlight_post_tag="</em>",
    )
    return _rows_to_chunk_snippets(payload.get("value", []))


def run_semantic_hybrid_search(
    *,
    config: AzureSearchConfig,
    index_name: str,
    embed_query: Callable[[str], list[float]],
    query: str,
    top_k: int = 5,
    filter: str | None = None,
) -> list[dict[str, Any]]:
    effective_filter = _merge_with_base_filter(filter)
    vector = embed_query(query)
    payload = search(
        config=config,
        index_name=index_name,
        query=query,
        top=top_k,
        filter=effective_filter,
        select="id,source_file,title,content,image_descriptions,table_descriptions,original_caption",
        highlight_fields=list(KEYWORD_TEXT_FIELDS),
        highlight_pre_tag="<em>",
        highlight_post_tag="</em>",
        vector_queries=[
            {
                "kind": "vector",
                "fields": "embedding",
                "vector": vector,
                "k": top_k,
            }
        ],
    )
    return _rows_to_chunk_snippets(payload.get("value", []))


def run_contextual_search(
    *,
    config: AzureSearchConfig,
    index_name: str,
    embed_query: Callable[[str], list[float]],
    query: str,
    top_k: int = 5,
    filter: str | None = None,
) -> list[dict[str, Any]]:
    allowlist_filter = (
        "(type eq 'page_chunk' or type eq 'figure_image' or type eq 'table_image')"
    )
    effective_filter = allowlist_filter
    if isinstance(filter, str) and filter.strip():
        effective_filter = f"({allowlist_filter}) and ({filter.strip()})"

    candidate_k = max(top_k * 4, top_k)
    vector = embed_query(query)

    search_kwargs = {
        "config": config,
        "index_name": index_name,
        "query": query,
        "top": candidate_k,
        "filter": effective_filter,
        "search_fields": [
            "title",
            "content",
            "image_descriptions",
            "table_descriptions",
            "original_caption",
        ],
        "select": (
            "id,source_file,type,page_number,title,content,image_descriptions,"
            "table_descriptions,original_caption"
        ),
        "highlight_fields": list(KEYWORD_TEXT_FIELDS),
        "highlight_pre_tag": "<em>",
        "highlight_post_tag": "</em>",
        "vector_queries": [
            {
                "kind": "vector",
                "fields": "embedding",
                "vector": vector,
                "k": candidate_k,
            }
        ],
    }

    combined = list(search(**search_kwargs).get("value", []))

    figure_table_docs = [
        d for d in combined if d.get("type") in {"figure_image", "table_image"}
    ]
    parent_score_map: dict[tuple[Any, Any], float] = {}
    if figure_table_docs:
        figure_table_docs.sort(
            key=lambda d: float(d.get("@search.score", 0) or 0), reverse=True
        )
        top_count = max(1, -(-len(figure_table_docs) // 5))  # ceil(20%)
        for doc in figure_table_docs[:top_count]:
            key = (doc.get("source_file"), doc.get("page_number"))
            if not key[0] or key[1] is None:
                continue
            score = float(doc.get("@search.score", 0) or 0)
            parent_score_map[key] = max(parent_score_map.get(key, 0), score)

    deduped: list[dict[str, Any]] = []
    existing_parent_keys: set[tuple[Any, Any]] = set()
    for doc in combined:
        doc_type = doc.get("type")
        key = (doc.get("source_file"), doc.get("page_number"))
        if doc_type in {"figure_image", "table_image"} and key in parent_score_map:
            continue
        if doc_type == "page_chunk" and key in parent_score_map:
            doc["@search.score"] = parent_score_map[key]
            existing_parent_keys.add(key)
        deduped.append(doc)

    missing_parent_keys = set(parent_score_map.keys()) - existing_parent_keys
    if missing_parent_keys:
        filter_parts: list[str] = []
        for source_file, page_number in missing_parent_keys:
            if not source_file or page_number is None:
                continue
            safe_source = str(source_file).replace("'", "''")
            try:
                page_num_int = int(page_number)
            except (TypeError, ValueError):
                continue
            filter_parts.append(
                f"(type eq 'page_chunk' and source_file eq '{safe_source}' and page_number eq {page_num_int})"
            )
        if filter_parts:
            parent_results = search(
                config=config,
                index_name=index_name,
                query="*",
                filter=" or ".join(filter_parts),
                select=(
                    "id,source_file,type,page_number,title,content,image_descriptions,"
                    "table_descriptions,original_caption"
                ),
                top=len(filter_parts),
            ).get("value", [])
            for doc in parent_results:
                key = (doc.get("source_file"), doc.get("page_number"))
                if key in parent_score_map:
                    doc["@search.score"] = parent_score_map[key]
                deduped.append(doc)

    seen_ids: set[str] = set()
    unique_docs: list[dict[str, Any]] = []
    for doc in deduped:
        doc_id = doc.get("id")
        if isinstance(doc_id, str) and doc_id:
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
        unique_docs.append(doc)

    unique_docs.sort(key=lambda d: float(d.get("@search.score", 0) or 0), reverse=True)
    final_docs = unique_docs[:top_k] if top_k > 0 else []
    return _rows_to_chunk_snippets(final_docs)


def _merge_with_base_filter(filter_value: str | None) -> str:
    if isinstance(filter_value, str) and filter_value.strip():
        return f"({BASE_FILTER}) and ({filter_value.strip()})"
    return BASE_FILTER


def _rows_to_chunk_snippets(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []

    results: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        doc_id = str(row.get("id") or "")
        if not doc_id:
            continue

        snippet = _build_rich_snippet_from_highlights(row.get("@search.highlights"))
        if not snippet:
            for field in ("content", "title", "source_file"):
                value = row.get(field)
                if isinstance(value, str):
                    text = value.strip()
                    if text:
                        snippet = text[:240]
                        break

        results.append(
            _build_chunk_result_row(
                doc_id=doc_id,
                snippet=snippet,
            )
        )
    return results


def _build_chunk_result_row(
    *,
    doc_id: str,
    snippet: str,
) -> dict[str, Any]:
    return {
        "id": doc_id,
        "snippet": snippet,
    }


def _build_rich_snippet_from_highlights(highlights: Any) -> str:
    if not isinstance(highlights, dict):
        return ""

    parts: list[str] = []
    for field in KEYWORD_TEXT_FIELDS:
        values = highlights.get(field)
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str):
                text = value.strip()
                if text:
                    parts.append(text)

    if not parts:
        return ""

    merged = " | ".join(dict.fromkeys(parts))
    return merged[:1000]


def _coerce_str_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [v.strip() for v in values if isinstance(v, str) and v.strip()]


def _chunk_row_to_read_result(row: dict[str, Any]) -> dict[str, Any]:
    doc_id = str(row.get("id") or "").strip()
    content = str(row.get("content") or "").strip()

    if not content:
        desc_parts: list[str] = []
        for key in ("image_descriptions", "table_descriptions"):
            values = row.get(key)
            if isinstance(values, list):
                for v in values:
                    if isinstance(v, str) and v.strip():
                        desc_parts.append(v.strip())
        content = " ".join(desc_parts).strip()

    metadata = {
        "source_file": row.get("source_file"),
        "type": row.get("type"),
        "page_number": row.get("page_number"),
        "title": row.get("title"),
        "node_id": row.get("node_id"),
        "process": row.get("process"),
        "owner": row.get("owner"),
        "start_at": row.get("start_at"),
        "page_blob_url": row.get("page_blob_url"),
        "figure_blob_urls": row.get("figure_blob_urls"),
        "table_blob_urls": row.get("table_blob_urls"),
        "original_caption": row.get("original_caption"),
        "row_count": row.get("row_count"),
        "column_count": row.get("column_count"),
    }
    return {
        "id": doc_id,
        "already_read": False,
        "content": content,
        "metadata": metadata,
    }


def _extract_thread_id(config: RunnableConfig | None) -> str | None:
    if not isinstance(config, dict):
        return None
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return None
    thread_id = configurable.get("thread_id")
    if isinstance(thread_id, str) and thread_id.strip():
        return thread_id.strip()
    return None


def _extract_request_id(config: RunnableConfig | None) -> str | None:
    if not isinstance(config, dict):
        return None
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return None
    request_id = configurable.get("request_id")
    if isinstance(request_id, str) and request_id.strip():
        return request_id.strip()
    return None


def _turn_read_store_namespace(thread_id: str, request_id: str) -> tuple[str, ...]:
    return ("research_paper_summary", "turn_read", thread_id, request_id)


def _citation_choice_store_namespace(
    thread_id: str, request_id: str
) -> tuple[str, ...]:
    return ("research_paper_summary", "citation_choice", thread_id, request_id)


def _build_chunk_workspace_path(thread_id: str, doc_id: str) -> str:
    safe_thread = _sanitize_path_component(thread_id)
    digest = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:32]
    return f"/workspace/threads/{safe_thread}/chunks/{digest}.json"


def _sanitize_path_component(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]", "_", value.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        cleaned = "default"
    return cleaned[:80]


def _resolve_workspace_real_path(
    workspace_root: str | Path, workspace_path: str
) -> Path | None:
    if not workspace_path.startswith("/workspace/"):
        return None
    root = Path(workspace_root).resolve()
    relative = workspace_path[len("/workspace/") :]
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def _chunk_file_is_expired(path: Path, cutoff_ts: float) -> bool:
    candidate_ts = path.stat().st_mtime
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            saved_at = raw.get("saved_at")
            if isinstance(saved_at, str) and saved_at.strip():
                parsed = datetime.fromisoformat(saved_at)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                candidate_ts = parsed.astimezone(timezone.utc).timestamp()
    except Exception:
        pass
    return candidate_ts < cutoff_ts


def _normalize_ids(ids: list[str] | None) -> list[str]:
    out: list[str] = []
    if isinstance(ids, list):
        for item in ids:
            if isinstance(item, str):
                v = item.strip()
                if v:
                    out.append(v)
    return list(dict.fromkeys(out))


def _expand_adjacent_chunk_ids(base_ids: list[str], adjacent_window: int) -> list[str]:
    expanded: list[str] = list(base_ids)
    for doc_id in base_ids:
        match = re.match(
            r"^(?P<prefix>.*?)(?P<kind>page|table|figure)_(?P<num>\d+)$", doc_id
        )
        if not match:
            continue
        prefix = match.group("prefix")
        kind = match.group("kind")
        raw_num = match.group("num")
        width = len(raw_num)
        num = int(raw_num)
        for delta in range(1, adjacent_window + 1):
            left = num - delta
            right = num + delta
            if left >= 0:
                expanded.append(f"{prefix}{kind}_{left:0{width}d}")
            expanded.append(f"{prefix}{kind}_{right:0{width}d}")
    return list(dict.fromkeys(expanded))
