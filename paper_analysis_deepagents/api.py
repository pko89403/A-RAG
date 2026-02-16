from __future__ import annotations
from dataclasses import asdict
import os
from pathlib import Path
import re
from typing import Any, Callable
import logging

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import OpenAIEmbeddings
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from paper_analysis_deepagents.config import Settings, load_settings
from paper_analysis_deepagents.research_paper_summary_agent import (
    build_research_paper_summary_agent,
)
from paper_analysis_deepagents.llm import build_llm
from paper_analysis_deepagents.tools.azure_search import (
    AzureSearchConfig,
    AzureSearchError,
)
from paper_analysis_deepagents.tools.research_paper_search import (
    EXPECTED_RESEARCH_PAPER_FIELDS,
    extract_search_field_capabilities,
    fetch_research_paper_index_schema,
    make_chunk_read,
    run_contextual_search,
    run_keyword_search,
    run_semantic_hybrid_search,
    validate_research_paper_schema,
)
from paper_analysis_deepagents.history import HistoryService
from paper_analysis_deepagents.history.service import (
    ConversationNotFoundError,
    MaxTurnsExceededError,
)

try:
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
except Exception:  # pragma: no cover - optional dependency
    LangfuseCallbackHandler = None

logger = logging.getLogger(__name__)
DEFAULT_AGENT_RECURSION_LIMIT = 160
MAX_AUTO_CITATIONS = 5
REFERENCE_MARKER_RE = re.compile(
    r"<\s*ref\s*(\d+)\s*>(?:\s*<\s*page\s*:\s*(\d+)\s*>)?",
    flags=re.IGNORECASE,
)

MAIN_INVOKE_REQUEST_EXAMPLES = {
    "basic": {
        "summary": "메인 에이전트 기본 질의",
        "value": {
            "query": "연구 논문 분석해줘",
            "conversationId": "conv-main-001",
        },
    },
    "with_configurable": {
        "summary": "메인 context_schema 전체 필드 예시",
        "value": {
            "query": "최근 LLM 연구 동향 요약해줘",
            "conversationId": "conv-main-002",
            "configurable": {
                "threadId": "conv-main-002",
                "userId": "researcher-01",
                "department": "AI연구소",
                "reportLanguage": "ko",
                "reportFormat": "markdown",
                "maxSearchResults": 10,
                "searchDateRangeYears": 3,
                "analysisDepth": "deep",
                "recursionLimit": 60,
            },
        },
    },
}

RESEARCH_PAPER_INVOKE_REQUEST_EXAMPLES = {
    "basic": {
        "summary": "논문 요약 에이전트 기본 질의",
        "value": {
            "query": "Transformer 논문의 핵심 기여가 뭐야?",
            "conversationId": "conv-paper-001",
        },
    },
    "with_configurable": {
        "summary": "논문 요약 context_schema 필드 예시",
        "value": {
            "query": "RAG 관련 논문의 한계점 분석해줘",
            "conversationId": "conv-paper-002",
            "configurable": {
                "threadId": "conv-paper-002",
                "sessionScope": "research-paper-summary",
                "requestId": "req-20260216-001",
                "recursionLimit": 160,
            },
        },
    },
}

KEYWORD_SEARCH_REQUEST_EXAMPLES = {
    "basic": {
        "summary": "키워드 검색",
        "value": {
            "query": "Transformer attention mechanism",
            "topK": 5,
            "filter": "process eq '논문'",
        },
    }
}

SEMANTIC_HYBRID_SEARCH_REQUEST_EXAMPLES = {
    "basic": {
        "summary": "의미 하이브리드 검색",
        "value": {
            "query": "LLM 환각 현상 원인",
            "topK": 5,
            "filter": "start_at ge '2023'",
        },
    }
}

CONTEXTUAL_SEARCH_REQUEST_EXAMPLES = {
    "basic": {
        "summary": "맥락 중심 검색",
        "value": {
            "query": "RAG 파이프라인 구성",
            "topK": 5,
            "filter": "process eq '논문'",
        },
    }
}

CHUNK_READ_REQUEST_EXAMPLES = {
    "basic": {
        "summary": "청크 ID로 본문/메타 읽기",
        "value": {
            "ids": [
                "56c5eb027b2d7a454b9f29ed94e727be7f5b5914ce3e4f06ad2c892912041ec4-page_1"
            ],
            "adjacentWindow": 0,
            "conversationId": "conv-paper-001",
        },
    },
    "with_adjacent": {
        "summary": "인접 청크 함께 읽기",
        "value": {
            "ids": [
                "56c5eb027b2d7a454b9f29ed94e727be7f5b5914ce3e4f06ad2c892912041ec4-page_1"
            ],
            "adjacentWindow": 1,
            "threadId": "conv-paper-001",
            "requestId": "req-20260216-001",
        },
    },
}


class ResearchPaperInvokeConfigurable(BaseModel):
    thread_id: str | None = Field(default=None, alias="threadId")
    session_scope: str | None = Field(default=None, alias="sessionScope")
    request_id: str | None = Field(default=None, alias="requestId")
    recursion_limit: int | None = Field(default=None, alias="recursionLimit")

    class Config:
        populate_by_name = True
        extra = "allow"


class ResearchPaperInvokeRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, description="User query for the research paper summary agent"
    )
    conversation_id: str | None = Field(
        default=None,
        alias="conversationId",
        description="Conversation ID for multi-turn context",
    )
    configurable: ResearchPaperInvokeConfigurable | None = Field(
        default=None,
        description=(
            "Optional runtime context for research paper summary context_schema. "
            "Supports thread_id, session_scope, request_id."
        ),
    )

    class Config:
        populate_by_name = True


class InvokeResponse(BaseModel):
    content: str
    messages: list[dict[str, Any]] | None = None
    references: list[dict[str, Any]] | None = None
    citations: list[dict[str, Any]] | None = None

    class Config:
        populate_by_name = True


# History API Models
class CreateConversationRequest(BaseModel):
    conversation_id: str = Field(..., alias="conversationId")
    title: str = ""
    initial_message: dict[str, Any] | None = Field(default=None, alias="initialMessage")

    class Config:
        populate_by_name = True


class AddMessageRequest(BaseModel):
    message: dict[str, Any]


class ConversationResponse(BaseModel):
    id: str
    user_id: str = Field(alias="userId")
    title: str
    created_at: str = Field(alias="createdAt")
    updated_at: str = Field(alias="updatedAt")
    turn_count: int = Field(alias="turnCount")
    messages: list[dict[str, Any]]

    class Config:
        populate_by_name = True


class ConversationListResponse(BaseModel):
    conversations: list[dict[str, Any]]


class ResearchPaperSchemaCheckResponse(BaseModel):
    index_name: str
    api_version: str
    is_compatible: bool
    expected_field_count: int
    actual_field_count: int
    missing_fields: list[str]
    extra_fields: list[str]
    expected_fields: list[str]
    filterable_fields: list[str]
    facetable_fields: list[str]
    searchable_fields: list[str]
    sortable_fields: list[str]
    filter_examples: list[str]


class ResearchPaperKeywordSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["Transformer attention mechanism"])
    top_k: int = Field(default=5, ge=1, le=50, alias="topK")
    filter: str | None = Field(
        default=None,
        description="선택적 OData 필터. 미사용 시 생략/null 사용. 'string' 리터럴 금지.",
        examples=["process eq '논문' and start_at ge '2023'"],
    )

    class Config:
        populate_by_name = True


class ResearchPaperChunkResult(BaseModel):
    id: str
    snippet: str


class ResearchPaperSemanticHybridSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["LLM 환각 현상 원인"])
    top_k: int = Field(default=5, ge=1, le=50, alias="topK")
    filter: str | None = Field(
        default=None,
        description="선택적 OData 필터. 미사용 시 생략/null 사용. 'string' 리터럴 금지.",
        examples=["start_at ge '2023'"],
    )

    class Config:
        populate_by_name = True


class ResearchPaperContextualSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["RAG 파이프라인 구성"])
    top_k: int = Field(default=5, ge=1, le=50, alias="topK")
    filter: str | None = Field(
        default=None,
        description="선택적 OData 필터. 미사용 시 생략/null 사용. 'string' 리터럴 금지.",
        examples=["process eq '논문'"],
    )

    class Config:
        populate_by_name = True


class ResearchPaperChunkReadRequest(BaseModel):
    ids: list[str] = Field(
        ...,
        min_length=1,
        description="읽을 문서 ID 목록",
        examples=[
            ["56c5eb027b2d7a454b9f29ed94e727be7f5b5914ce3e4f06ad2c892912041ec4-page_1"]
        ],
    )
    adjacent_window: int = Field(
        default=0,
        ge=0,
        le=2,
        alias="adjacentWindow",
        description="인접 청크 확장 폭. 0이면 확장 없음, 1이면 ±1 청크 추가",
    )
    conversation_id: str | None = Field(
        default=None,
        alias="conversationId",
        description="thread_id로 사용될 대화 식별자(선택)",
    )
    thread_id: str | None = Field(
        default=None,
        alias="threadId",
        description="conversationId가 없을 때 사용할 thread_id(선택)",
    )
    request_id: str | None = Field(
        default=None,
        alias="requestId",
        description="중복 읽기 판정 범위를 제어할 request_id(선택)",
    )

    class Config:
        populate_by_name = True


class ResearchPaperChunkReadResponse(BaseModel):
    requested_ids: list[str] = Field(default_factory=list)
    resolved_ids: list[str] = Field(default_factory=list)
    adjacent_window: int = 0
    results: list[dict[str, Any]] = Field(default_factory=list)
    notice: str | None = None


def create_app() -> FastAPI:
    """Create a FastAPI app that exposes the DeepAgents runner over HTTP."""
    app = FastAPI(title="paper-analysis-agent API", version="0.1.0")

    cors = (os.environ.get("DEEPAGENTS_API_CORS_ORIGINS") or "").strip()
    if cors:
        origins = [o.strip() for o in cors.split(",") if o.strip()]
        allow_credentials = "*" not in origins
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/config")
    def config() -> dict[str, object]:
        # Redacted settings for debugging (do not expose secrets).
        return _redact_settings(_get_settings())

    @app.get(
        "/tools/research-paper-summary/index-schema",
        response_model=ResearchPaperSchemaCheckResponse,
    )
    def research_paper_index_schema() -> ResearchPaperSchemaCheckResponse:
        settings = _get_settings()
        if not settings.azure_search_endpoint or not settings.azure_search_api_key:
            raise HTTPException(
                status_code=503,
                detail="AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY must be configured",
            )

        config = AzureSearchConfig(
            endpoint=settings.azure_search_endpoint,
            api_key=settings.azure_search_api_key,
            api_version=settings.azure_search_api_version,
        )
        try:
            schema = fetch_research_paper_index_schema(
                config,
                index_name=settings.research_paper_index_name,
            )
            result = validate_research_paper_schema(schema)
            capabilities = extract_search_field_capabilities(schema)
            return ResearchPaperSchemaCheckResponse(
                index_name=settings.research_paper_index_name,
                api_version=settings.azure_search_api_version,
                expected_fields=list(EXPECTED_RESEARCH_PAPER_FIELDS),
                **result,
                **capabilities,
            )
        except AzureSearchError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e

    @app.post(
        "/tools/research-paper-summary/keyword-search",
        response_model=list[ResearchPaperChunkResult],
        response_model_exclude_none=True,
    )
    def research_paper_keyword_search(
        req: ResearchPaperKeywordSearchRequest = Body(
            ..., openapi_examples=KEYWORD_SEARCH_REQUEST_EXAMPLES
        ),
    ) -> list[ResearchPaperChunkResult]:
        settings = _get_settings()
        if not settings.azure_search_endpoint or not settings.azure_search_api_key:
            raise HTTPException(
                status_code=503,
                detail="AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY must be configured",
            )

        config = AzureSearchConfig(
            endpoint=settings.azure_search_endpoint,
            api_key=settings.azure_search_api_key,
            api_version=settings.azure_search_api_version,
        )
        try:
            rows = run_keyword_search(
                config=config,
                index_name=settings.research_paper_index_name,
                query=req.query,
                top_k=req.top_k,
                filter=_normalize_filter(req.filter),
            )
            if not isinstance(rows, list):
                raise HTTPException(
                    status_code=502, detail="Unexpected keyword_search response shape"
                )
            return [
                _to_chunk_result_model(row)
                for row in rows
                if isinstance(row, dict) and row.get("id")
            ]
        except AzureSearchError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e

    @app.post(
        "/tools/research-paper-summary/semantic-hybrid-search",
        response_model=list[ResearchPaperChunkResult],
        response_model_exclude_none=True,
    )
    def research_paper_semantic_hybrid_search(
        req: ResearchPaperSemanticHybridSearchRequest = Body(
            ..., openapi_examples=SEMANTIC_HYBRID_SEARCH_REQUEST_EXAMPLES
        ),
    ) -> list[ResearchPaperChunkResult]:
        settings = _get_settings()
        if not settings.azure_search_endpoint or not settings.azure_search_api_key:
            raise HTTPException(
                status_code=503,
                detail="AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY must be configured",
            )

        config = AzureSearchConfig(
            endpoint=settings.azure_search_endpoint,
            api_key=settings.azure_search_api_key,
            api_version=settings.azure_search_api_version,
        )
        try:
            embed_query = _resolve_embed_query(settings)
            rows = run_semantic_hybrid_search(
                config=config,
                index_name=settings.research_paper_index_name,
                embed_query=embed_query,
                query=req.query,
                top_k=req.top_k,
                filter=_normalize_filter(req.filter),
            )
            if not isinstance(rows, list):
                raise HTTPException(
                    status_code=502,
                    detail="Unexpected semantic_hybrid_search response shape",
                )
            return [
                _to_chunk_result_model(row)
                for row in rows
                if isinstance(row, dict) and row.get("id")
            ]
        except HTTPException:
            raise
        except AzureSearchError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=(
                    "semantic_hybrid_search failed. "
                    "Check EMBEDDING_ENDPOINT(/openai/v1), EMBEDDING_MODELNAME/DEPLOYMENT, and Azure deployment availability. "
                    f"Raw error: {e}"
                ),
            ) from e

    @app.post(
        "/tools/research-paper-summary/contextual-search",
        response_model=list[ResearchPaperChunkResult],
        response_model_exclude_none=True,
    )
    def research_paper_contextual_search(
        req: ResearchPaperContextualSearchRequest = Body(
            ..., openapi_examples=CONTEXTUAL_SEARCH_REQUEST_EXAMPLES
        ),
    ) -> list[ResearchPaperChunkResult]:
        settings = _get_settings()
        if not settings.azure_search_endpoint or not settings.azure_search_api_key:
            raise HTTPException(
                status_code=503,
                detail="AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY must be configured",
            )

        config = AzureSearchConfig(
            endpoint=settings.azure_search_endpoint,
            api_key=settings.azure_search_api_key,
            api_version=settings.azure_search_api_version,
        )
        try:
            embed_query = _resolve_embed_query(settings)
            rows = run_contextual_search(
                config=config,
                index_name=settings.research_paper_index_name,
                embed_query=embed_query,
                query=req.query,
                top_k=req.top_k,
                filter=_normalize_filter(req.filter),
            )
            if not isinstance(rows, list):
                raise HTTPException(
                    status_code=502,
                    detail="Unexpected contextual_search response shape",
                )
            return [
                _to_chunk_result_model(row)
                for row in rows
                if isinstance(row, dict) and row.get("id")
            ]
        except HTTPException:
            raise
        except AzureSearchError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"contextual_search failed. Raw error: {e}",
            ) from e

    @app.post(
        "/tools/research-paper-summary/chunk-read",
        response_model=ResearchPaperChunkReadResponse,
    )
    def research_paper_chunk_read(
        req: ResearchPaperChunkReadRequest = Body(
            ..., openapi_examples=CHUNK_READ_REQUEST_EXAMPLES
        ),
    ) -> ResearchPaperChunkReadResponse:
        settings = _get_settings()
        if not settings.azure_search_endpoint or not settings.azure_search_api_key:
            raise HTTPException(
                status_code=503,
                detail="AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY must be configured",
            )

        config = AzureSearchConfig(
            endpoint=settings.azure_search_endpoint,
            api_key=settings.azure_search_api_key,
            api_version=settings.azure_search_api_version,
        )
        try:
            tool = make_chunk_read(
                config=config,
                index_name=settings.research_paper_index_name,
                store=_get_chunk_read_store(),
                workspace_root=Path(__file__).resolve().parents[1],
            )
            thread_id = (req.conversation_id or req.thread_id or "").strip() or None
            request_id = (req.request_id or "").strip() or None
            configurable: dict[str, Any] = {}
            if thread_id:
                configurable["thread_id"] = thread_id
            if request_id:
                configurable["request_id"] = request_id
            invoke_config = {"configurable": configurable} if configurable else None
            out = tool.invoke(
                {
                    "ids": req.ids,
                    "adjacent_window": req.adjacent_window,
                },
                config=invoke_config,
            )
            if not isinstance(out, dict):
                raise HTTPException(
                    status_code=502, detail="Unexpected chunk_read response shape"
                )

            return ResearchPaperChunkReadResponse(
                requested_ids=out.get("requested_ids")
                if isinstance(out.get("requested_ids"), list)
                else [],
                resolved_ids=out.get("resolved_ids")
                if isinstance(out.get("resolved_ids"), list)
                else [],
                adjacent_window=int(out.get("adjacent_window") or 0),
                results=out.get("results")
                if isinstance(out.get("results"), list)
                else [],
                notice=str(out.get("notice")) if out.get("notice") else None,
            )
        except AzureSearchError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"chunk_read failed. Raw error: {e}",
            ) from e

    @app.post("/agents/research-paper-summary/invoke", response_model=InvokeResponse)
    def invoke_research_paper_summary(req: ResearchPaperInvokeRequest):
        """Invoke the research paper summary agent directly using HTTP."""
        return _invoke_agent_sync(_get_research_paper_summary_agent(), req)

    @app.post("/chat/history")
    def create_conversation(req: CreateConversationRequest) -> ConversationResponse:
        service = _get_history_service()
        try:
            conv = service.create_conversation(
                conversation_id=req.conversation_id,
                title=req.title,
                initial_message=req.initial_message,
            )
            return ConversationResponse(
                id=conv.id,
                userId=conv.user_id,
                title=conv.title,
                createdAt=conv.created_at,
                updatedAt=conv.updated_at,
                turnCount=len(conv.turns),
                messages=_flatten_turns_to_messages(conv.turns),
            )
        except Exception as e:
            logger.exception("Failed to create conversation %s", req.conversation_id)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/chat/history/{conversation_id}/messages")
    def add_message(
        conversation_id: str, req: AddMessageRequest
    ) -> ConversationResponse:
        service = _get_history_service()
        try:
            conv = service.add_message(conversation_id, req.message)
            return ConversationResponse(
                id=conv.id,
                userId=conv.user_id,
                title=conv.title,
                createdAt=conv.created_at,
                updatedAt=conv.updated_at,
                turnCount=len(conv.turns),
                messages=_flatten_turns_to_messages(conv.turns),
            )
        except ConversationNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except MaxTurnsExceededError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.exception("Failed to add message to %s", conversation_id)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/chat/history/{conversation_id}")
    def get_conversation(conversation_id: str) -> ConversationResponse:
        service = _get_history_service()
        try:
            conv = service.get_conversation(conversation_id)
            return ConversationResponse(
                id=conv.id,
                userId=conv.user_id,
                title=conv.title,
                createdAt=conv.created_at,
                updatedAt=conv.updated_at,
                turnCount=len(conv.turns),
                messages=_flatten_turns_to_messages(conv.turns),
            )
        except ConversationNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except Exception as e:
            logger.exception("Failed to get conversation %s", conversation_id)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.delete("/chat/history/{conversation_id}")
    def delete_conversation(conversation_id: str):
        service = _get_history_service()
        try:
            service.delete_conversation(conversation_id)
            return {"status": "deleted", "id": conversation_id}
        except ConversationNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except Exception as e:
            logger.exception("Failed to delete conversation %s", conversation_id)
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


# ----------------------------------------------------------------------------
# Internal Helpers
# ----------------------------------------------------------------------------
def _invoke_agent_sync(agent_executor: Any, req: BaseModel) -> InvokeResponse:
    if not hasattr(agent_executor, "invoke"):
        raise HTTPException(status_code=500, detail="Invalid agent executor")

    # input/config mapping based on request type
    input_payload: dict[str, Any] = {}
    config_payload: dict[str, Any] = {}

    if hasattr(req, "query"):
        input_payload["messages"] = [HumanMessage(content=getattr(req, "query"))]

    req_configurable = getattr(req, "configurable", None)
    if req_configurable:
        conf_dict = req_configurable.model_dump(by_alias=True, exclude_none=True)
        config_payload["configurable"] = conf_dict

    try:
        if isinstance(req, ResearchPaperInvokeRequest) and req.conversation_id:
            config_payload.setdefault("configurable", {})["thread_id"] = (
                req.conversation_id
            )
            # Also use conversation_id as default persistence thread if not explicitly given
            if "thread_id" not in config_payload["configurable"]:
                config_payload["configurable"]["thread_id"] = req.conversation_id

        result = agent_executor.invoke(input_payload, config=config_payload)
        output_messages = result.get("messages", [])
        last_message = output_messages[-1] if output_messages else None
        content = last_message.content if isinstance(last_message, BaseMessage) else ""

        # References/Citations extraction from history/store
        refs: list[dict[str, Any]] = []
        citations: list[dict[str, Any]] = []
        # (Simplified: in a real app, you might parse artifacts or store contents)

        # Serialize messages for response
        serialized_messages = []
        for m in output_messages:
            if isinstance(m, (HumanMessage, AIMessage, ToolMessage)):
                serialized_messages.append(m.model_dump())

        return InvokeResponse(
            content=str(content),
            messages=serialized_messages,
            references=refs,
            citations=citations,
        )

    except Exception as e:
        logger.exception("Agent invocation failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


def _get_settings() -> Settings:
    return load_settings()


def _get_research_paper_summary_agent():
    settings = _get_settings()
    llm = build_llm(settings)
    return build_research_paper_summary_agent(settings, llm=llm)


def _get_chunk_read_store() -> InMemoryStore:
    # In a real app, this should be a persistent store or singleton.
    # For PoC, we return a new store or a cached one if possible.
    return InMemoryStore()


def _get_history_service() -> HistoryService:
    settings = _get_settings()
    data_dir = list(Path(__file__).parents)[1] / "data" / "history"
    data_dir.mkdir(parents=True, exist_ok=True)
    return HistoryService(
        root_dir=data_dir,
        max_turns=settings.history_max_turns,
    )


def _flatten_turns_to_messages(turns: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for turn in turns:
        # turn is likely dict or object from HistoryService
        # Adapt to specific needs
        pass
    return out


def _redact_settings(settings: Settings) -> dict[str, object]:
    data = asdict(settings)
    for key in data:
        if "key" in key.lower() or "token" in key.lower() or "secret" in key.lower():
            if data[key]:
                data[key] = "***"
    return data


def _resolve_embed_query(settings: Settings) -> Callable[[str], list[float]]:
    if not settings.embedding_endpoint or not settings.embedding_api_key:
        raise HTTPException(
            status_code=503,
            detail="EMBEDDING_ENDPOINT/API_KEY not configured",
        )
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_modelname,
        deployment=settings.embedding_deployment,
        openai_api_key=settings.embedding_api_key,
        openai_api_base=settings.embedding_endpoint,
        check_embedding_ctx_length=False,
    )
    return embeddings.embed_query


def _normalize_filter(f: str | None) -> str | None:
    if f and f.strip():
        return f.strip()
    return None


def _to_chunk_result_model(row: dict[str, Any]) -> ResearchPaperChunkResult:
    return ResearchPaperChunkResult(
        id=str(row.get("id") or ""),
        snippet=str(row.get("snippet") or ""),
    )
