"""LLM initialization utilities."""

from __future__ import annotations

import os
from typing import Callable
from urllib.parse import urlparse

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from paper_analysis_deepagents.config import Settings


def build_llm(settings: Settings) -> BaseChatModel:
    """Build the chat model used by the DeepAgent.

    Uses OpenAI-compatible endpoint via `ChatOpenAI(base_url=...)` (works with Azure OpenAI `/openai/v1` too)

    Args:
        settings: Runtime settings

    Returns:
        Initialized LLM instance

    Raises:
        ValueError: If no valid LLM configuration is found
    """
    if settings.openai_endpoint and settings.openai_api_key:
        model_name = settings.openai_model_name
        if model_name:
            base_url = _normalize_openai_base_url(settings.openai_endpoint)
            # GPT-5 models may consume entire budget on reasoning tokens, use "none" to prevent empty responses
            reasoning_effort = settings.openai_reasoning_effort or (
                "minimal" if model_name.startswith("gpt-5") else None
            )
            # Some model families reject temperature/top_p/logprobs entirely. Avoid setting
            # temperature unless explicitly configured and non-GPT-5.
            temperature = (
                settings.openai_temperature
                if (
                    settings.openai_temperature is not None
                    and not model_name.startswith("gpt-5")
                )
                else None
            )
            return ChatOpenAI(
                model=model_name,
                api_key=settings.openai_api_key,
                base_url=base_url,
                # Prefer Chat Completions for predictable `AIMessage.content`.
                # Some providers/models may support Responses API but return empty content blocks.
                use_responses_api=False,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
            )

    # No valid LLM configuration found
    raise ValueError(
        "No LLM configuration found. Please set OPENAI_ENDPOINT + OPENAI_API_KEY + (OPENAI_MODELNAME or OPENAI_DEPLOYMENT)"
    )


def _normalize_openai_base_url(endpoint: str) -> str:
    """Normalize OpenAI-compatible base URL.

    Azure AI Foundry endpoints are often provided without `/openai/v1`.
    ChatOpenAI expects the OpenAI-compatible base URL, so append it when needed.
    """
    raw = endpoint.strip().rstrip("/")
    if not raw:
        return raw

    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    path = parsed.path.rstrip("/")

    if path.endswith("/openai/v1"):
        return raw
    if path.endswith("/v1"):
        return raw

    is_azure_host = host.endswith(".cognitiveservices.azure.com") or host.endswith(
        ".openai.azure.com"
    )
    if is_azure_host:
        return f"{raw}/openai/v1"
    return raw


def build_embed_query(settings: Settings) -> Callable[[str], list[float]]:
    """Build embedding query function.

    Args:
        settings: Runtime settings

    Returns:
        Function that takes text and returns embedding vector

    Raises:
        ValueError: If embedding configuration is missing or invalid
    """
    embedding_endpoint = (
        os.environ.get("EMBEDDING_ENDPOINT") or settings.openai_endpoint
    )
    embedding_api_key = os.environ.get("EMBEDDING_API_KEY") or settings.openai_api_key
    if not embedding_endpoint or not embedding_api_key:
        raise ValueError(
            "Configure EMBEDDING_ENDPOINT/EMBEDDING_API_KEY "
            "(or fallback OPENAI_ENDPOINT/OPENAI_API_KEY)"
        )

    embedding_model = os.environ.get("EMBEDDING_MODELNAME") or os.environ.get(
        "EMBEDDING_DEPLOYMENT"
    )
    if not embedding_model:
        raise ValueError("Configure EMBEDDING_MODELNAME or EMBEDDING_DEPLOYMENT")

    if "/openai/v1" not in embedding_endpoint:
        raise ValueError("EMBEDDING_ENDPOINT must end with /openai/v1")

    batch_size_raw = os.environ.get("EMBEDDING_BATCH_SIZE")
    kwargs: dict[str, int] = {}
    if batch_size_raw:
        try:
            batch_size = int(batch_size_raw)
            if batch_size > 0:
                kwargs["chunk_size"] = batch_size
        except ValueError:
            pass

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=embedding_api_key,
        base_url=embedding_endpoint,
        **kwargs,
    )
    return lambda text: embeddings.embed_query(text)
