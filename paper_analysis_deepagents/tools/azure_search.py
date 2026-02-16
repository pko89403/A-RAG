from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
from urllib import request
from urllib.error import HTTPError, URLError


@dataclass(frozen=True)
class AzureSearchConfig:
    endpoint: str
    api_key: str
    api_version: str = "2023-11-01"


class AzureSearchError(RuntimeError):
    pass


def _format_http_error(prefix: str, e: HTTPError) -> AzureSearchError:
    detail = ""
    try:
        raw = e.read().decode("utf-8", errors="replace")
        payload = json.loads(raw)
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                code = err.get("code")
                msg = err.get("message")
                if code or msg:
                    detail = f" ({code}: {msg})"
            elif raw:
                detail = f" ({raw})"
    except Exception:
        detail = ""
    return AzureSearchError(f"{prefix}: {e.code} {e.reason}{detail}")


def get_index_schema(config: AzureSearchConfig, *, index_name: str) -> dict[str, Any]:
    """Fetch the Azure AI Search index schema.

    This is used by `--doctor` to verify that expected fields exist.
    """
    url = f"{config.endpoint.rstrip('/')}/indexes/{index_name}?api-version={config.api_version}"
    req = request.Request(
        url,
        method="GET",
        headers={
            "Content-Type": "application/json",
            "api-key": config.api_key,
        },
    )
    try:
        with request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:  # pragma: no cover
        raise _format_http_error("Azure AI Search HTTPError", e) from e
    except URLError as e:  # pragma: no cover
        raise AzureSearchError(f"Azure AI Search URLError: {e.reason}") from e


def _post_search(
    config: AzureSearchConfig, *, index_name: str, payload: dict[str, Any]
) -> dict[str, Any]:
    """POST `/docs/search` to Azure AI Search and return the decoded JSON payload."""
    url = f"{config.endpoint.rstrip('/')}/indexes/{index_name}/docs/search?api-version={config.api_version}"
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "api-key": config.api_key,
        },
        data=body,
    )
    try:
        with request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:  # pragma: no cover
        raise _format_http_error("Azure AI Search HTTPError", e) from e
    except URLError as e:  # pragma: no cover
        raise AzureSearchError(f"Azure AI Search URLError: {e.reason}") from e


def search(
    config: AzureSearchConfig,
    index_name: str,
    query: str = "*",
    *,
    top: int = 5,
    skip: int = 0,
    filter: str | None = None,
    facets: list[str] | None = None,
    select: str | None = None,
    highlight_fields: list[str] | None = None,
    highlight_pre_tag: str | None = None,
    highlight_post_tag: str | None = None,
    vector_queries: list[dict[str, Any]] | None = None,
    search_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Perform a generic search against Azure AI Search."""
    payload: dict[str, Any] = {
        "search": query,
        "top": top,
        "skip": skip,
    }
    if filter:
        payload["filter"] = filter
    if facets:
        payload["facets"] = facets
    if select:
        payload["select"] = select
    if highlight_fields:
        payload["highlight"] = ",".join(highlight_fields)
    if highlight_pre_tag:
        payload["highlightPreTag"] = highlight_pre_tag
    if highlight_post_tag:
        payload["highlightPostTag"] = highlight_post_tag
    if vector_queries:
        payload["vectorQueries"] = vector_queries
    if search_fields:
        payload["searchFields"] = ",".join(search_fields)

    return _post_search(config, index_name=index_name, payload=payload)
