from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolRuntime
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol
from deepagents.middleware.skills import _list_skills


@dataclass(frozen=True)
class StdoutTraceConfig:
    trace_tools: bool = False
    trace_skills: bool = False
    max_value_chars: int = 300


class StdoutTraceMiddleware(AgentMiddleware):
    """Unified stdout tracing for tools and skills."""

    def __init__(
        self,
        *,
        backend: BACKEND_TYPES | None = None,
        sources: list[str] | None = None,
        config: StdoutTraceConfig | None = None,
        trace_tools: bool | None = None,
        trace_skills: bool | None = None,
        max_value_chars: int | None = None,
    ) -> None:
        super().__init__()
        self._backend = backend
        self.sources = sources or []
        if config is not None:
            self._config = config
        else:
            self._config = StdoutTraceConfig(
                trace_tools=trace_tools if trace_tools is not None else False,
                trace_skills=trace_skills if trace_skills is not None else False,
                max_value_chars=max_value_chars or 300,
            )

    def _get_backend(
        self,
        state,
        runtime: Runtime,
        config: RunnableConfig,
    ) -> BackendProtocol | None:
        if self._backend is None:
            return None
        if callable(self._backend):
            tool_runtime = ToolRuntime(
                state=state,
                context=runtime.context,
                stream_writer=runtime.stream_writer,
                store=runtime.store,
                config=config,
                tool_call_id=None,
            )
            return self._backend(tool_runtime)
        return self._backend

    def before_agent(self, state, runtime: Runtime, config: RunnableConfig):  # type: ignore[override]
        if not self._config.trace_skills:
            return None

        if not self.sources:
            print("[skills] sources: none")
            return None

        try:
            backend = self._get_backend(state, runtime, config)
            if backend is None:
                print("[skills] backend: <none>")
                return None

            print(f"[skills] backend type: {type(backend)}")
            root_dir = getattr(backend, "cwd", None) or getattr(
                backend, "root_dir", None
            )
            if root_dir is not None:
                print(f"[skills] backend root: {root_dir}")
            else:
                alt_root = getattr(backend, "_root_dir", None) or getattr(
                    backend, "base_path", None
                )
                if alt_root is not None:
                    print(f"[skills] backend root (alt): {alt_root}")
                else:
                    print("[skills] backend root: <none>")

            for source_path in self.sources:
                if root_dir is not None:
                    fs_path = root_dir / source_path.strip("/")
                    print(
                        f"[skills] fs path check: {fs_path} exists={fs_path.exists()}"
                    )
                else:
                    print(f"[skills] source path (posix): {source_path}")
                discovered = _list_skills(backend, source_path)
                if discovered:
                    print(f"[skills] discovered from {source_path}:")
                    for skill in discovered:
                        print(
                            f"- {skill['name']}: {skill['description']} ({skill['path']})"
                        )
                else:
                    print(f"[skills] discovered from {source_path}: none")
        except Exception as exc:
            print(f"[skills] discovery error: {exc}")

        skills = state.get("skills_metadata", [])
        if skills:
            print("[skills] metadata loaded:")
            for skill in skills:
                print(f"- {skill['name']}: {skill['description']} ({skill['path']})")
        else:
            print("[skills] metadata loaded: none")
        return None

    def modify_request(self, request):  # type: ignore[override]
        if self._config.trace_skills:
            print("[skills] injected into system prompt")
        return request

    def wrap_tool_call(  # type: ignore[override]
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        name = request.tool_call.get("name")
        args = request.tool_call.get("args")
        if not (self._config.trace_tools or self._config.trace_skills):
            return handler(request)

        self._trace_before(name=name, args=args)
        result = handler(request)
        self._trace_after(name=name, args=args, result=result)
        return result

    async def awrap_tool_call(  # type: ignore[override]
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        name = request.tool_call.get("name")
        args = request.tool_call.get("args")
        if not (self._config.trace_tools or self._config.trace_skills):
            return await handler(request)

        self._trace_before(name=name, args=args)
        result = await handler(request)
        self._trace_after(name=name, args=args, result=result)
        return result

    def _trace_before(self, *, name: Any, args: Any) -> None:
        if self._config.trace_tools:
            if name == "task" and isinstance(args, dict):
                subagent_type = args.get("subagent_type")
                if subagent_type:
                    print(f"[task] subagent_type={subagent_type}")
                else:
                    print("[task] subagent_type=<missing>")
            print(
                f"[tool] -> {name} args={_truncate(args, self._config.max_value_chars)}"
            )

    def _trace_after(
        self, *, name: Any, args: Any, result: ToolMessage | Command[Any]
    ) -> None:
        if isinstance(args, dict) and name == "read_file":
            path = args.get("path") or args.get("file") or ""
            if isinstance(path, str) and path.endswith("SKILL.md"):
                print(f"[skills] loaded: {path}")

        if self._config.trace_tools:
            if isinstance(result, ToolMessage):
                print(
                    f"[tool] <- {name} ToolMessage status={getattr(result, 'status', None)} "
                    f"size={len(result.content or '')}"
                )
            else:
                print(
                    f"[tool] <- {name} Command(update_keys={list(result.update.keys())})"
                )


def _truncate(value: Any, max_chars: int) -> str:
    text = repr(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
