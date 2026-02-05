from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolResult:
    name: str
    ok: bool
    payload: Any
    error: str | None = None


class MCPToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}

    def register(self, name: str, handler: Callable[..., Any], description: str) -> None:
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = {"handler": handler, "description": description}

    def describe(self) -> list[dict[str, str]]:
        return [
            {"name": name, "description": meta["description"]}
            for name, meta in self._tools.items()
        ]

    def run(self, name: str, **kwargs: Any) -> ToolResult:
        if name not in self._tools:
            return ToolResult(name=name, ok=False, payload=None, error=f"Unknown tool: {name}")
        handler = self._tools[name]["handler"]
        try:
            payload = handler(**kwargs)
        except Exception as exc:
            return ToolResult(name=name, ok=False, payload=None, error=str(exc))
        return ToolResult(name=name, ok=True, payload=payload, error=None)
