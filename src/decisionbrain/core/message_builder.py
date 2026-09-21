"""Build stage-local system, developer, user, assistant, and tool messages.

Initial messages come from AgentState, and assistant/tool messages accumulate during tool
calling. MessageBuilder is transient and is neither persisted nor shared across stages.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .models import ChatRequest


class MessageBuilder:
    """Build the message list used within a stage.

    Example::

        mb = MessageBuilder.from_prompts(system="You are an optimization expert", developer="Output JSON")
        mb.add_user("[Problem definition] ...")
        response = await llm.complete(mb.build())
        mb.add_assistant(content=None, tool_calls=[...])
        mb.add_tool_result("c1", "Constraint validation passed")
    """

    def __init__(self) -> None:
        self._messages: list[dict[str, Any]] = []
        self._tools: list[dict[str, Any]] = []

    # Initial setup for each stage

    def set_system(self, content: str) -> None:
        """Set the system message and keep it first in the message list."""
        self._replace_or_insert(role="system", content=content)

    def set_developer(self, content: str) -> None:
        """Append developer content to the system message.

        DeepSeek does not support ``role=developer``, so developer content is
        appended to the system message with a blank-line separator."""
        for i, m in enumerate(self._messages):
            if m["role"] == "system":
                self._messages[i] = {"role": "system", "content": m["content"] + "\n\n" + content}
                return
        # Insert a system message when none exists.
        self._replace_or_insert(role="system", content=content)

    def prepend_to_system(self, content: str) -> None:
        """Prepend content to the system message with a blank-line separator.

        This places shared metaprompts before stage-specific prompts."""
        for i, m in enumerate(self._messages):
            if m["role"] == "system":
                self._messages[i] = {"role": "system", "content": content + "\n\n" + m["content"]}
                return
        # Insert a system message when none exists.
        self._replace_or_insert(role="system", content=content)

    @classmethod
    def from_prompts(cls, system: str, developer: str = "") -> "MessageBuilder":
        """Create a builder from ``PromptBundle`` fields.

        ``system`` uses the system role. Developer content is appended to it
        because DeepSeek does not support the developer role."""
        mb = cls()
        mb.set_system(system)
        if developer:
            mb.set_developer(developer)
        return mb

    # Messages appended during the tool-calling loop

    def add_user(self, content: str) -> None:
        """Append a user message such as task data or validation feedback."""
        self._messages.append({"role": "user", "content": content})

    def add_assistant(
        self,
        content: str | None = None,
        *,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        """Append an assistant message.

        OpenAI requires ``content`` to be ``None`` when tool calls are present.
        ``reasoning_content`` carries reasoning from models such as DeepSeek.
        """
        msg: dict[str, Any] = {"role": "assistant"}
        if content is not None:
            msg["content"] = content
        if tool_calls is not None:
            msg["tool_calls"] = tool_calls
        if reasoning_content:
            msg["reasoning_content"] = reasoning_content
        self._messages.append(msg)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """Append a tool message linked to an assistant tool call."""
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content,
            }
        )

    def add_function_tool(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        """Append an OpenAI Chat Completions function tool."""
        self._tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": deepcopy(parameters),
                },
            }
        )

    # Access

    def build(self, *, tool_choice: str = "auto") -> ChatRequest:
        """Return the complete request for ``LLMClient.complete()``."""
        return ChatRequest(
            messages=deepcopy(self._messages),
            tools=deepcopy(self._tools),
            tool_choice=tool_choice,
        )

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return the current message list as a read-only view."""
        return self._messages

    # Internals

    def _replace_or_insert(self, *, role: str, content: str) -> None:
        """Replace a role's message or insert it in the proper position."""
        for i, m in enumerate(self._messages):
            if m["role"] == role:
                self._messages[i] = {"role": role, "content": content}
                return
        if role == "system":
            self._messages.insert(0, {"role": role, "content": content})
