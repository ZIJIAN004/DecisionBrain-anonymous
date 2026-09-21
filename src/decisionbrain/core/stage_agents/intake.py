"""Intake stage: inspect data through tool calls and clarify the problem definition."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ..contracts import CoreServices
from ..intake import _validate_clarification_decision
from ..message_builder import MessageBuilder
from ..models import AgentReport, AgentStage, AssistantMessage
from ..stage_agent import OutputValidationIssue, StageAgent, StageOutputValidationError
from ..stage_outputs import STAGE_OUTPUT_FILES


class IntakeAgent(StageAgent):
    """Inspect data files through tool calls and clarify the optimization problem.

    The Agent no longer receives a precomputed data profile. It must inspect
    uploaded files itself through workspace tools.
    """

    def __init__(self, services: CoreServices):
        super().__init__(
            stage=AgentStage.INTAKE,
            services=services,
            tools=services.workspace_tools_for(AgentStage.INTAKE),
        )
        self._current_definition: dict[str, Any] | None = None

    def build_messages(self, state: Any) -> MessageBuilder:
        prompts = self.services.prompts
        mb = MessageBuilder.from_prompts(prompts.intake_system, prompts.intake_contract)

        self._current_definition = state.problem_definition

        if self._has_full_transcript(state.conversation):
            # Resume mode retains the complete prior tool-calling transcript. Append the
            # current problem context, then replay every stored message.
            mb.add_user(self._workspace_context(resume=True))
            for entry in state.conversation:
                self._add_transcript_entry(mb, entry)
        else:
            # First run: build the initial context.
            mb.add_user(self._workspace_context(resume=False))
            for entry in state.conversation:
                role = entry.get("role", "")
                content = entry.get("content", "")
                if role == "user":
                    mb.add_user(content)
                elif role == "assistant":
                    mb.add_assistant(content=content)
        return mb

    def _workspace_context(self, *, resume: bool) -> str:
        mode = "续接澄清" if resume else "首次澄清"
        return (
            f"【当前阶段】intake ({mode})\n"
            "【必须读取】\n"
            "- `problem.md`: 原始问题描述。\n"
            "- `data/`: 用户提供的数据文件目录；请用 list_files/read_file/shell 探查。\n"
            "【可选读取】\n"
            f"- `{STAGE_OUTPUT_FILES[AgentStage.INTAKE]}`: 若这是澄清续接，可读取上一轮部分问题定义。\n"
            "【必须写入】\n"
            f"- 最终阶段 JSON 必须写入 `{STAGE_OUTPUT_FILES[AgentStage.INTAKE]}`。\n"
            "写完后请自行读取或用 python json.load 验收该文件；最终 message 只需简短说明完成，"
            "不要粘贴完整 JSON。"
        )

    @staticmethod
    def _has_full_transcript(conversation) -> bool:
        """Return whether the conversation contains complete tool-call records."""
        for entry in conversation:
            if entry.get("role") == "tool" or "tool_calls" in entry:
                return True
        return False

    @staticmethod
    def _add_transcript_entry(mb: MessageBuilder, entry: dict[str, Any]) -> None:
        """Append a conversation entry to the builder with full role semantics."""
        role = entry.get("role", "")
        if role == "user":
            mb.add_user(entry.get("content", ""))
        elif role == "assistant":
            content = entry.get("content")
            tool_calls = entry.get("tool_calls")
            reasoning = entry.get("reasoning_content")
            if isinstance(tool_calls, list):
                mb.add_assistant(tool_calls=tool_calls, reasoning_content=reasoning)
            else:
                mb.add_assistant(content=content, reasoning_content=reasoning)
        elif role == "tool":
            mb.add_tool_result(
                entry.get("tool_call_id", ""),
                entry.get("content", ""),
            )

    def parse_output_file(self, data: dict[str, Any], state: Any) -> dict[str, Any]:
        try:
            message, problem_definition, ready, questions = _validate_clarification_decision(
                data, self._current_definition
            )
            return {
                "reply": message,
                "problem_definition": problem_definition,
                "ready": ready,
                "questions": questions,
            }
        except ValueError as exc:
            raise StageOutputValidationError(
                "intake 输出文件不符合澄清契约",
                issues=(
                    OutputValidationIssue(
                        path="$",
                        message=str(exc),
                        expected="符合 intake_contract 的 intake JSON 对象",
                    ),
                ),
                file_path=self.output_file(state),
            ) from exc

    async def _after_parse(self, output: dict[str, Any], state: Any) -> AsyncIterator[AgentReport]:
        yield AssistantMessage(
            stage=AgentStage.INTAKE,
            message=output.get("reply") or "判断完成",
        )
