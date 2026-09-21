"""Information gathering, data clarification, and problem-definition generation.

The Intake stage is implemented by IntakeAgent using StageAgent tool calling.
This module retains validation functions shared by IntakeAgent.
"""

from typing import Any


def _validate_clarification_questions(parsed: dict) -> list[dict[str, Any]]:
    """Validate LLM questions against the clarification contract."""
    raw_questions = parsed.get("questions")
    if not isinstance(raw_questions, list):
        raise ValueError("questions must be an array")

    questions: list[dict[str, Any]] = []
    for index, item in enumerate(raw_questions, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"questions[{index}] must be an object")

        question_id = item.get("id")
        text = item.get("text")
        answer_type = item.get("answer_type")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"questions[{index}].id must be a non-empty string")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"questions[{index}].text must be a non-empty string")
        if answer_type not in {"text", "boolean", "choice"}:
            raise ValueError(f"questions[{index}].answer_type must be text, boolean, or choice")

        question: dict[str, Any] = {
            "id": question_id,
            "text": text,
            "answer_type": answer_type,
        }
        options = item.get("options")
        if answer_type == "choice":
            if not isinstance(options, list) or not options:
                raise ValueError(f"questions[{index}].options must be a non-empty array")
            if not all(isinstance(option, str) and option.strip() for option in options):
                raise ValueError(f"questions[{index}].options must contain only non-empty strings")
            question["options"] = options
        elif options is not None:
            if not isinstance(options, list):
                raise ValueError(f"questions[{index}].options must be an array")
            if not all(isinstance(option, str) for option in options):
                raise ValueError(f"questions[{index}].options must contain only strings")
            question["options"] = options
        questions.append(question)
    return questions


def _validate_clarification_decision(
    parsed: dict, current_definition: dict[str, Any] | None
) -> tuple[str, dict[str, Any] | None, bool, list[dict[str, Any]]]:
    """Validate structured routing without inferring state from prose."""
    decision = parsed.get("decision")
    if decision not in {"clarify", "proceed"}:
        raise ValueError("decision must be clarify or proceed")

    message = str(parsed.get("message") or "").strip()
    questions = _validate_clarification_questions(parsed)
    supplied_definition = parsed.get("problem_definition")
    if supplied_definition is not None and not isinstance(supplied_definition, dict):
        raise ValueError("problem_definition must be an object or null")

    if decision == "clarify":
        if not questions:
            raise ValueError("decision=clarify requires one to three questions")
        return message, supplied_definition or current_definition, False, questions

    if questions:
        raise ValueError("decision=proceed requires an empty questions array")
    if not supplied_definition:
        raise ValueError("decision=proceed requires a complete problem_definition")
    return message, supplied_definition, True, []
