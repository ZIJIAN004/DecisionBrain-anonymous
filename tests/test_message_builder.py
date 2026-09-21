from decisionbrain.core.message_builder import MessageBuilder


def test_builds_ordered_prompt_messages_and_function_tools():
    builder = MessageBuilder.from_prompts(system="system", developer="contract")
    parameters = {"type": "object", "properties": {"key": {"type": "string"}}}

    builder.add_user("problem")
    builder.add_function_tool(
        name="lookup",
        description="Lookup data",
        parameters=parameters,
    )

    request = builder.build()
    assert request.messages == [
        {"role": "system", "content": "system\n\ncontract"},
        {"role": "user", "content": "problem"},
    ]
    assert request.tools == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup data",
                "parameters": parameters,
            },
        }
    ]


def test_tool_call_messages_are_linked_by_tool_call_id():
    builder = MessageBuilder.from_prompts(system="system")
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "validate", "arguments": "{}"},
    }

    builder.add_user("problem")
    builder.add_assistant(tool_calls=[tool_call])
    builder.add_tool_result("call-1", "ok")
    builder.add_assistant(content='{"done": true}')

    assert builder.build().messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "problem"},
        {"role": "assistant", "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
        {"role": "assistant", "content": '{"done": true}'},
    ]


def test_prompt_replacement_preserves_system_developer_order():
    """set_developer 将内容拼接到 system 消息末尾；set_system 替换整个 system 消息。"""
    builder = MessageBuilder()
    builder.add_user("problem")
    builder.set_system("system-v1")
    builder.set_developer("contract-v1")
    builder.set_system("system-v2")
    builder.set_developer("contract-v2")

    assert builder.build().messages == [
        {"role": "system", "content": "system-v2\n\ncontract-v2"},
        {"role": "user", "content": "problem"},
    ]


def test_build_returns_independent_mutable_state():
    builder = MessageBuilder.from_prompts(system="system", developer="contract")
    builder.add_user("problem")

    built = builder.build()
    built.messages[0]["content"] = "mutated"
    builder.add_assistant(content="answer")

    assert builder.build().messages[0]["content"] == "system\n\ncontract"
    assert builder.build().messages[-1] == {"role": "assistant", "content": "answer"}
