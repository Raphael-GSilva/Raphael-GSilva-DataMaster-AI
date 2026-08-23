from datamaster_ai.tools.registry import create_tool_registry


def test_tool_registry() -> None:
    registry = create_tool_registry()

    tools = registry.list_tools()

    assert isinstance(tools, list)
    assert "file" in tools
    assert "python" in tools
    assert registry.count() >= 2

    result = registry.execute(
        "file",
        "teste_file_tool.txt",
    )

    assert result is not None