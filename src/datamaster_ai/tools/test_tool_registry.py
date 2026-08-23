from datamaster_ai.tools.registry import DataMasterToolRegistry


def test_tool_registry_execution() -> None:
    registry = DataMasterToolRegistry()

    tools = registry.list_tools()

    assert isinstance(tools, list)
    assert "file" in tools
    assert "python" in tools
    assert registry.count() >= 2

    file_result = registry.execute(
        "file",
        "teste_file_tool.txt",
    )

    assert file_result is not None
    assert "Raphael-GSilva DataMaster AI" in str(file_result)

    python_result = registry.execute(
        "python",
        "print(125 * 48)",
    )

    assert python_result is not None
    assert "6000" in str(python_result)