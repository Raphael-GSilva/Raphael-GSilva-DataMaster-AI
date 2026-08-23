from datamaster_ai.core.python_tool import PythonTool


def test_python_tool() -> None:
    tool = PythonTool()

    result = tool.execute("print(125 * 48)")

    assert result is not None
    assert result["success"] is True
    assert result["return_code"] == 0
    assert "6000" in result["stdout"]