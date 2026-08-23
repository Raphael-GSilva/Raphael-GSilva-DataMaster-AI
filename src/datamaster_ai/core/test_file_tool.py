from datamaster_ai.core.file_tool import FileTool


def test_file_tool() -> None:
    tool = FileTool()

    assert tool.workspace.exists()

    test_file = tool.workspace / "teste_file_tool.txt"

    test_content = (
        "Raphael-GSilva DataMaster AI\n"
        "Teste da FileTool funcionando.\n"
    )

    test_file.write_text(
        test_content,
        encoding="utf-8",
    )

    assert tool.exists("teste_file_tool.txt")

    content = tool.read("teste_file_tool.txt")

    assert content == test_content