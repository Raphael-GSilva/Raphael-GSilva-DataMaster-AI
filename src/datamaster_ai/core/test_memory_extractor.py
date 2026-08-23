from datamaster_ai.core.memory_extractor import MemoryExtractor


def test_memory_extractor() -> None:
    extractor = MemoryExtractor()

    message = (
        "Meu projeto principal é o Raphael-GSilva DataMaster AI. "
        "Ele está sendo desenvolvido em Python e utiliza LangGraph "
        "para orquestração dos agentes."
    )

    saved_items = extractor.extract_and_save(message)

    assert saved_items
    assert isinstance(saved_items, list)

    values = [
        item["value"]
        for item in saved_items
    ]

    assert "Raphael-GSilva DataMaster AI" in values
    assert "Python" in values
    assert "LangGraph" in values