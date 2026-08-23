from datamaster_ai.core.structured_memory import StructuredMemory


def test_structured_memory() -> None:
    memory = StructuredMemory()

    items = memory.get_all()

    assert isinstance(items, list)

    for item in items:
        assert "category" in item
        assert "key" in item
        assert "value" in item

    if items:
        assert all(
            item["category"]
            and item["key"]
            and item["value"]
            for item in items
        )