from __future__ import annotations

from pathlib import Path
from tempfile import gettempdir
from uuid import uuid4

from datamaster_ai.core.knowledge_base import KnowledgeBase
from datamaster_ai.core.vector_memory import VectorMemory


def create_test_database_path() -> Path:
    path = (
        Path(gettempdir())
        / f"datamaster_kb_test_{uuid4().hex}"
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def test_knowledge_base_add_text() -> None:
    database_path = create_test_database_path()

    memory = VectorMemory(
        database_path=database_path
    )

    try:
        knowledge_base = KnowledgeBase(
            vector_memory=memory,
            chunk_size=200,
            chunk_overlap=20,
        )

        content = (
            "O Raphael-GSilva DataMaster AI utiliza "
            "Python, LangGraph e Ollama. "
            "O sistema possui memória vetorial persistente "
            "e ferramentas para execução de tarefas."
        )

        memory_ids = knowledge_base.add_text(
            content=content,
            source="test",
        )

        assert memory_ids

        assert memory.count() >= 1

        results = knowledge_base.search(
            "Quais tecnologias o DataMaster AI utiliza?"
        )

        assert results

        assert any(
            "LangGraph" in result.get(
                "content",
                "",
            )
            for result in results
        )

    finally:
        memory.close()


def test_knowledge_base_add_file() -> None:
    database_path = create_test_database_path()

    test_file = (
        database_path
        / "knowledge_test.md"
    )

    test_file.write_text(
        (
            "# DataMaster AI\n\n"
            "O agente utiliza Python para desenvolvimento "
            "e LangGraph para orquestração."
        ),
        encoding="utf-8",
    )

    memory = VectorMemory(
        database_path=database_path
        / "vector_db"
    )

    try:
        knowledge_base = KnowledgeBase(
            vector_memory=memory
        )

        memory_ids = knowledge_base.add_file(
            test_file
        )

        assert memory_ids

        results = knowledge_base.search(
            "Qual tecnologia é usada para orquestração?"
        )

        assert results

        assert any(
            "LangGraph" in result.get(
                "content",
                "",
            )
            for result in results
        )

        assert any(
            result.get(
                "metadata",
                {},
            ).get(
                "filename"
            )
            == "knowledge_test.md"
            for result in results
        )

    finally:
        memory.close()


def test_knowledge_base_chunking() -> None:
    database_path = create_test_database_path()

    memory = VectorMemory(
        database_path=database_path
    )

    try:
        knowledge_base = KnowledgeBase(
            vector_memory=memory,
            chunk_size=100,
            chunk_overlap=20,
        )

        content = (
            "Raphael-GSilva DataMaster AI. "
            * 30
        )

        memory_ids = knowledge_base.add_text(
            content=content,
            source="chunk_test",
        )

        assert len(memory_ids) > 1

        assert memory.count() == len(
            memory_ids
        )

    finally:
        memory.close()