from __future__ import annotations

from pathlib import Path
from tempfile import gettempdir
from uuid import uuid4

from datamaster_ai.core.knowledge_base import (
    KnowledgeBase,
)
from datamaster_ai.core.project_indexer import (
    ProjectIndexer,
)
from datamaster_ai.core.vector_memory import (
    VectorMemory,
)


def create_test_directory(
    prefix: str,
) -> Path:
    """
    Cria diretório exclusivo para um teste.
    """

    path = (
        Path(gettempdir())
        / f"{prefix}_{uuid4().hex}"
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def test_project_indexer_indexes_supported_files() -> None:
    root = create_test_directory(
        "datamaster_project_index"
    )

    vector_path = create_test_directory(
        "datamaster_project_vector"
    )

    (
        root
        / "main.py"
    ).write_text(
        (
            "PROJECT_INDEX_TEST = True\n"
            "\n"
            "def execute():\n"
            "    return 'LangGraph'\n"
        ),
        encoding="utf-8",
    )

    (
        root
        / "README.md"
    ).write_text(
        (
            "# Projeto de Teste\n\n"
            "Projeto utilizando Python e LangGraph."
        ),
        encoding="utf-8",
    )

    memory = VectorMemory(
        database_path=vector_path
    )

    try:
        knowledge_base = (
            KnowledgeBase(
                vector_memory=memory
            )
        )

        indexer = ProjectIndexer(
            knowledge_base=knowledge_base
        )

        report = (
            indexer.index_project(
                root
            )
        )

        assert report.indexed_files == 2

        assert report.indexed_chunks >= 2

        assert report.failed_files == 0

        results = knowledge_base.search(
            "PROJECT_INDEX_TEST"
        )

        assert results

        assert any(
            "PROJECT_INDEX_TEST"
            in result.get(
                "content",
                "",
            )
            for result in results
        )

    finally:
        memory.close()


def test_project_indexer_ignores_internal_directories() -> None:
    root = create_test_directory(
        "datamaster_project_ignore"
    )

    vector_path = create_test_directory(
        "datamaster_project_ignore_vector"
    )

    hidden_directory = (
        root
        / ".venv"
    )

    hidden_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        hidden_directory
        / "ignored.py"
    ).write_text(
        (
            "THIS_FILE_MUST_NOT_BE_INDEXED = True"
        ),
        encoding="utf-8",
    )

    (
        root
        / "valid.py"
    ).write_text(
        (
            "THIS_FILE_MUST_BE_INDEXED = True"
        ),
        encoding="utf-8",
    )

    memory = VectorMemory(
        database_path=vector_path
    )

    try:
        knowledge_base = (
            KnowledgeBase(
                vector_memory=memory
            )
        )

        indexer = ProjectIndexer(
            knowledge_base=knowledge_base
        )

        report = (
            indexer.index_project(
                root
            )
        )

        assert report.indexed_files == 1

        results = knowledge_base.search(
            "THIS_FILE_MUST_NOT_BE_INDEXED"
        )

        assert not any(
            "THIS_FILE_MUST_NOT_BE_INDEXED"
            in result.get(
                "content",
                "",
            )
            for result in results
        )

    finally:
        memory.close()


def test_project_indexer_skips_unsupported_files() -> None:
    root = create_test_directory(
        "datamaster_project_unsupported"
    )

    vector_path = create_test_directory(
        "datamaster_project_unsupported_vector"
    )

    (
        root
        / "valid.txt"
    ).write_text(
        "Arquivo válido para indexação.",
        encoding="utf-8",
    )

    (
        root
        / "image.exe"
    ).write_bytes(
        b"NOT_A_REAL_EXECUTABLE"
    )

    memory = VectorMemory(
        database_path=vector_path
    )

    try:
        knowledge_base = (
            KnowledgeBase(
                vector_memory=memory
            )
        )

        indexer = ProjectIndexer(
            knowledge_base=knowledge_base
        )

        report = (
            indexer.index_project(
                root
            )
        )

        assert report.indexed_files == 1

        assert report.skipped_files == 1

        assert report.failed_files == 0

    finally:
        memory.close()