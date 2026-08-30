from pathlib import Path
from tempfile import gettempdir
from uuid import uuid4

from datamaster_ai.core.vector_memory import VectorMemory


def create_test_database_path() -> Path:
    """
    Cria um diretório exclusivo para os testes.
    """

    path = (
        Path(gettempdir())
        / f"datamaster_vector_test_{uuid4().hex}"
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def cleanup_database_path(
    database_path: Path,
) -> None:
    """
    Remove o diretório de teste após a execução.

    O Windows pode manter arquivos do ChromaDB
    temporariamente bloqueados. Por isso, a limpeza
    é feita de forma tolerante.
    """

    if not database_path.exists():
        return

    for path in database_path.rglob("*"):
        try:
            if path.is_file():
                path.unlink(
                    missing_ok=True
                )
        except PermissionError:
            pass

    for path in sorted(
        database_path.rglob("*"),
        reverse=True,
    ):
        try:
            if path.is_dir():
                path.rmdir()
        except OSError:
            pass

    try:
        database_path.rmdir()
    except OSError:
        pass


def test_vector_memory_add_and_search() -> None:
    database_path = create_test_database_path()

    try:
        memory = VectorMemory(
            database_path=database_path
        )

        content = (
            "TESTE_VECTOR_MEMORY: "
            "O Raphael-GSilva DataMaster AI utiliza "
            "LangGraph para construir e orquestrar seus agentes."
        )

        memory.add(content)

        results = memory.search(
            "Qual tecnologia o DataMaster AI utiliza "
            "para construir e orquestrar seus agentes?",
            limit=5,
        )

        assert results

        assert any(
            result.get("content") == content
            for result in results
        )

        memory.close()

    finally:
        cleanup_database_path(
            database_path
        )


def test_vector_memory_persistence() -> None:
    database_path = create_test_database_path()

    try:
        first_memory = VectorMemory(
            database_path=database_path
        )

        content = (
            "TESTE_PERSISTENCIA: "
            "O Raphael-GSilva DataMaster AI utiliza "
            "memória vetorial para recuperar informações."
        )

        first_memory.add(content)

        assert first_memory.count() == 1

        first_memory.close()

        second_memory = VectorMemory(
            database_path=database_path
        )

        try:
            assert second_memory.count() == 1

            results = second_memory.search(
                "Qual informação está armazenada "
                "na memória vetorial?",
                limit=5,
            )

            assert results

            assert any(
                result.get("content") == content
                for result in results
            )

        finally:
            second_memory.close()

    finally:
        cleanup_database_path(
            database_path
        )


def test_vector_memory_clear() -> None:
    database_path = create_test_database_path()

    try:
        memory = VectorMemory(
            database_path=database_path
        )

        memory.add(
            "Primeira memória do DataMaster AI."
        )

        memory.add(
            "Segunda memória do DataMaster AI."
        )

        assert memory.count() == 2

        memory.clear()

        assert memory.count() == 0

        memory.close()

    finally:
        cleanup_database_path(
            database_path
        )