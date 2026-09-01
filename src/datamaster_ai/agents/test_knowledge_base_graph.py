from __future__ import annotations

from pathlib import Path
from tempfile import gettempdir
from uuid import uuid4

from datamaster_ai.agents.graph import (
    DataMasterGraph,
)
from datamaster_ai.core.knowledge_base import (
    KnowledgeBase,
)
from datamaster_ai.core.vector_memory import (
    VectorMemory,
)


def create_test_database_path() -> Path:
    path = (
        Path(gettempdir())
        / (
            "datamaster_graph_kb_test_"
            f"{uuid4().hex}"
        )
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def test_graph_loads_knowledge_base() -> None:
    database_path = (
        create_test_database_path()
    )

    memory = VectorMemory(
        database_path=database_path
    )

    try:
        knowledge_base = (
            KnowledgeBase(
                vector_memory=memory
            )
        )

        content = (
            "DOCUMENTO_RAG_TESTE: "
            "O módulo responsável pela orquestração "
            "dos agentes do DataMaster AI é o LangGraph."
        )

        knowledge_base.add_text(
            content=content,
            source="rag_test",
        )

        graph = object.__new__(
            DataMasterGraph
        )

        graph.knowledge_base = (
            knowledge_base
        )

        state = {
            "message": (
                "Qual módulo é responsável "
                "pela orquestração dos agentes?"
            ),
            "response": "",
            "context": "",
            "vector_context": "",
            "knowledge_context": "",
            "tool_result": "",
            "tool_name": "none",
            "plan": "",
        }

        result = (
            graph._load_knowledge_base(
                state
            )
        )

        assert (
            "BASE DE CONHECIMENTO"
            in result[
                "knowledge_context"
            ]
        )

        assert (
            "LangGraph"
            in result[
                "knowledge_context"
            ]
        )

        assert (
            "DOCUMENTO_RAG_TESTE"
            in result[
                "knowledge_context"
            ]
        )

    finally:
        memory.close()