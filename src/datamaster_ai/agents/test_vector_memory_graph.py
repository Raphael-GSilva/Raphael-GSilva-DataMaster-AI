from __future__ import annotations

from tempfile import TemporaryDirectory

from datamaster_ai.agents.graph import DataMasterGraph
from datamaster_ai.core.vector_memory import VectorMemory


def test_graph_loads_relevant_vector_memory() -> None:
    """
    Verifica se o grafo consegue carregar uma memória vetorial
    relevante para a mensagem atual.
    """

    with TemporaryDirectory() as temp_dir:
        vector_memory = VectorMemory(
            persist_directory=temp_dir
        )

        try:
            vector_memory.add(
                "O projeto utiliza Python para Engenharia de Dados."
            )

            graph = DataMasterGraph()

            graph.vector_memory.close()

            graph.vector_memory = vector_memory

            state = {
                "message": (
                    "Qual linguagem estamos "
                    "utilizando no projeto?"
                ),
                "response": "",
                "context": "",
                "vector_context": "",
                "tool_result": "",
                "tool_name": "none",
                "plan": "",
            }

            loaded_state = graph._load_vector_memory(
                state
            )

            assert (
                "Python"
                in loaded_state["vector_context"]
            )

        finally:
            vector_memory.close()

            # Pequena margem para o Windows liberar
            # completamente os handles do SQLite/HNSW.
            import gc

            gc.collect()