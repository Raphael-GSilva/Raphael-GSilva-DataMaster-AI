from loguru import logger

from datamaster_ai.graph.state import AgentState


class AgentNodes:
    """
    Nós responsáveis pelo fluxo principal do LangGraph.
    """

    @staticmethod
    def start(state: AgentState) -> AgentState:
        """
        Primeiro nó do fluxo.
        """

        logger.info("Iniciando execução do agente.")

        state["current_task"] = "start"
        state["next_step"] = "planner"

        return state

    @staticmethod
    def planner(state: AgentState) -> AgentState:
        """
        Planejador inicial.
        """

        logger.info("Planejando tarefa...")

        state["current_task"] = "planner"

        state["result"] = (
            f"Tarefa recebida: {state['user_input']}"
        )

        state["next_step"] = "finish"

        return state

    @staticmethod
    def finish(state: AgentState) -> AgentState:
        """
        Finaliza a execução.
        """

        logger.info("Fluxo finalizado.")

        state["current_task"] = "finish"

        return state