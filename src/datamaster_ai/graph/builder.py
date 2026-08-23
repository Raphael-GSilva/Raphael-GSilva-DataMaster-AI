from datamaster_ai.graph.nodes import AgentNodes
from datamaster_ai.graph.state import AgentState


class GraphBuilder:
    """
    Responsável por orquestrar o fluxo principal do agente.

    Nesta primeira versão, a execução é sequencial.
    Posteriormente será substituída pelo LangGraph.
    """

    def run(self, user_input: str) -> AgentState:
        state: AgentState = {
            "user_input": user_input,
            "messages": [],
            "memory": {},
            "current_task": "",
            "next_step": "",
            "result": "",
            "error": "",
        }

        state = AgentNodes.start(state)
        state = AgentNodes.planner(state)
        state = AgentNodes.finish(state)

        return state