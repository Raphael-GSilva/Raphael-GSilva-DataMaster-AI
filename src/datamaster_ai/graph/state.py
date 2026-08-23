from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict):
    """
    Estado compartilhado entre todos os nós do LangGraph.
    """

    user_input: str

    messages: List[Dict[str, Any]]

    memory: Dict[str, Any]

    current_task: str

    next_step: str

    result: str

    error: str