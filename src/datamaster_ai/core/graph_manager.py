from typing import TypedDict

from langgraph.graph import StateGraph, END

from datamaster_ai.core.model_manager import ModelManager


class GraphState(TypedDict):
    prompt: str
    response: str


class GraphManager:

    def __init__(self):
        self.model = ModelManager()
        self.graph = self._build()

    def chatbot(self, state: GraphState):

        resposta = self.model.ask(state["prompt"])

        return {
            "prompt": state["prompt"],
            "response": resposta,
        }

    def _build(self):

        workflow = StateGraph(GraphState)

        workflow.add_node("chatbot", self.chatbot)

        workflow.set_entry_point("chatbot")

        workflow.add_edge("chatbot", END)

        return workflow.compile()

    def invoke(self, prompt: str):

        return self.graph.invoke(
            {
                "prompt": prompt,
                "response": "",
            }
        )