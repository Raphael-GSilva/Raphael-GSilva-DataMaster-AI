from datamaster_ai.agents.graph import DataMasterGraph


def test_datamaster_graph_tools_and_invoke() -> None:
    agent = DataMasterGraph()

    tools = agent.tools.list_tools()

    assert isinstance(tools, list)
    assert agent.tools.count() >= 1

    response = agent.invoke(
        "Explique brevemente o que é Engenharia de Dados."
    )

    assert response is not None
    assert str(response).strip() != ""