from datamaster_ai.agents.graph import DataMasterGraph


def test_datamaster_graph_memory() -> None:
    agent = DataMasterGraph()

    response_1 = agent.invoke(
        "Meu projeto principal se chama Raphael-GSilva DataMaster AI."
    )

    assert response_1 is not None
    assert str(response_1).strip() != ""

    response_2 = agent.invoke(
        "Qual é o nome do meu projeto principal?"
    )

    assert response_2 is not None

    response_text = str(response_2).lower()

    assert "raphael-gsilva datamaster ai" in response_text