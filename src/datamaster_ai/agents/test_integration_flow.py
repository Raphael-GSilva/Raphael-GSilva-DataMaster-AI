from datamaster_ai.agents.graph import DataMasterGraph


def test_general_conversation() -> None:
    agent = DataMasterGraph()

    response = agent.invoke(
        "Explique em uma frase o que é Engenharia de Dados."
    )

    assert response
    assert isinstance(response, str)


def test_python_execution() -> None:
    agent = DataMasterGraph()

    response = agent.invoke(
        "calcule 125 * 48"
    )

    assert response
    assert "6000" in response


def test_file_execution() -> None:
    agent = DataMasterGraph()

    response = agent.invoke(
        "leia o arquivo teste_file_tool.txt"
    )

    assert response
    assert "DataMaster AI" in response