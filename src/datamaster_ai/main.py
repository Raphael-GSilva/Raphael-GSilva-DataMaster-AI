from datamaster_ai.agents.graph import DataMasterGraph


def main() -> None:
    """
    Ponto de entrada principal do Raphael-GSilva DataMaster AI.
    """

    print("\n" + "=" * 60)
    print("Raphael-GSilva DataMaster AI")
    print("=" * 60)
    print()

    print("Inicializando DataMasterGraph...\n")

    agent = DataMasterGraph()

    print("DataMasterGraph iniciado com sucesso.")
    print()

    prompt = (
        "Explique em uma frase o que é o Raphael-GSilva DataMaster AI."
    )

    print("Pergunta:")
    print(prompt)
    print()

    print("Resposta:")
    print(agent.invoke(prompt))

    print()
    print("=" * 60)
    print("Execução concluída.")
    print("=" * 60)


if __name__ == "__main__":
    main()