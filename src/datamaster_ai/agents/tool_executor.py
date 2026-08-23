from typing import Any

from datamaster_ai.tools.registry import DataMasterToolRegistry


class ToolExecutor:
    """
    Executor central das ferramentas do Raphael-GSilva DataMaster AI.

    Responsável exclusivamente por executar a ferramenta
    escolhida pelo ToolRouter.
    """

    def __init__(
        self,
        registry: DataMasterToolRegistry,
    ) -> None:
        self.registry = registry

    def execute(
        self,
        tool_name: str,
        argument: str,
    ) -> Any:
        """
        Executa a ferramenta selecionada.
        """

        if tool_name == "file":
            return self.registry.execute(
                "file",
                argument,
            )

        if tool_name == "python":
            return self.registry.execute(
                "python",
                argument,
            )

        if tool_name == "none":
            return ""

        raise ValueError(
            f"Ferramenta desconhecida: {tool_name}"
        )