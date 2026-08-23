from collections.abc import Callable
from typing import Any


class ToolRegistry:
    """
    Registro central de ferramentas do Raphael-GSilva DataMaster AI.

    Responsável por:
    - registrar ferramentas;
    - verificar ferramentas disponíveis;
    - recuperar ferramentas;
    - executar ferramentas;
    - listar ferramentas registradas.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(
        self,
        name: str,
        tool: Callable[..., Any],
    ) -> None:
        """
        Registra uma ferramenta.
        """

        if not name.strip():
            raise ValueError(
                "O nome da ferramenta não pode ser vazio."
            )

        if not callable(tool):
            raise TypeError(
                f"A ferramenta '{name}' precisa ser executável."
            )

        self._tools[name] = tool

    def unregister(
        self,
        name: str,
    ) -> None:
        """
        Remove uma ferramenta registrada.
        """

        self._tools.pop(name, None)

    def has(
        self,
        name: str,
    ) -> bool:
        """
        Verifica se uma ferramenta está registrada.
        """

        return name in self._tools

    def get(
        self,
        name: str,
    ) -> Callable[..., Any]:
        """
        Recupera uma ferramenta registrada.
        """

        if name not in self._tools:
            raise KeyError(
                f"A ferramenta '{name}' não está registrada."
            )

        return self._tools[name]

    def execute(
        self,
        name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Executa uma ferramenta registrada.
        """

        tool = self.get(name)

        return tool(
            *args,
            **kwargs,
        )

    def list_tools(self) -> list[str]:
        """
        Retorna os nomes das ferramentas registradas.
        """

        return sorted(self._tools.keys())

    def count(self) -> int:
        """
        Retorna a quantidade de ferramentas registradas.
        """

        return len(self._tools)

    def clear(self) -> None:
        """
        Remove todas as ferramentas registradas.
        """

        self._tools.clear()


def create_tool_registry() -> ToolRegistry:
    """
    Cria um registro de ferramentas vazio.
    """

    return ToolRegistry()