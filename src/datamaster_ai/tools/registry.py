from datamaster_ai.core.file_tool import FileTool
from datamaster_ai.core.python_tool import PythonTool
from datamaster_ai.tools.tool_registry import ToolRegistry


class DataMasterToolRegistry:
    """
    Registro central das ferramentas do Raphael-GSilva DataMaster AI.
    """

    def __init__(self) -> None:
        self.registry = ToolRegistry()

        self.file_tool = FileTool()
        self.python_tool = PythonTool()

        self._register_tools()

    def _register_tools(self) -> None:
        """
        Registra todas as ferramentas disponíveis.
        """

        self.registry.register(
            "file",
            self.file_tool.read,
        )

        self.registry.register(
            "python",
            self.python_tool.execute,
        )

    def list_tools(self) -> list[str]:
        """
        Retorna os nomes das ferramentas disponíveis.
        """

        return self.registry.list_tools()

    def count(self) -> int:
        """
        Retorna a quantidade de ferramentas disponíveis.
        """

        return self.registry.count()

    def execute(
        self,
        tool_name: str,
        argument,
    ):
        """
        Executa uma ferramenta registrada.
        """

        return self.registry.execute(
            tool_name,
            argument,
        )


def create_tool_registry() -> DataMasterToolRegistry:
    """
    Cria o registro central de ferramentas.
    """

    return DataMasterToolRegistry()