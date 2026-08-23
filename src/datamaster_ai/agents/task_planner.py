from dataclasses import dataclass


@dataclass(frozen=True)
class TaskPlan:
    """
    Representa o plano de execução de uma tarefa.
    """

    tool_name: str
    instruction: str


class TaskPlanner:
    """
    Planeja a execução de uma tarefa a partir
    da ferramenta selecionada pelo ToolRouter.
    """

    def create_plan(
        self,
        message: str,
        tool_name: str,
    ) -> TaskPlan:
        """
        Cria um plano simples e determinístico.
        """

        if tool_name == "python":
            return TaskPlan(
                tool_name="python",
                instruction=(
                    "Executar a solicitação utilizando "
                    "Python e utilizar o resultado real "
                    "da execução."
                ),
            )

        if tool_name == "file":
            return TaskPlan(
                tool_name="file",
                instruction=(
                    "Localizar e ler o arquivo solicitado "
                    "dentro do Workspace."
                ),
            )

        return TaskPlan(
            tool_name="none",
            instruction=(
                "Responder utilizando o modelo sem "
                "executar ferramenta."
            ),
        )