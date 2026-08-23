import subprocess
import sys
from typing import Dict


class PythonTool:
    """
    Ferramenta responsável por executar código Python
    de forma controlada para o DataMaster AI.
    """

    def __init__(self) -> None:
        self.python_executable = sys.executable

    def execute(self, code: str) -> Dict[str, object]:
        """
        Executa código Python utilizando o mesmo ambiente
        virtual atualmente ativo.
        """

        if not code.strip():
            return {
                "success": False,
                "stdout": "",
                "stderr": "Nenhum código Python foi fornecido.",
                "return_code": -1,
            }

        try:
            process = subprocess.run(
                [
                    self.python_executable,
                    "-c",
                    code,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            return {
                "success": process.returncode == 0,
                "stdout": process.stdout.strip(),
                "stderr": process.stderr.strip(),
                "return_code": process.returncode,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "A execução excedeu o limite de 30 segundos.",
                "return_code": -1,
            }

        except Exception as exc:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(exc),
                "return_code": -1,
            }


def create_python_tool() -> PythonTool:
    """
    Cria uma instância da ferramenta Python.
    """

    return PythonTool()