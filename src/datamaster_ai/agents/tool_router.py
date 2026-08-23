from typing import Literal


ToolName = Literal["file", "python", "none"]


class ToolRouter:
    """
    Decide qual ferramenta deve ser utilizada
    pelo Raphael-GSilva DataMaster AI.
    """

    FILE_KEYWORDS = (
        "leia o arquivo",
        "ler o arquivo",
        "leia arquivo",
        "ler arquivo",
        "leia o conteúdo do arquivo",
        "ler o conteúdo do arquivo",
        "leia o conteudo do arquivo",
        "ler o conteudo do arquivo",
        "conteúdo do arquivo",
        "conteudo do arquivo",
        "arquivo txt",
        "arquivo .txt",
        "arquivo md",
        "arquivo .md",
        "arquivo json",
        "arquivo .json",
        "arquivo csv",
        "arquivo .csv",
        "arquivo yaml",
        "arquivo .yaml",
        "arquivo yml",
        "arquivo .yml",
        "arquivo log",
        "arquivo .log",
    )

    PYTHON_KEYWORDS = (
        "calcule usando python",
        "calcular usando python",
        "execute em python",
        "executar em python",
        "rode em python",
        "rodar em python",
        "execute python",
        "executar python",
        "use python",
        "usando python",
        "código python",
        "codigo python",
        "script python",
    )

    CALCULATION_KEYWORDS = (
        "calcule",
        "calcular",
        "calcula",
        "faça a conta",
        "faca a conta",
        "faça o cálculo",
        "faca o calculo",
        "resultado de",
    )

    def detect(self, message: str) -> ToolName:
        """
        Identifica qual ferramenta deve ser utilizada.
        """

        if not isinstance(message, str):
            raise TypeError(
                "A mensagem precisa ser uma string."
            )

        normalized = (
            message.lower()
            .strip()
        )

        if self._is_file_request(normalized):
            return "file"

        if self._is_python_request(normalized):
            return "python"

        return "none"

    def _is_file_request(
        self,
        message: str,
    ) -> bool:
        """
        Verifica se a mensagem solicita leitura de arquivo.
        """

        return any(
            keyword in message
            for keyword in self.FILE_KEYWORDS
        )

    def _is_python_request(
        self,
        message: str,
    ) -> bool:
        """
        Verifica se a mensagem solicita execução Python.
        """

        if any(
            keyword in message
            for keyword in self.PYTHON_KEYWORDS
        ):
            return True

        if any(
            keyword in message
            for keyword in self.CALCULATION_KEYWORDS
        ):
            return self._contains_math_expression(message)

        return False

    @staticmethod
    def _contains_math_expression(
        message: str,
    ) -> bool:
        """
        Identifica expressões matemáticas simples
        presentes na mensagem.
        """

        operators = (
            "+",
            "-",
            "*",
            "/",
            "%",
            "^",
        )

        has_operator = any(
            operator in message
            for operator in operators
        )

        if not has_operator:
            return False

        has_number = any(
            character.isdigit()
            for character in message
        )

        return has_number