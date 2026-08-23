from pathlib import Path
from typing import Optional

from datamaster_ai.config.settings import settings


class FileTool:
    """
    Ferramenta de leitura de arquivos do Raphael-GSilva DataMaster AI.

    Permite ao agente consultar arquivos dentro do Workspace
    sem modificar o conteúdo original.
    """

    ALLOWED_EXTENSIONS = {
        ".txt",
        ".md",
        ".py",
        ".json",
        ".csv",
        ".yaml",
        ".yml",
        ".log",
    }

    def __init__(
        self,
        workspace: Optional[Path] = None,
    ) -> None:
        self.workspace = (
            workspace
            or settings.WORKSPACE_DIR
        )

        self.workspace.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _resolve_path(
        self,
        file_path: str,
    ) -> Path:
        """
        Resolve o caminho garantindo que o arquivo
        permaneça dentro do Workspace.
        """

        requested = (
            self.workspace / file_path
        ).resolve()

        workspace = self.workspace.resolve()

        try:
            requested.relative_to(workspace)
        except ValueError as exc:
            raise ValueError(
                "Acesso ao arquivo fora do Workspace não é permitido."
            ) from exc

        return requested

    def exists(
        self,
        file_path: str,
    ) -> bool:
        """
        Verifica se um arquivo existe.
        """

        path = self._resolve_path(file_path)

        return path.is_file()

    def read(
        self,
        file_path: str,
        encoding: str = "utf-8",
    ) -> str:
        """
        Lê o conteúdo de um arquivo.
        """

        path = self._resolve_path(file_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {file_path}"
            )

        if (
            path.suffix.lower()
            not in self.ALLOWED_EXTENSIONS
        ):
            raise ValueError(
                f"Extensão não permitida: {path.suffix}"
            )

        return path.read_text(
            encoding=encoding,
        )

    def list_files(self) -> list[str]:
        """
        Lista os arquivos permitidos dentro do Workspace.
        """

        files = []

        for path in self.workspace.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower()
                in self.ALLOWED_EXTENSIONS
            ):
                relative = path.relative_to(
                    self.workspace
                )

                files.append(
                    str(relative)
                )

        return sorted(files)