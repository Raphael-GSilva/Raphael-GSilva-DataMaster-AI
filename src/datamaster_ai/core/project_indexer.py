from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from datamaster_ai.core.knowledge_base import KnowledgeBase


@dataclass(frozen=True)
class ProjectIndexReport:
    """
    Resultado da indexação de um projeto.
    """

    root: Path
    indexed_files: int
    indexed_chunks: int
    skipped_files: int
    failed_files: int
    errors: tuple[str, ...]


class ProjectIndexer:
    """
    Indexador de projetos do Raphael-GSilva DataMaster AI.

    Responsabilidades:

    - percorrer projetos recursivamente;
    - ignorar diretórios internos e desnecessários;
    - ignorar arquivos muito grandes;
    - selecionar somente extensões suportadas;
    - enviar arquivos válidos para a KnowledgeBase;
    - gerar relatório final da indexação.
    """

    DEFAULT_IGNORED_DIRECTORIES = {
        ".git",
        ".github",
        ".idea",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
        "06_Vector_DB",
    }

    DEFAULT_IGNORED_FILES = {
        ".env",
        ".env.local",
        ".env.production",
        ".env.development",
    }

    DEFAULT_MAX_FILE_SIZE_BYTES = (
        2 * 1024 * 1024
    )

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        max_file_size_bytes: int = (
            DEFAULT_MAX_FILE_SIZE_BYTES
        ),
        ignored_directories: set[str] | None = None,
        ignored_files: set[str] | None = None,
    ) -> None:
        """
        Inicializa o indexador.

        Args:
            knowledge_base:
                Base onde os arquivos serão indexados.

            max_file_size_bytes:
                Tamanho máximo permitido por arquivo.

            ignored_directories:
                Diretórios adicionais que devem ser ignorados.

            ignored_files:
                Arquivos adicionais que devem ser ignorados.
        """

        if not isinstance(
            knowledge_base,
            KnowledgeBase,
        ):
            raise TypeError(
                "knowledge_base deve ser uma instância "
                "de KnowledgeBase."
            )

        if max_file_size_bytes <= 0:
            raise ValueError(
                "max_file_size_bytes deve ser maior que zero."
            )

        self.knowledge_base = knowledge_base

        self.max_file_size_bytes = (
            max_file_size_bytes
        )

        self.ignored_directories = set(
            self.DEFAULT_IGNORED_DIRECTORIES
        )

        if ignored_directories:
            self.ignored_directories.update(
                ignored_directories
            )

        self.ignored_files = set(
            self.DEFAULT_IGNORED_FILES
        )

        if ignored_files:
            self.ignored_files.update(
                ignored_files
            )

    # ------------------------------------------------------------------
    # INDEXAÇÃO
    # ------------------------------------------------------------------

    def index_project(
        self,
        project_path: str | Path,
    ) -> ProjectIndexReport:
        """
        Indexa um projeto completo.

        Args:
            project_path:
                Diretório raiz do projeto.

        Returns:
            ProjectIndexReport com o resultado da operação.
        """

        root = Path(
            project_path
        ).resolve()

        if not root.exists():
            raise FileNotFoundError(
                f"Projeto não encontrado: {root}"
            )

        if not root.is_dir():
            raise ValueError(
                f"O caminho não é um diretório: {root}"
            )

        indexed_files = 0
        indexed_chunks = 0
        skipped_files = 0
        failed_files = 0

        errors: list[str] = []

        for file_path in self._iter_project_files(
            root
        ):
            if not self._should_index_file(
                file_path
            ):
                skipped_files += 1
                continue

            try:
                memory_ids = (
                    self.knowledge_base.add_file(
                        file_path,
                        metadata={
                            "project_root": str(
                                root
                            ),
                            "relative_path": str(
                                file_path.relative_to(
                                    root
                                )
                            ),
                            "knowledge_origin": (
                                "project_indexer"
                            ),
                        },
                    )
                )

                indexed_files += 1

                indexed_chunks += len(
                    memory_ids
                )

            except Exception as exc:
                failed_files += 1

                errors.append(
                    f"{file_path}: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

        return ProjectIndexReport(
            root=root,
            indexed_files=indexed_files,
            indexed_chunks=indexed_chunks,
            skipped_files=skipped_files,
            failed_files=failed_files,
            errors=tuple(errors),
        )

    # ------------------------------------------------------------------
    # DESCOBERTA DE ARQUIVOS
    # ------------------------------------------------------------------

    def _iter_project_files(
        self,
        root: Path,
    ):
        """
        Percorre os arquivos do projeto.

        Diretórios ignorados são eliminados
        antes de seus conteúdos serem processados.
        """

        stack = [
            root
        ]

        while stack:
            current_directory = (
                stack.pop()
            )

            try:
                entries = list(
                    current_directory.iterdir()
                )
            except OSError:
                continue

            for entry in entries:
                if entry.is_dir():
                    if (
                        entry.name
                        in self.ignored_directories
                    ):
                        continue

                    stack.append(
                        entry
                    )

                    continue

                if entry.is_file():
                    yield entry

    # ------------------------------------------------------------------
    # FILTROS
    # ------------------------------------------------------------------

    def _should_index_file(
        self,
        file_path: Path,
    ) -> bool:
        """
        Define se um arquivo pode entrar
        na base de conhecimento.
        """

        if (
            file_path.name
            in self.ignored_files
        ):
            return False

        if (
            file_path.suffix.lower()
            not in self.knowledge_base.ALLOWED_EXTENSIONS
        ):
            return False

        try:
            file_size = (
                file_path.stat().st_size
            )
        except OSError:
            return False

        if (
            file_size
            > self.max_file_size_bytes
        ):
            return False

        return True