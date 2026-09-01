from __future__ import annotations

from pathlib import Path
from typing import Any

from datamaster_ai.core.vector_memory import VectorMemory


class KnowledgeBase:
    """
    Base de conhecimento do Raphael-GSilva DataMaster AI.

    Responsabilidades:

    - armazenar conhecimento textual;
    - carregar arquivos de texto;
    - carregar diretórios;
    - dividir documentos em chunks;
    - utilizar a VectorMemory existente;
    - recuperar conhecimento por busca semântica;
    - preservar metadados de origem;
    - funcionar integrada ao LangGraph.
    """

    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 150

    DEFAULT_COLLECTION_NAME = "datamaster_knowledge"

    DEFAULT_EMBEDDING_MODEL = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    ALLOWED_EXTENSIONS = {
        ".txt",
        ".md",
        ".py",
        ".json",
        ".csv",
        ".yaml",
        ".yml",
        ".log",
        ".toml",
    }

    def __init__(
        self,
        vector_memory: VectorMemory | None = None,
        persist_directory: str | Path | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        """
        Inicializa a base de conhecimento.

        A KnowledgeBase pode receber uma VectorMemory já existente
        ou criar sua própria instância.

        Quando uma VectorMemory externa é fornecida, a KnowledgeBase
        não assume propriedade sobre ela e não a fecha automaticamente.
        """

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size deve ser maior que zero."
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap não pode ser negativo."
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap deve ser menor que chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if vector_memory is not None:
            self.vector_memory = vector_memory
            self._owns_vector_memory = False

        else:
            self.vector_memory = VectorMemory(
                persist_directory=persist_directory,
                collection_name=collection_name,
                embedding_model=embedding_model,
            )

            self._owns_vector_memory = True

        # Alias mantido por compatibilidade com
        # versões anteriores da KnowledgeBase.
        self.memory = self.vector_memory

    # ------------------------------------------------------------------
    # CHUNKING
    # ------------------------------------------------------------------

    def _split_text(
        self,
        content: str,
    ) -> list[str]:
        """
        Divide um texto em chunks com sobreposição.
        """

        if not isinstance(content, str):
            raise TypeError(
                "content deve ser uma string."
            )

        content = content.strip()

        if not content:
            return []

        if len(content) <= self.chunk_size:
            return [
                content
            ]

        chunks: list[str] = []

        start = 0
        content_length = len(content)

        while start < content_length:
            end = min(
                start + self.chunk_size,
                content_length,
            )

            chunk = content[
                start:end
            ].strip()

            if chunk:
                chunks.append(
                    chunk
                )

            if end >= content_length:
                break

            start = (
                end
                - self.chunk_overlap
            )

        return chunks

    def _chunk_text(
        self,
        text: str,
    ) -> list[str]:
        """
        Alias de compatibilidade para _split_text().
        """

        return self._split_text(
            text
        )

    # ------------------------------------------------------------------
    # TEXTO
    # ------------------------------------------------------------------

    def add_text(
        self,
        content: str | None = None,
        source: str = "manual",
        metadata: dict[str, Any] | None = None,
        *,
        text: str | None = None,
    ) -> list[str]:
        """
        Adiciona um texto à base de conhecimento.

        Aceita 'content' como interface principal.

        O parâmetro 'text' é mantido como alias para
        compatibilidade com versões anteriores.

        Retorna os IDs dos chunks armazenados.
        """

        if content is not None and text is not None:
            raise ValueError(
                "Informe apenas 'content' ou 'text', "
                "não os dois simultaneamente."
            )

        final_content = (
            content
            if content is not None
            else text
        )

        if final_content is None:
            raise ValueError(
                "Nenhum conteúdo foi informado."
            )

        if not isinstance(
            final_content,
            str,
        ):
            raise TypeError(
                "O conteúdo deve ser uma string."
            )

        final_content = (
            final_content.strip()
        )

        if not final_content:
            raise ValueError(
                "O conteúdo não pode estar vazio."
            )

        chunks = self._split_text(
            final_content
        )

        memory_ids: list[str] = []

        total_chunks = len(
            chunks
        )

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):
            chunk_metadata: dict[str, Any] = {
                "source": source,
                "type": "knowledge",
                "chunk": index,
                "chunk_index": index - 1,
                "total_chunks": total_chunks,
                "source_type": "text",
            }

            if metadata:
                for key, value in metadata.items():
                    if value is None:
                        continue

                    chunk_metadata[
                        str(key)
                    ] = value

            memory_id = (
                self.vector_memory.add(
                    content=chunk,
                    metadata=chunk_metadata,
                )
            )

            memory_ids.append(
                memory_id
            )

        return memory_ids

    # ------------------------------------------------------------------
    # ARQUIVOS
    # ------------------------------------------------------------------

    def add_file(
        self,
        file_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """
        Carrega e indexa um arquivo permitido.
        """

        path = Path(
            file_path
        ).resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"O caminho informado não é um arquivo: {path}"
            )

        extension = (
            path.suffix.lower()
        )

        if (
            extension
            not in self.ALLOWED_EXTENSIONS
        ):
            raise ValueError(
                f"Extensão não permitida: {extension}"
            )

        content = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        file_metadata: dict[str, Any] = {
            "filename": path.name,
            "extension": extension,
            "source_file": str(path),
            "source_name": path.name,
            "source_type": "file",
        }

        if metadata:
            for key, value in metadata.items():
                if value is None:
                    continue

                file_metadata[
                    str(key)
                ] = value

        return self.add_text(
            content=content,
            source=str(path),
            metadata=file_metadata,
        )

    # ------------------------------------------------------------------
    # DIRETÓRIOS
    # ------------------------------------------------------------------

    def add_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
    ) -> dict[str, list[str]]:
        """
        Indexa todos os arquivos permitidos de um diretório.

        Args:
            directory:
                Diretório que será analisado.

            recursive:
                Se True, percorre também subdiretórios.

        Returns:
            Dicionário no formato:

            {
                "arquivo": ["id1", "id2", ...]
            }
        """

        path = Path(
            directory
        ).resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"Diretório não encontrado: {path}"
            )

        if not path.is_dir():
            raise ValueError(
                f"O caminho informado não é um diretório: {path}"
            )

        candidates = (
            path.rglob("*")
            if recursive
            else path.glob("*")
        )

        indexed_files: dict[
            str,
            list[str],
        ] = {}

        for file_path in candidates:
            if not file_path.is_file():
                continue

            extension = (
                file_path.suffix.lower()
            )

            if (
                extension
                not in self.ALLOWED_EXTENSIONS
            ):
                continue

            memory_ids = self.add_file(
                file_path
            )

            indexed_files[
                str(file_path)
            ] = memory_ids

        return indexed_files

    # ------------------------------------------------------------------
    # BUSCA
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Realiza busca semântica na base de conhecimento.

        Quando a VectorMemory suporta filtro por metadados,
        busca exclusivamente registros do tipo 'knowledge'.

        A compatibilidade com versões anteriores da
        VectorMemory também é preservada.
        """

        if not isinstance(query, str):
            raise TypeError(
                "query deve ser uma string."
            )

        query = query.strip()

        if not query:
            return []

        if limit <= 0:
            return []

        try:
            results = (
                self.vector_memory.search(
                    query=query,
                    limit=limit,
                    metadata_filter={
                        "type": "knowledge"
                    },
                )
            )

            return results

        except TypeError:
            results = (
                self.vector_memory.search(
                    query=query,
                    limit=limit,
                )
            )

        filtered_results: list[
            dict[str, Any]
        ] = []

        for result in results:
            metadata = result.get(
                "metadata",
                {},
            )

            if not isinstance(
                metadata,
                dict,
            ):
                continue

            if (
                metadata.get("type")
                != "knowledge"
            ):
                continue

            filtered_results.append(
                result
            )

        return filtered_results

    # ------------------------------------------------------------------
    # CONTAGEM
    # ------------------------------------------------------------------

    def count(self) -> int:
        """
        Retorna a quantidade de registros
        presentes na memória vetorial associada.
        """

        return (
            self.vector_memory.count()
        )

    # ------------------------------------------------------------------
    # LIMPEZA
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """
        Limpa a memória vetorial associada.

        Observação:
        como atualmente KnowledgeBase e VectorMemory podem
        compartilhar a mesma coleção, este método limpa
        toda a coleção utilizada pela instância.
        """

        self.vector_memory.clear()

    # ------------------------------------------------------------------
    # FECHAMENTO
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Fecha os recursos somente quando a própria
        KnowledgeBase criou a VectorMemory.

        Se VectorMemory foi injetada externamente,
        sua responsabilidade permanece com o chamador.
        """

        if self._owns_vector_memory:
            self.vector_memory.close()

    # ------------------------------------------------------------------
    # CONTEXT MANAGER
    # ------------------------------------------------------------------

    def __enter__(
        self,
    ) -> "KnowledgeBase":
        """
        Permite:

        with KnowledgeBase(...) as knowledge_base:
            ...
        """

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        """
        Finaliza a KnowledgeBase ao sair do contexto.
        """

        self.close()