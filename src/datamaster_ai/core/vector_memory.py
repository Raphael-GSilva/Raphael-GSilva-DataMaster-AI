from __future__ import annotations

import gc
import uuid
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings


class VectorMemory:
    """
    Memória vetorial persistente do Raphael-GSilva DataMaster AI.

    Responsabilidades:

    - gerar embeddings dos textos;
    - armazenar textos no ChromaDB;
    - realizar busca semântica;
    - manter persistência em disco;
    - permitir filtros por metadados;
    - permitir limpeza da coleção;
    - liberar recursos do ChromaDB;
    - funcionar corretamente no Windows;
    - fornecer interface para o LangGraph e Knowledge Base.
    """

    DEFAULT_COLLECTION_NAME = "datamaster_memory"

    DEFAULT_EMBEDDING_MODEL = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        database_path: str | Path | None = None,
    ) -> None:
        """
        Inicializa a memória vetorial.
        """

        if (
            persist_directory is not None
            and database_path is not None
        ):
            if (
                Path(persist_directory).resolve()
                != Path(database_path).resolve()
            ):
                raise ValueError(
                    "persist_directory e database_path "
                    "apontam para diretórios diferentes."
                )

        if database_path is not None:
            persist_directory = database_path

        if persist_directory is None:
            persist_directory = Path(
                "06_Vector_DB"
            )

        self.persist_directory = Path(
            persist_directory
        ).resolve()

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.collection_name = collection_name
        self.embedding_model = embedding_model

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
        )

        self.client = self._create_client()

        self.collection = self._create_collection()

        self._closed = False

    def _create_client(self):
        """
        Cria cliente persistente do ChromaDB.
        """

        chroma_settings = Settings(
            anonymized_telemetry=False,
            chroma_api_impl=(
                "chromadb.api.segment.SegmentAPI"
            ),
            is_persistent=True,
            persist_directory=str(
                self.persist_directory
            ),
        )

        return chromadb.Client(
            settings=chroma_settings,
        )

    def _create_collection(self):
        """
        Cria ou recupera a coleção vetorial.
        """

        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": (
                    "Memoria vetorial "
                    "do Raphael-GSilva DataMaster AI"
                )
            },
        )

    def _ensure_open(self) -> None:
        """
        Garante que a instância continua aberta.
        """

        if self._closed:
            raise RuntimeError(
                "VectorMemory já foi encerrada."
            )

    @staticmethod
    def _generate_id() -> str:
        """
        Gera identificador único.
        """

        return str(
            uuid.uuid4()
        )

    def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Adiciona uma memória à coleção vetorial.
        """

        self._ensure_open()

        if not isinstance(content, str):
            raise TypeError(
                "content deve ser uma string."
            )

        content = content.strip()

        if not content:
            raise ValueError(
                "content não pode ser vazio."
            )

        memory_id = self._generate_id()

        final_metadata: dict[str, Any] = {
            "source": "datamaster_ai",
            "type": "memory",
        }

        if metadata:
            for key, value in metadata.items():
                if value is None:
                    continue

                if isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool,
                    ),
                ):
                    final_metadata[
                        str(key)
                    ] = value
                else:
                    final_metadata[
                        str(key)
                    ] = str(value)

        self.collection.add(
            ids=[
                memory_id
            ],
            documents=[
                content
            ],
            metadatas=[
                final_metadata
            ],
        )

        return memory_id

    def search(
        self,
        query: str,
        limit: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Realiza busca semântica.

        metadata_filter permite limitar a busca por
        metadados do ChromaDB.

        Exemplo:

            {"type": "knowledge"}
        """

        self._ensure_open()

        if not isinstance(query, str):
            raise TypeError(
                "query deve ser uma string."
            )

        query = query.strip()

        if not query:
            return []

        if limit <= 0:
            return []

        count = self.collection.count()

        if count == 0:
            return []

        query_kwargs: dict[str, Any] = {
            "query_texts": [
                query
            ],
            "n_results": min(
                limit,
                count,
            ),
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if metadata_filter:
            query_kwargs[
                "where"
            ] = metadata_filter

        try:
            results = self.collection.query(
                **query_kwargs
            )
        except Exception:
            return []

        documents = results.get(
            "documents",
            [[]],
        )

        metadatas = results.get(
            "metadatas",
            [[]],
        )

        distances = results.get(
            "distances",
            [[]],
        )

        ids = results.get(
            "ids",
            [[]],
        )

        documents = (
            documents[0]
            if documents
            else []
        )

        metadatas = (
            metadatas[0]
            if metadatas
            else []
        )

        distances = (
            distances[0]
            if distances
            else []
        )

        ids = (
            ids[0]
            if ids
            else []
        )

        output: list[
            dict[str, Any]
        ] = []

        for index, document in enumerate(
            documents
        ):
            output.append(
                {
                    "id": (
                        ids[index]
                        if index < len(ids)
                        else ""
                    ),
                    "content": document,
                    "metadata": (
                        metadatas[index]
                        if index
                        < len(metadatas)
                        else {}
                    ),
                    "distance": (
                        distances[index]
                        if index
                        < len(distances)
                        else None
                    ),
                }
            )

        return output

    def count(self) -> int:
        """
        Retorna a quantidade total de registros.
        """

        self._ensure_open()

        return self.collection.count()

    def clear(self) -> None:
        """
        Remove todos os registros da coleção.
        """

        self._ensure_open()

        self.client.delete_collection(
            name=self.collection_name
        )

        self.collection = (
            self._create_collection()
        )

    def close(self) -> None:
        """
        Libera referências utilizadas pelo ChromaDB.
        """

        if self._closed:
            return

        client = self.client

        self.collection = None

        try:
            close_method = getattr(
                client,
                "close",
                None,
            )

            if callable(
                close_method
            ):
                close_method()

        except Exception:
            pass

        self.client = None
        self.embeddings = None

        self._closed = True

        gc.collect()

    def __enter__(
        self,
    ) -> "VectorMemory":
        self._ensure_open()

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass