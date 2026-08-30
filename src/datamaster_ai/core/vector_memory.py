from __future__ import annotations

import gc
import uuid
from pathlib import Path
from typing import Any

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings


class VectorMemory:
    """
    Memória vetorial persistente do Raphael-GSilva DataMaster AI.

    Responsabilidades:

    - gerar embeddings dos textos;
    - armazenar textos no ChromaDB;
    - realizar busca semântica;
    - manter persistência em disco;
    - permitir limpeza da coleção;
    - liberar recursos do ChromaDB;
    - funcionar corretamente no Windows;
    - fornecer interface simples para o LangGraph.
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

        Args:
            persist_directory:
                Diretório onde o ChromaDB armazenará os dados.

            collection_name:
                Nome da coleção vetorial.

            embedding_model:
                Modelo Sentence Transformers utilizado
                para gerar embeddings.

            database_path:
                Alias compatível com os testes e versões
                anteriores do projeto.
        """

        if (
            persist_directory is not None
            and database_path is not None
        ):
            persist_path = Path(
                persist_directory
            ).resolve()

            database_path_resolved = Path(
                database_path
            ).resolve()

            if persist_path != database_path_resolved:
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

        self._closed = False

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
        )

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=chromadb.Settings(
                anonymized_telemetry=False,
            ),
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": (
                        "Memoria vetorial "
                        "do Raphael-GSilva DataMaster AI"
                    )
                },
            )
        )

    # ------------------------------------------------------------------
    # UTILITÁRIOS
    # ------------------------------------------------------------------

    def _ensure_open(self) -> None:
        """
        Garante que a memória ainda está aberta.
        """

        if self._closed:
            raise RuntimeError(
                "VectorMemory já foi encerrada."
            )

    def _generate_id(self) -> str:
        """
        Gera identificador único para cada memória.
        """

        return str(uuid.uuid4())

    # ------------------------------------------------------------------
    # EMBEDDINGS
    # ------------------------------------------------------------------

    def _embed_documents(
        self,
        documents: list[str],
    ) -> list[list[float]]:
        """
        Gera embeddings para documentos.

        O método utiliza diretamente o objeto de embeddings
        do LangChain para manter a geração de vetores
        independente do mecanismo interno do ChromaDB.
        """

        if not documents:
            return []

        embeddings = self.embeddings.embed_documents(
            documents
        )

        return [
            [float(value) for value in embedding]
            for embedding in embeddings
        ]

    def _embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Gera o embedding de uma consulta.
        """

        embedding = self.embeddings.embed_query(
            query
        )

        return [
            float(value)
            for value in embedding
        ]

    # ------------------------------------------------------------------
    # ADICIONAR MEMÓRIA
    # ------------------------------------------------------------------

    def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Adiciona uma memória à coleção vetorial.

        Args:
            content:
                Texto que será armazenado.

            metadata:
                Metadados opcionais.

        Returns:
            ID da memória criada.
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
                    (str, int, float, bool),
                ):
                    final_metadata[str(key)] = value
                else:
                    final_metadata[str(key)] = str(
                        value
                    )

        embedding = self._embed_documents(
            [content]
        )[0]

        self.collection.add(
            ids=[memory_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[final_metadata],
        )

        return memory_id

    # ------------------------------------------------------------------
    # BUSCA SEMÂNTICA
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Realiza busca semântica na memória vetorial.

        Args:
            query:
                Texto utilizado como consulta.

            limit:
                Número máximo de resultados.

        Returns:
            Lista de memórias relevantes.
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

        query_embedding = self._embed_query(
            query
        )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(limit, count),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

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

        output: list[dict[str, Any]] = []

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
                        if index < len(metadatas)
                        else {}
                    ),
                    "distance": (
                        distances[index]
                        if index < len(distances)
                        else None
                    ),
                }
            )

        return output

    # ------------------------------------------------------------------
    # CONTAGEM
    # ------------------------------------------------------------------

    def count(self) -> int:
        """
        Retorna a quantidade de memórias armazenadas.
        """

        self._ensure_open()

        return self.collection.count()

    # ------------------------------------------------------------------
    # LIMPAR
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """
        Remove todas as memórias da coleção.

        A coleção é recriada imediatamente para que
        o objeto continue utilizável.
        """

        self._ensure_open()

        self.client.delete_collection(
            name=self.collection_name
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": (
                        "Memoria vetorial "
                        "do Raphael-GSilva DataMaster AI"
                    )
                },
            )
        )

    # ------------------------------------------------------------------
    # FECHAMENTO
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Libera os recursos utilizados pelo ChromaDB.

        O ChromaDB 1.5.9 utiliza um sistema interno que
        gerencia os recursos persistentes. Não acessamos
        atributos privados como _server ou _system.
        """

        if self._closed:
            return

        client = getattr(
            self,
            "client",
            None,
        )

        self.collection = None

        if client is not None:
            try:
                close_method = getattr(
                    client,
                    "close",
                    None,
                )

                if callable(close_method):
                    close_method()

            except Exception:
                pass

        self.client = None
        self.embeddings = None
        self._closed = True

        gc.collect()

    # ------------------------------------------------------------------
    # CONTEXT MANAGER
    # ------------------------------------------------------------------

    def __enter__(
        self,
    ) -> "VectorMemory":
        """
        Permite utilização com:

            with VectorMemory(...) as memory:
                ...
        """

        self._ensure_open()

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        """
        Fecha automaticamente a memória.
        """

        self.close()

    # ------------------------------------------------------------------
    # DESTRUTOR
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        """
        Tentativa final de liberar recursos.
        """

        try:
            self.close()
        except Exception:
            pass