from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from datamaster_ai.config.settings import settings


class VectorMemory:
    """
    Memória vetorial do Raphael-GSilva DataMaster AI.

    Utiliza:
    - ChromaDB para armazenamento vetorial;
    - SentenceTransformers para geração de embeddings.
    """

    def __init__(
        self,
        database_path: Optional[Path] = None,
    ) -> None:
        self.database_path = (
            database_path
            or settings.VECTOR_DB_DIR
        )

        self.database_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=str(self.database_path)
        )

        self.collection = self.client.get_or_create_collection(
            name="datamaster_memory"
        )

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def add(
        self,
        content: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Adiciona uma memória à base vetorial.

        Retorna o ID do registro criado.
        """

        if not content or not content.strip():
            raise ValueError(
                "O conteúdo da memória não pode estar vazio."
            )

        if metadata is None:
            metadata = {
                "source": "datamaster_ai"
            }

        if not metadata:
            metadata = {
                "source": "datamaster_ai"
            }

        memory_id = (
            f"memory_{self.collection.count() + 1}"
        )

        embedding = self.embedding_model.encode(
            content
        ).tolist()

        self.collection.add(
            ids=[memory_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata],
        )

        return memory_id

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """
        Busca memórias semanticamente semelhantes.
        """

        if not query or not query.strip():
            return []

        query_embedding = (
            self.embedding_model.encode(
                query
            ).tolist()
        )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
        )

        documents = (
            results.get("documents", [[]])[0]
        )

        ids = (
            results.get("ids", [[]])[0]
        )

        metadatas = (
            results.get("metadatas", [[]])[0]
        )

        distances = (
            results.get("distances", [[]])[0]
        )

        memories = []

        for index, document in enumerate(documents):
            memories.append(
                {
                    "id": ids[index],
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

        return memories

    def count(self) -> int:
        """
        Retorna a quantidade de memórias armazenadas.
        """

        return self.collection.count()

    def clear(self) -> None:
        """
        Remove todas as memórias vetoriais.
        """

        existing_ids = self.collection.get().get(
            "ids",
            []
        )

        if existing_ids:
            self.collection.delete(
                ids=existing_ids
            )