from pathlib import Path
import sqlite3
from typing import Optional


from datamaster_ai.config.settings import settings


class MemoryManager:
    """
    Gerenciador de memória persistente do Raphael-GSilva DataMaster AI.
    """

    def __init__(self, database_path: Optional[Path] = None) -> None:
        self.database_path = database_path or settings.SQLITE_DATABASE

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_database()

    def _get_connection(self) -> sqlite3.Connection:
        """
        Cria uma conexão com o banco SQLite.
        """
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row

        return connection

    def _initialize_database(self) -> None:
        """
        Cria a estrutura inicial do banco de memória.
        """
        with self._get_connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            connection.commit()

    def save_message(
        self,
        role: str = "user",
        content: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        """
        Salva uma mensagem na memória.

        Aceita tanto:
            save_message(role="user", content="...")
        quanto:
            save_message(message="...")
        """
        if message is not None:
            content = message

            if role == "user":
                role = "assistant"

        if content is None:
            raise ValueError(
                "É necessário informar 'content' ou 'message'."
            )

        with self._get_connection() as connection:
            connection.execute(
                """
                INSERT INTO messages (role, content)
                VALUES (?, ?)
                """,
                (role, content),
            )

            connection.commit()

    def get_messages(self, limit: int = 20) -> list[dict]:
        """
        Recupera as mensagens mais recentes.
        """
        with self._get_connection() as connection:
            rows = connection.execute(
                """
                SELECT id, role, content, created_at
                FROM messages
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [dict(row) for row in reversed(rows)]

    def get_recent_messages(self, limit: int = 20) -> list[dict]:
        """
        Recupera as mensagens mais recentes
        para utilização pelos agentes.
        """
        return self.get_messages(limit=limit)

    def count_messages(self) -> int:
        """
        Retorna a quantidade total de mensagens armazenadas.
        """
        with self._get_connection() as connection:
            result = connection.execute(
                """
                SELECT COUNT(*) AS total
                FROM messages
                """
            ).fetchone()

        return int(result["total"])

    def clear(self) -> None:
        """
        Limpa todas as mensagens da memória.
        """
        with self._get_connection() as connection:
            connection.execute(
                "DELETE FROM messages"
            )

            connection.commit()