from pathlib import Path
import sqlite3
from typing import Optional

from datamaster_ai.config.settings import settings


class StructuredMemory:
    """
    Memória estruturada do Raphael-GSilva DataMaster AI.

    Armazena informações importantes e persistentes
    relacionadas ao usuário, projetos, preferências e contexto.
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
        Cria a tabela de memória estruturada.
        """
        with self._get_connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS structured_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(category, key)
                )
                """
            )

            connection.commit()

    def save(
        self,
        category: str,
        key: str,
        value: str,
    ) -> None:
        """
        Salva ou atualiza uma informação estruturada.
        """
        with self._get_connection() as connection:
            connection.execute(
                """
                INSERT INTO structured_memory (
                    category,
                    key,
                    value
                )
                VALUES (?, ?, ?)

                ON CONFLICT(category, key)
                DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (category, key, value),
            )

            connection.commit()

    def get(
        self,
        category: str,
        key: str,
    ) -> Optional[str]:
        """
        Recupera uma informação específica.
        """
        with self._get_connection() as connection:
            row = connection.execute(
                """
                SELECT value
                FROM structured_memory
                WHERE category = ?
                AND key = ?
                """,
                (category, key),
            ).fetchone()

        if row is None:
            return None

        return str(row["value"])

    def get_category(
        self,
        category: str,
    ) -> list[dict]:
        """
        Recupera todas as informações de uma categoria.
        """
        with self._get_connection() as connection:
            rows = connection.execute(
                """
                SELECT key, value, created_at, updated_at
                FROM structured_memory
                WHERE category = ?
                ORDER BY key
                """,
                (category,),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_all(self) -> list[dict]:
        """
        Recupera toda a memória estruturada.
        """
        with self._get_connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    category,
                    key,
                    value,
                    created_at,
                    updated_at
                FROM structured_memory
                ORDER BY category, key
                """
            ).fetchall()

        return [dict(row) for row in rows]

    def delete(
        self,
        category: str,
        key: str,
    ) -> None:
        """
        Remove uma informação específica.
        """
        with self._get_connection() as connection:
            connection.execute(
                """
                DELETE FROM structured_memory
                WHERE category = ?
                AND key = ?
                """,
                (category, key),
            )

            connection.commit()

    def clear_category(
        self,
        category: str,
    ) -> None:
        """
        Remove todas as informações de uma categoria.
        """
        with self._get_connection() as connection:
            connection.execute(
                """
                DELETE FROM structured_memory
                WHERE category = ?
                """,
                (category,),
            )

            connection.commit()