import sqlite3
from pathlib import Path

from datamaster_ai.config.settings import settings


class DatabaseManager:
    """
    Gerencia o banco SQLite utilizado pela memória do agente.
    """

    def __init__(self) -> None:
        settings.MEMORY_DIR.mkdir(parents=True, exist_ok=True)

        self.database_path: Path = settings.SQLITE_DATABASE

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    def initialize(self) -> None:
        connection = self.connect()

        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        connection.commit()
        connection.close()