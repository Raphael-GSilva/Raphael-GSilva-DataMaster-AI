from sqlite3 import Row

from datamaster_ai.memory.database import DatabaseManager


class MemoryService:
    """
    Serviço responsável por armazenar e recuperar
    informações persistentes do agente.
    """

    def __init__(self) -> None:
        self.database = DatabaseManager()

    def save_message(
        self,
        role: str,
        content: str,
    ) -> None:
        connection = self.database.connect()

        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO conversations
            (
                role,
                content
            )
            VALUES
            (
                ?,
                ?
            )
            """,
            (
                role,
                content,
            ),
        )

        connection.commit()
        connection.close()

    def get_history(self) -> list[Row]:
        connection = self.database.connect()

        connection.row_factory = Row

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                role,
                content,
                created_at
            FROM conversations
            ORDER BY id ASC
            """
        )

        rows = cursor.fetchall()

        connection.close()

        return rows

    def clear_history(self) -> None:
        connection = self.database.connect()

        cursor = connection.cursor()

        cursor.execute(
            """
            DELETE FROM conversations
            """
        )

        connection.commit()

        connection.close()