from rich.console import Console
from rich.panel import Panel

from datamaster_ai.config.settings import settings
from datamaster_ai.core.logger import setup_logger
from datamaster_ai.core.startup import startup_checks
from datamaster_ai.memory.database import DatabaseManager
from datamaster_ai.memory.service import MemoryService

console = Console()


class SystemManager:
    """
    Gerencia toda a inicialização do Raphael-GSilva DataMaster AI.
    """

    def __init__(self) -> None:
        self.database = DatabaseManager()
        self.memory = MemoryService()

    def start(self) -> None:
        setup_logger()

        startup_checks()

        self.database.initialize()

        self.memory.save_message(
            role="system",
            content="Core Engine iniciado com sucesso."
        )

        total_messages = len(self.memory.get_history())

        console.print(
            Panel.fit(
                (
                    f"[bold green]{settings.APP_NAME}[/bold green]\n\n"
                    f"Versão: {settings.VERSION}\n"
                    f"Ambiente: {settings.ENVIRONMENT}\n"
                    f"Modelo padrão: {settings.DEFAULT_MODEL}\n"
                    f"Banco de memória: {settings.SQLITE_DATABASE.name}\n"
                    f"Mensagens armazenadas: {total_messages}\n\n"
                    "Core Engine iniciado com sucesso."
                ),
                title="Core Engine",
            )
        )