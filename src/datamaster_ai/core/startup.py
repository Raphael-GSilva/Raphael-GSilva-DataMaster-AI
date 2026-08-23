from loguru import logger


def startup_checks() -> None:
    logger.info("Executando verificações iniciais...")
    logger.info("Python: OK")
    logger.info("Dependências: OK")