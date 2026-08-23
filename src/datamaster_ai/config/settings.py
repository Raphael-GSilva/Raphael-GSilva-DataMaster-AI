from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    """
    Configurações globais do Raphael-GSilva DataMaster AI.
    """

    APP_NAME: str = "Raphael-GSilva DataMaster AI"
    VERSION: str = "1.0.0-dev"
    ENVIRONMENT: str = "development"

    PROJECT_ROOT: Path = Path.cwd()

    WORKSPACE_DIR: Path = PROJECT_ROOT / "01_Workspace"
    MEMORY_DIR: Path = PROJECT_ROOT / "04_Memory"
    KNOWLEDGE_BASE_DIR: Path = PROJECT_ROOT / "05_Knowledge_Base"
    VECTOR_DB_DIR: Path = PROJECT_ROOT / "06_Vector_DB"
    DATASETS_DIR: Path = PROJECT_ROOT / "07_Datasets"
    LOGS_DIR: Path = PROJECT_ROOT / "12_Logs"

    DEFAULT_MODEL: str = "qwen2.5-coder:7b"

    SQLITE_DATABASE: Path = MEMORY_DIR / "memory.db"


settings = Settings()