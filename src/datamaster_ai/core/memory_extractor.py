from typing import Optional

from langchain_ollama import ChatOllama

from datamaster_ai.config.settings import settings
from datamaster_ai.core.structured_memory import StructuredMemory


class MemoryExtractor:
    """
    Identifica informações importantes nas mensagens do usuário
    e as armazena na memória estruturada.
    """

    def __init__(
        self,
        memory: Optional[StructuredMemory] = None,
    ) -> None:
        self.memory = memory or StructuredMemory()

        self.model = ChatOllama(
            model=settings.DEFAULT_MODEL,
            base_url="http://localhost:11434",
            temperature=0,
            num_ctx=2048,
            num_predict=256,
        )

    def extract_and_save(self, message: str) -> list[dict]:
        """
        Analisa uma mensagem e identifica informações persistentes.

        Retorna uma lista com as informações armazenadas.
        """

        prompt = f"""
Você é um extrator de memória para o Raphael-GSilva DataMaster AI.

Sua tarefa é identificar SOMENTE informações importantes e persistentes
sobre projetos, tecnologias, ferramentas, preferências de desenvolvimento
e configurações técnicas mencionadas pelo usuário.

Mensagem do usuário:
{message}

Responda EXATAMENTE neste formato:

CATEGORY|KEY|VALUE

Se houver mais de uma informação, coloque uma informação por linha.

Se não houver nenhuma informação importante para armazenar, responda:

NONE

Categorias permitidas:

project
technology
tool
preference
configuration
workflow

Regras:

1. Não invente informações.
2. Não transforme perguntas em informações.
3. Não armazene informações temporárias.
4. Não armazene a própria resposta do assistente.
5. Use valores curtos e objetivos.
6. Não utilize Markdown.
7. Não coloque explicações.
"""

        response = self.model.invoke(prompt)

        content = response.content.strip()

        if not content or content.upper() == "NONE":
            return []

        saved_items: list[dict] = []

        for line in content.splitlines():
            line = line.strip()

            if not line or line.upper() == "NONE":
                continue

            parts = line.split("|", 2)

            if len(parts) != 3:
                continue

            category = parts[0].strip()
            key = parts[1].strip()
            value = parts[2].strip()

            allowed_categories = {
                "project",
                "technology",
                "tool",
                "preference",
                "configuration",
                "workflow",
            }

            if category not in allowed_categories:
                continue

            if not key or not value:
                continue

            self.memory.save(
                category=category,
                key=key,
                value=value,
            )

            saved_items.append(
                {
                    "category": category,
                    "key": key,
                    "value": value,
                }
            )

        return saved_items