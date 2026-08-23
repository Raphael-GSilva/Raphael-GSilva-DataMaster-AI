from datamaster_ai.core.model_manager import ModelManager


class AssistantAgent:

    def __init__(self):
        self.model = ModelManager()

    def execute(self, prompt: str) -> str:

        system_prompt = f"""
Você é Raphael-GSilva DataMaster AI.

Você é um especialista em:

- Python
- Engenharia de Dados
- Machine Learning
- Inteligência Artificial
- LangGraph
- LangChain
- SQL
- Git
- Docker
- APIs
- Automação

Sempre responda em português.

Pergunta:

{prompt}
"""

        return self.model.ask(system_prompt)