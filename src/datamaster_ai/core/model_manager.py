from langchain_ollama import ChatOllama


class ModelManager:
    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        temperature: float = 0.2,
    ):
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
        )

    def ask(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content