import ast
import re
from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from datamaster_ai.agents.task_planner import (
    TaskPlanner,
)
from datamaster_ai.agents.tool_executor import (
    ToolExecutor,
)
from datamaster_ai.agents.tool_router import (
    ToolRouter,
)
from datamaster_ai.config.settings import settings
from datamaster_ai.core.knowledge_base import (
    KnowledgeBase,
)
from datamaster_ai.core.memory import MemoryManager
from datamaster_ai.core.memory_extractor import (
    MemoryExtractor,
)
from datamaster_ai.core.structured_memory import (
    StructuredMemory,
)
from datamaster_ai.core.vector_memory import (
    VectorMemory,
)
from datamaster_ai.tools.registry import (
    DataMasterToolRegistry,
)


class AgentState(TypedDict):
    message: str
    response: str
    context: str
    vector_context: str
    knowledge_context: str
    tool_result: str
    tool_name: str
    plan: str


class DataMasterGraph:
    """
    Grafo principal do Raphael-GSilva DataMaster AI.

    Fluxo:

    - memória conversacional;
    - memória estruturada;
    - memória vetorial;
    - base de conhecimento;
    - roteamento de ferramentas;
    - planejamento;
    - execução;
    - geração da resposta;
    - persistência das memórias.
    """

    def __init__(self) -> None:
        self.model = ChatOllama(
            model=settings.DEFAULT_MODEL,
            base_url="http://localhost:11434",
            temperature=0,
            num_ctx=2048,
            num_predict=512,
        )

        self.memory = (
            MemoryManager()
        )

        self.structured_memory = (
            StructuredMemory()
        )

        self.memory_extractor = (
            MemoryExtractor(
                memory=(
                    self.structured_memory
                )
            )
        )

        self.vector_memory = (
            VectorMemory()
        )

        self.knowledge_base = (
            KnowledgeBase(
                vector_memory=(
                    self.vector_memory
                )
            )
        )

        self.tools = (
            DataMasterToolRegistry()
        )

        self.router = (
            ToolRouter()
        )

        self.planner = (
            TaskPlanner()
        )

        self.executor = (
            ToolExecutor(
                self.tools
            )
        )

        self.graph = (
            self._build_graph()
        )

    def _load_memory(
        self,
        state: AgentState,
    ) -> AgentState:
        """
        Carrega memória conversacional
        e memória estruturada.
        """

        memories = (
            self.memory.get_messages(
                limit=5
            )
        )

        context_parts = []

        if memories:
            context_lines = []

            for memory in memories:
                context_lines.append(
                    f"{memory['role']}: "
                    f"{memory['content']}"
                )

            context_parts.append(
                "MEMÓRIA CONVERSACIONAL:\n"
                + "\n\n".join(
                    context_lines
                )
            )

        structured_items = (
            self.structured_memory.get_all()
        )

        if structured_items:
            structured_lines = []

            for item in structured_items:
                structured_lines.append(
                    f"{item['category']} | "
                    f"{item['key']} | "
                    f"{item['value']}"
                )

            context_parts.append(
                "MEMÓRIA ESTRUTURADA:\n"
                + "\n".join(
                    structured_lines
                )
            )

        if context_parts:
            context = "\n\n".join(
                context_parts
            )
        else:
            context = (
                "Nenhuma memória anterior disponível."
            )

        return {
            "message": state["message"],
            "response": "",
            "context": context,
            "vector_context": "",
            "knowledge_context": "",
            "tool_result": "",
            "tool_name": "none",
            "plan": "",
        }

    def _load_vector_memory(
        self,
        state: AgentState,
    ) -> AgentState:
        """
        Consulta memória vetorial semântica.
        """

        try:
            results = (
                self.vector_memory.search(
                    state["message"]
                )
            )
        except Exception:
            results = []

        if results:
            lines = []

            for result in results:
                content = result.get(
                    "content",
                    "",
                )

                if content:
                    lines.append(
                        f"- {content}"
                    )

            if lines:
                vector_context = (
                    "MEMÓRIA VETORIAL RELEVANTE:\n"
                    + "\n".join(
                        lines
                    )
                )
            else:
                vector_context = (
                    "Nenhuma memória vetorial "
                    "relevante encontrada."
                )
        else:
            vector_context = (
                "Nenhuma memória vetorial "
                "relevante encontrada."
            )

        return {
            "message": state["message"],
            "response": state.get(
                "response",
                "",
            ),
            "context": state.get(
                "context",
                "",
            ),
            "vector_context": (
                vector_context
            ),
            "knowledge_context": (
                state.get(
                    "knowledge_context",
                    "",
                )
            ),
            "tool_result": state.get(
                "tool_result",
                "",
            ),
            "tool_name": state.get(
                "tool_name",
                "none",
            ),
            "plan": state.get(
                "plan",
                "",
            ),
        }

    def _load_knowledge_base(
        self,
        state: AgentState,
    ) -> AgentState:
        """
        Recupera documentos relevantes
        da Knowledge Base.
        """

        try:
            results = (
                self.knowledge_base.search(
                    state["message"],
                    limit=5,
                )
            )
        except Exception:
            results = []

        if results:
            lines = []

            for result in results:
                content = result.get(
                    "content",
                    "",
                )

                metadata = result.get(
                    "metadata",
                    {},
                )

                source = metadata.get(
                    "filename",
                    metadata.get(
                        "source",
                        "desconhecida",
                    ),
                )

                if content:
                    lines.append(
                        f"[Fonte: {source}]\n"
                        f"{content}"
                    )

            if lines:
                knowledge_context = (
                    "BASE DE CONHECIMENTO "
                    "RELEVANTE:\n\n"
                    + "\n\n".join(
                        lines
                    )
                )
            else:
                knowledge_context = (
                    "Nenhum conhecimento "
                    "relevante encontrado."
                )
        else:
            knowledge_context = (
                "Nenhum conhecimento "
                "relevante encontrado."
            )

        return {
            "message": state["message"],
            "response": state["response"],
            "context": state["context"],
            "vector_context": (
                state["vector_context"]
            ),
            "knowledge_context": (
                knowledge_context
            ),
            "tool_result": (
                state["tool_result"]
            ),
            "tool_name": (
                state["tool_name"]
            ),
            "plan": state["plan"],
        }

    def _detect_tool(
        self,
        state: AgentState,
    ) -> AgentState:
        tool_name = self.router.detect(
            state["message"]
        )

        return {
            **state,
            "response": "",
            "tool_result": "",
            "tool_name": tool_name,
            "plan": "",
        }

    def _create_plan(
        self,
        state: AgentState,
    ) -> AgentState:
        task_plan = (
            self.planner.create_plan(
                state["message"],
                state["tool_name"],
            )
        )

        return {
            **state,
            "response": "",
            "tool_result": "",
            "plan": (
                task_plan.instruction
            ),
        }

    def _execute_tool(
        self,
        state: AgentState,
    ) -> AgentState:
        tool_name = state[
            "tool_name"
        ]

        if tool_name == "file":
            tool_result = (
                self._execute_file_tool(
                    state["message"]
                )
            )

        elif tool_name == "python":
            tool_result = (
                self._execute_python_tool(
                    state["message"]
                )
            )

        else:
            tool_result = ""

        return {
            **state,
            "response": "",
            "tool_result": tool_result,
        }

    def _execute_file_tool(
        self,
        message: str,
    ) -> str:
        match = re.search(
            r"([A-Za-z0-9_.-]+\.[A-Za-z0-9]+)",
            message,
            re.IGNORECASE,
        )

        if not match:
            return (
                "Não foi possível identificar "
                "o arquivo solicitado."
            )

        filename = match.group(1)

        workspace_file = (
            settings.WORKSPACE_DIR
            / filename
        )

        if not workspace_file.is_file():
            return (
                f"Arquivo '{filename}' "
                "não encontrado no Workspace.\n"
                f"Workspace: "
                f"{settings.WORKSPACE_DIR}"
            )

        result = (
            self.executor.execute(
                "file",
                filename,
            )
        )

        return str(
            result
        )

    def _execute_python_tool(
        self,
        message: str,
    ) -> str:
        code = (
            self._extract_python_code(
                message
            )
        )

        result = (
            self.executor.execute(
                "python",
                code,
            )
        )

        return str(
            result
        )

    def _extract_python_code(
        self,
        message: str,
    ) -> str:
        code_match = re.search(
            r"```python\s*(.*?)```",
            message,
            re.IGNORECASE
            | re.DOTALL,
        )

        if code_match:
            return (
                code_match.group(
                    1
                ).strip()
            )

        expression_match = re.search(
            r"(\d+(?:\.\d+)?\s*"
            r"[\+\-\*/\%\^]\s*"
            r"\d+(?:\.\d+)?)",
            message,
        )

        if expression_match:
            expression = (
                expression_match.group(
                    1
                )
            )

            expression = (
                expression.replace(
                    "^",
                    "**",
                )
            )

            return (
                "resultado = "
                f"{expression}\n"
                "print(resultado)"
            )

        return message

    @staticmethod
    def _extract_python_stdout(
        tool_result: str,
    ) -> str:
        try:
            parsed = ast.literal_eval(
                tool_result
            )
        except (
            ValueError,
            SyntaxError,
        ):
            return tool_result

        if not isinstance(
            parsed,
            dict,
        ):
            return tool_result

        if not parsed.get(
            "success",
            False,
        ):
            return tool_result

        stdout = parsed.get(
            "stdout",
            "",
        )

        return str(
            stdout
        ).strip()

    def _process_message(
        self,
        state: AgentState,
    ) -> AgentState:
        if (
            state["tool_name"]
            == "python"
        ):
            response = (
                self._extract_python_stdout(
                    state[
                        "tool_result"
                    ]
                )
            )

            return {
                **state,
                "response": response,
            }

        if (
            state["tool_name"]
            == "file"
        ):
            return {
                **state,
                "response": (
                    state[
                        "tool_result"
                    ]
                ),
            }

        prompt = f"""
Você é o Raphael-GSilva DataMaster AI.

Você é um assistente pessoal especializado em:

- Python
- Engenharia de Dados
- Machine Learning
- Inteligência Artificial
- LangChain
- LangGraph
- análise e tratamento de dados
- desenvolvimento de software
- revisão e análise de código
- projetos de tecnologia

MEMÓRIA CONVERSACIONAL E ESTRUTURADA:

{state["context"]}

MEMÓRIA VETORIAL:

{state["vector_context"]}

BASE DE CONHECIMENTO / RAG:

{state["knowledge_context"]}

PLANO:

{state["plan"]}

NOVA MENSAGEM DO USUÁRIO:

{state["message"]}

Regras:

1. Utilize as memórias quando forem relevantes.
2. Priorize a Base de Conhecimento quando ela contiver
   informações relacionadas à pergunta.
3. Não invente informações ausentes no contexto.
4. Não trate informações irrelevantes como fatos.
5. Responda diretamente à solicitação.
6. Quando utilizar a Base de Conhecimento, mantenha
   fidelidade ao conteúdo recuperado.
7. Responda em português.
8. Seja objetivo, técnico e útil.
"""

        response = (
            self.model.invoke(
                prompt
            )
        )

        return {
            **state,
            "response": (
                response.content
            ),
        }

    def _save_memory(
        self,
        state: AgentState,
    ) -> AgentState:
        self.memory.save_message(
            role="user",
            content=state[
                "message"
            ],
        )

        self.memory.save_message(
            role="assistant",
            content=state[
                "response"
            ],
        )

        try:
            (
                self.memory_extractor
                .extract_and_save(
                    state[
                        "message"
                    ]
                )
            )
        except Exception:
            pass

        try:
            self.vector_memory.add(
                state["message"],
                metadata={
                    "source": (
                        "conversation"
                    ),
                    "type": "memory",
                },
            )
        except Exception:
            pass

        return state

    def _build_graph(
        self,
    ):
        workflow = StateGraph(
            AgentState
        )

        workflow.add_node(
            "load_memory",
            self._load_memory,
        )

        workflow.add_node(
            "load_vector_memory",
            self._load_vector_memory,
        )

        workflow.add_node(
            "load_knowledge_base",
            self._load_knowledge_base,
        )

        workflow.add_node(
            "detect_tool",
            self._detect_tool,
        )

        workflow.add_node(
            "create_plan",
            self._create_plan,
        )

        workflow.add_node(
            "execute_tool",
            self._execute_tool,
        )

        workflow.add_node(
            "process_message",
            self._process_message,
        )

        workflow.add_node(
            "save_memory",
            self._save_memory,
        )

        workflow.add_edge(
            START,
            "load_memory",
        )

        workflow.add_edge(
            "load_memory",
            "load_vector_memory",
        )

        workflow.add_edge(
            "load_vector_memory",
            "load_knowledge_base",
        )

        workflow.add_edge(
            "load_knowledge_base",
            "detect_tool",
        )

        workflow.add_edge(
            "detect_tool",
            "create_plan",
        )

        workflow.add_edge(
            "create_plan",
            "execute_tool",
        )

        workflow.add_edge(
            "execute_tool",
            "process_message",
        )

        workflow.add_edge(
            "process_message",
            "save_memory",
        )

        workflow.add_edge(
            "save_memory",
            END,
        )

        return workflow.compile()

    def invoke(
        self,
        message: str,
    ) -> str:
        result = self.graph.invoke(
            {
                "message": message,
                "response": "",
                "context": "",
                "vector_context": "",
                "knowledge_context": "",
                "tool_result": "",
                "tool_name": "none",
                "plan": "",
            }
        )

        return result[
            "response"
        ]


def create_graph() -> DataMasterGraph:
    """
    Cria o grafo principal.
    """

    return DataMasterGraph()