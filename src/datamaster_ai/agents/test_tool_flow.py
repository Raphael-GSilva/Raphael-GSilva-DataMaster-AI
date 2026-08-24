from datamaster_ai.agents.task_planner import TaskPlanner
from datamaster_ai.agents.tool_router import ToolRouter
from datamaster_ai.agents.tool_executor import ToolExecutor
from datamaster_ai.tools.registry import DataMasterToolRegistry


def test_tool_router_python() -> None:
    router = ToolRouter()

    assert router.detect("calcule 125 * 48") == "python"


def test_tool_router_file() -> None:
    router = ToolRouter()

    assert router.detect("leia o arquivo teste_file_tool.txt") == "file"


def test_tool_router_none() -> None:
    router = ToolRouter()

    assert router.detect("Explique o que é LangGraph") == "none"


def test_task_planner_python() -> None:
    planner = TaskPlanner()

    plan = planner.create_plan(
        "calcule 125 * 48",
        "python",
    )

    assert plan.tool_name == "python"
    assert "Python" in plan.instruction


def test_task_planner_file() -> None:
    planner = TaskPlanner()

    plan = planner.create_plan(
        "leia o arquivo teste_file_tool.txt",
        "file",
    )

    assert plan.tool_name == "file"


def test_tool_executor_python() -> None:
    registry = DataMasterToolRegistry()
    executor = ToolExecutor(registry)

    result = executor.execute(
        "python",
        "print(125 * 48)",
    )

    assert result is not None
    assert "6000" in str(result)