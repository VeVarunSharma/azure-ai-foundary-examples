from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import agent_runtime


@pytest.fixture
def agent_config() -> agent_runtime.AgentConfig:
    return agent_runtime.AgentConfig(
        name="test-agent",
        instructions="Follow the user input exactly.",
        tools=None,
        model_deployment="test-model",
    )


def _install_project_client(monkeypatch: pytest.MonkeyPatch, project_client: MagicMock) -> None:
    @contextmanager
    def _fake_context():
        yield project_client

    monkeypatch.setattr(agent_runtime, "project_client_context", _fake_context)


def _make_agent_client() -> MagicMock:
    project_client = MagicMock()
    agents = project_client.agents
    agents.create_agent.return_value = SimpleNamespace(id="agent-123", name="test-agent")
    agents.threads.create.return_value = SimpleNamespace(id="thread-456")
    agents.runs.create_and_process.return_value = SimpleNamespace(id="run-789", status="completed")
    agents.runs.create.return_value = SimpleNamespace(id="run-tool")
    agents.messages.list.return_value = [SimpleNamespace(role="assistant", content="All done", image_contents=[])]
    agents.messages.create.return_value = None
    agents.delete_agent.return_value = None
    return project_client


def test_run_agent_interaction_requires_user_input(agent_config: agent_runtime.AgentConfig) -> None:
    with pytest.raises(ValueError):
        agent_runtime.run_agent_interaction(config=agent_config, user_input="")


def test_run_agent_interaction_processes_run(monkeypatch: pytest.MonkeyPatch, agent_config: agent_runtime.AgentConfig) -> None:
    project_client = _make_agent_client()
    _install_project_client(monkeypatch, project_client)

    post_run_hook = MagicMock()

    result = agent_runtime.run_agent_interaction(
        config=agent_config,
        user_input="Hello",
        post_run_hook=post_run_hook,
        auto_delete_agent=True,
    )

    assert result.run_status == "completed"
    assert result.thread_id == "thread-456"
    assert len(result.messages) == 1
    post_run_hook.assert_called_once()
    project_client.agents.delete_agent.assert_called_once_with("agent-123")


def test_run_agent_interaction_handles_tool_runs(
    monkeypatch: pytest.MonkeyPatch, agent_config: agent_runtime.AgentConfig
) -> None:
    project_client = _make_agent_client()
    project_client.agents.runs.create.return_value = SimpleNamespace(id="run-with-tools")
    project_client.agents.runs.create_and_process.side_effect = AssertionError(
        "create_and_process should not be called when handle_tool_calls is provided"
    )
    project_client.agents.threads.create.return_value = SimpleNamespace(id="thread-tools")
    _install_project_client(monkeypatch, project_client)

    poll_result = SimpleNamespace(id="run-with-tools", status="completed")
    poll_spy = MagicMock(return_value=poll_result)
    monkeypatch.setattr(agent_runtime, "_poll_run_with_tools", poll_spy)

    handle_tool_calls = MagicMock()

    result = agent_runtime.run_agent_interaction(
        config=agent_config,
        user_input="Please run tools",
        handle_tool_calls=handle_tool_calls,
        poll_interval_seconds=0.25,
    )

    assert result.run_id == "run-with-tools"
    poll_spy.assert_called_once_with(
        project_client=project_client,
        thread_id="thread-tools",
        run_id="run-with-tools",
        handle_tool_calls=handle_tool_calls,
        poll_interval=0.25,
    )


def test_poll_run_with_tools_processes_until_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    project_client = MagicMock()

    required_action = SimpleNamespace(
        submit_tool_outputs=SimpleNamespace(
            tool_calls=[SimpleNamespace(id="call-1"), SimpleNamespace(id="call-2")]
        )
    )

    requires_action_run = SimpleNamespace(
        id="run-1",
        status="requires_action",
        required_action=required_action,
    )
    in_progress_run = SimpleNamespace(id="run-1", status="in_progress")
    completed_run = SimpleNamespace(id="run-1", status="completed")

    project_client.agents.runs.get.side_effect = [
        requires_action_run,
        in_progress_run,
        completed_run,
    ]

    project_client.agents.runs.submit_tool_outputs.return_value = None

    outputs = [{"tool_call_id": "call-1", "output": "42"}]
    handle_tool_calls = MagicMock(return_value=outputs)

    monkeypatch.setattr(agent_runtime.time, "sleep", lambda *_: None)

    result = agent_runtime._poll_run_with_tools(
        project_client=project_client,
        thread_id="thread-abc",
        run_id="run-1",
        handle_tool_calls=handle_tool_calls,
        poll_interval=0.0,
    )

    assert result is completed_run
    handle_tool_calls.assert_called_once_with(project_client, required_action.submit_tool_outputs.tool_calls)
    project_client.agents.runs.submit_tool_outputs.assert_called_once_with(
        thread_id="thread-abc",
        run_id="run-1",
        tool_outputs=outputs,
    )


def test_print_thread_messages_writes_human_readable_output(capsys: pytest.CaptureFixture[str]) -> None:
    message = SimpleNamespace(role="assistant", content="All good", image_contents=["img-bytes"])
    result = agent_runtime.AgentRunResult(
        agent_id="agent-1",
        agent_name="helper",
        thread_id="thread-xyz",
        run_id="run-123",
        run_status="completed",
        messages=[message],
    )

    agent_runtime.print_thread_messages(result)

    output = capsys.readouterr().out
    assert "Run run-123" in output
    assert "[assistant] All good" in output
    assert "image attachment" in output
