from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from config.azure.ai_foundary_config import (
    MODEL_DEPLOYMENT_NAME,
    project_client_context,
)

TerminalStatuses = {"completed", "failed", "cancelled", "expired"}
PollingStatuses = {"queued", "in_progress"}


@dataclass(frozen=True)
class AgentConfig:
    """Declarative definition for creating an Azure AI Foundry agent."""

    name: str
    instructions: str
    tools: Optional[Sequence[Any]] = None
    model_deployment: str = MODEL_DEPLOYMENT_NAME


@dataclass(frozen=True)
class AgentRunResult:
    """Simple DTO collecting the important objects from a run."""

    agent_id: str
    agent_name: str
    thread_id: str
    run_id: str
    run_status: str
    messages: list[Any]


ToolCallHandler = Callable[[Any, Sequence[Any]], list[dict[str, Any]]]
PostRunHook = Callable[[Any, AgentRunResult], None]


def run_agent_interaction(
    *,
    config: AgentConfig,
    user_input: str,
    additional_instructions: Optional[str] = None,
    handle_tool_calls: Optional[ToolCallHandler] = None,
    poll_interval_seconds: float = 1.0,
    post_run_hook: Optional[PostRunHook] = None,
    auto_delete_agent: bool = False,
) -> AgentRunResult:
    """Create an agent, send the user input, and process the run.

    Parameters
    ----------
    config:
        Declarative agent definition (name, instructions, tools, model).
    user_input:
        The prompt or message that should be sent as the first user turn.
    additional_instructions:
        Optional instructions scoped only to this run, mirroring the SDK parameter.
    handle_tool_calls:
        Callback invoked whenever the run status is ``requires_action``.
    poll_interval_seconds:
        Delay between status polls for runs that require tool outputs.
    post_run_hook:
        Optional callback invoked *before* the client context closes. Useful for
        persisting files that require the SDK client.
    auto_delete_agent:
        When ``True``, delete the temporary agent after the run completes.
    """

    if not user_input:
        raise ValueError("user_input must not be empty")

    with project_client_context() as project_client:
        agent = project_client.agents.create_agent(
            model=config.model_deployment,
            name=config.name,
            instructions=config.instructions,
            tools=config.tools or [],
        )

        thread = project_client.agents.threads.create()

        project_client.agents.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_input,
        )

        if handle_tool_calls:
            run = project_client.agents.runs.create(
                thread_id=thread.id,
                agent_id=agent.id,
                additional_instructions=additional_instructions,
            )
            run = _poll_run_with_tools(
                project_client=project_client,
                thread_id=thread.id,
                run_id=run.id,
                handle_tool_calls=handle_tool_calls,
                poll_interval=poll_interval_seconds,
            )
        else:
            run = project_client.agents.runs.create_and_process(
                thread_id=thread.id,
                agent_id=agent.id,
                additional_instructions=additional_instructions,
            )

        messages = list(project_client.agents.messages.list(thread_id=thread.id))

        result = AgentRunResult(
            agent_id=agent.id,
            agent_name=agent.name,
            thread_id=thread.id,
            run_id=run.id,
            run_status=getattr(run, "status", "unknown"),
            messages=messages,
        )

        if post_run_hook:
            post_run_hook(project_client, result)

        if auto_delete_agent:
            project_client.agents.delete_agent(agent.id)

    return result


def _poll_run_with_tools(
    *,
    project_client: Any,
    thread_id: str,
    run_id: str,
    handle_tool_calls: ToolCallHandler,
    poll_interval: float,
):
    """Handle requires_action loops until the run reaches a terminal status."""

    while True:
        run = project_client.agents.runs.get(thread_id=thread_id, run_id=run_id)
        status = getattr(run, "status", "unknown")

        if status in TerminalStatuses:
            return run

        if status == "requires_action":
            required = getattr(run, "required_action", None)
            tool_calls = getattr(required, "submit_tool_outputs", None)
            tool_calls = getattr(tool_calls, "tool_calls", [])
            outputs = handle_tool_calls(project_client, tool_calls)
            if not outputs:
                raise RuntimeError("Run requires tool outputs but handler returned none.")
            project_client.agents.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=outputs,
            )
            continue

        if status in PollingStatuses:
            time.sleep(poll_interval)
            continue

        # Unknown intermediate status; wait a moment before re-checking
        time.sleep(poll_interval)


def print_thread_messages(result: AgentRunResult) -> None:
    """Utility helper to echo the collected conversation to stdout."""

    header = (
        f"Run {result.run_id} for agent '{result.agent_name}' (thread {result.thread_id}) "
        f"finished with status: {result.run_status}"
    )
    print(header)
    print("-" * len(header))

    for message in result.messages:
        role = getattr(message, "role", "unknown")
        content = getattr(message, "content", "")
        print(f"[{role}] {content}")
        attachments = getattr(message, "image_contents", []) or []
        if attachments:
            print(f"  ↳ {len(attachments)} image attachment(s) available.")

    print()
