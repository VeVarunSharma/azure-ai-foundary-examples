"""Utilities for interacting with the math-focused Azure AI Foundry agent."""

from pathlib import Path
from typing import Optional

from azure.ai.agents.models import CodeInterpreterTool

from utils.agent_runtime import (
    AgentConfig,
    AgentRunResult,
    run_agent_interaction,
)

code_interpreter = CodeInterpreterTool()

_DEFAULT_ADDITIONAL_INSTRUCTIONS = (
    "Please address the user as Jane Doe. The user has a premium account."
)
_IMAGE_OUTPUT_DIR = Path("tmp/images")


def run(
    user_input: str,
    *,
    additional_instructions: Optional[str] = None,
    auto_delete_agent: bool = False,
) -> AgentRunResult:
    """Execute the math agent against the provided user input."""

    config = AgentConfig(
        name="math-agent-v1",
        instructions=(
            "You politely help with math questions. Use the Code Interpreter tool "
            "when asked to visualize numbers."
        ),
        tools=code_interpreter.definitions,
    )

    return run_agent_interaction(
        config=config,
        user_input=user_input,
        additional_instructions=additional_instructions or _DEFAULT_ADDITIONAL_INSTRUCTIONS,
        post_run_hook=_save_generated_images,
        auto_delete_agent=auto_delete_agent,
    )


def _save_generated_images(project_client, result: AgentRunResult) -> None:
    """Persist any image outputs to disk for easy inspection."""

    if not _IMAGE_OUTPUT_DIR.exists():
        _IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for message in result.messages:
        attachments = getattr(message, "image_contents", []) or []
        for attachment in attachments:
            image_file = getattr(attachment, "image_file", None)
            if not image_file:
                continue
            file_id = getattr(image_file, "file_id", None)
            if not file_id:
                continue
            file_path = _IMAGE_OUTPUT_DIR / f"{file_id}_image_file.png"
            project_client.agents.files.save(
                file_id=file_id,
                file_name=str(file_path),
            )
            print(f"Saved image file to: {file_path.resolve()}")


if __name__ == "__main__":
    from utils.agent_runtime import print_thread_messages

    sample = "Hi, Agent! Draw a graph for a line with a slope of 4 and y-intercept of 9."
    print_thread_messages(run(sample))