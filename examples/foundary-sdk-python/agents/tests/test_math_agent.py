from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents import math_agent
from utils.agent_runtime import AgentConfig


def test_run_uses_default_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_result = object()
    run_spy = MagicMock(return_value=fake_result)
    monkeypatch.setattr(math_agent, "run_agent_interaction", run_spy)

    fake_definitions = ["code-interpreter-tool"]
    monkeypatch.setattr(math_agent, "code_interpreter", SimpleNamespace(definitions=fake_definitions))

    result = math_agent.run("Add 1 and 2", auto_delete_agent=True)

    assert result is fake_result

    run_spy.assert_called_once()
    kwargs = run_spy.call_args.kwargs

    assert kwargs["user_input"] == "Add 1 and 2"
    assert kwargs["additional_instructions"] == math_agent._DEFAULT_ADDITIONAL_INSTRUCTIONS
    assert kwargs["post_run_hook"] is math_agent._save_generated_images
    assert kwargs["auto_delete_agent"] is True

    config = kwargs["config"]
    assert isinstance(config, AgentConfig)
    assert config.name == "math-agent-v1"
    assert (
        config.instructions
        == "You politely help with math questions. Use the Code Interpreter tool when asked to visualize numbers."
    )
    assert config.tools == fake_definitions


def test_run_allows_custom_additional_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    run_spy = MagicMock()
    monkeypatch.setattr(math_agent, "run_agent_interaction", run_spy)
    monkeypatch.setattr(math_agent, "code_interpreter", SimpleNamespace(definitions=[]))

    custom_instructions = "Only answer with matrix operations."

    math_agent.run("Explain eigenvalues", additional_instructions=custom_instructions)

    run_spy.assert_called_once()
    kwargs = run_spy.call_args.kwargs
    assert kwargs["additional_instructions"] == custom_instructions


def test_save_generated_images_creates_directory_and_saves_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_client = MagicMock()
    project_client.agents.files.save = MagicMock()

    attachment = SimpleNamespace(image_file=SimpleNamespace(file_id="file-123"))
    message = SimpleNamespace(image_contents=[attachment])
    result = SimpleNamespace(messages=[message])

    target_dir = tmp_path / "images"
    monkeypatch.setattr(math_agent, "_IMAGE_OUTPUT_DIR", target_dir)

    math_agent._save_generated_images(project_client, result)

    expected_path = target_dir / "file-123_image_file.png"
    project_client.agents.files.save.assert_called_once_with(
        file_id="file-123",
        file_name=str(expected_path),
    )
    assert target_dir.exists()


def test_save_generated_images_skips_missing_file_ids(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project_client = MagicMock()
    project_client.agents.files.save = MagicMock()

    attachments = [
        SimpleNamespace(image_file=None),
        SimpleNamespace(image_file=SimpleNamespace(file_id=None)),
    ]
    message = SimpleNamespace(image_contents=attachments)
    result = SimpleNamespace(messages=[message])

    target_dir = tmp_path / "images"
    monkeypatch.setattr(math_agent, "_IMAGE_OUTPUT_DIR", target_dir)

    math_agent._save_generated_images(project_client, result)

    assert target_dir.exists()
    project_client.agents.files.save.assert_not_called()
