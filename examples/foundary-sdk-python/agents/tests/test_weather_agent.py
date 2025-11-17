from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from requests import RequestException

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents import weather_agent
from utils.agent_runtime import AgentConfig


def test_run_uses_expected_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_result = object()
    run_spy = MagicMock(return_value=fake_result)
    monkeypatch.setattr(weather_agent, "run_agent_interaction", run_spy)

    result = weather_agent.run("Weather in Seattle?")

    assert result is fake_result

    run_spy.assert_called_once()
    kwargs = run_spy.call_args.kwargs

    assert kwargs["user_input"] == "Weather in Seattle?"
    assert (
        kwargs["additional_instructions"]
        == "If no date is given, assume the request is for today and echo that assumption."
    )
    assert kwargs["handle_tool_calls"] is weather_agent._handle_tool_calls

    config = kwargs["config"]
    assert isinstance(config, AgentConfig)
    assert config.name == "weather-assistant"
    assert weather_agent.weather_tool_definition in config.tools


def test_run_allows_custom_additional_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    run_spy = MagicMock()
    monkeypatch.setattr(weather_agent, "run_agent_interaction", run_spy)

    custom = "Always mention UV index."
    weather_agent.run("Any storms tomorrow?", additional_instructions=custom)

    kwargs = run_spy.call_args.kwargs
    assert kwargs["additional_instructions"] == custom


def test_handle_tool_calls_invokes_weather_function(monkeypatch: pytest.MonkeyPatch) -> None:
    weather_fn = MagicMock(return_value="Sunny")
    monkeypatch.setattr(weather_agent, "get_weatherstack_weather", weather_fn)

    tool_call = SimpleNamespace(
        id="tool-123",
        type="function",
        function=SimpleNamespace(
            name="get_weatherstack_weather",
            arguments=json.dumps({"location": "Seattle", "date": "2025-05-01"}),
        ),
    )
    ignored_call = SimpleNamespace(type="function", function=SimpleNamespace(name="other"))

    outputs = weather_agent._handle_tool_calls(project_client=MagicMock(), tool_calls=[tool_call, ignored_call])

    assert outputs == [{"tool_call_id": "tool-123", "output": "Sunny"}]
    weather_fn.assert_called_once_with(location="Seattle", date="2025-05-01")


def test_get_weatherstack_weather_handles_request_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_, **__):
        raise RequestException("boom")

    monkeypatch.setattr(weather_agent.requests, "get", _raise)

    result = weather_agent.get_weatherstack_weather("Paris")

    assert "request failed" in result.lower()


def test_get_weatherstack_weather_success(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "location": {"name": "Seattle", "country": "USA", "localtime": "2025-05-01"},
        "current": {
            "temperature": 12,
            "humidity": 75,
            "weather_descriptions": ["Light rain", "Windy"],
        },
    }

    class DummyResponse:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._data

    request_spy = MagicMock(return_value=DummyResponse(payload))
    monkeypatch.setattr(weather_agent.requests, "get", request_spy)
    monkeypatch.setattr(weather_agent, "WEATHERSTACK_API_KEY", "test-key")

    summary = weather_agent.get_weatherstack_weather("seattle", date="2025-04-30")

    assert "Seattle" in summary
    assert "Light rain" in summary
    assert "Country: USA" in summary
    assert "Historical dates are not supported" in summary

    request_spy.assert_called_once()
    _, kwargs = request_spy.call_args
    assert kwargs["params"]["access_key"] == "test-key"
    assert kwargs["params"]["query"] == "seattle"


def test_extract_condition_handles_empty_values() -> None:
    assert weather_agent._extract_condition([]) == "Unknown conditions"
    assert weather_agent._extract_condition(["", None]) == "Unknown conditions"
    assert weather_agent._extract_condition(["Sunny", "Warm"]) == "Sunny, Warm"
