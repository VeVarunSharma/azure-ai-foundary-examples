"""Weather agent utilities."""

import json
from dataclasses import dataclass
from typing import Any, Optional

import requests
from requests import RequestException

from config.weatherstack_config import (
    WEATHERSTACK_API_URL,
    WEATHERSTACK_TIMEOUT_SECONDS,
    get_weatherstack_api_key,
)
from utils.agent_runtime import AgentConfig, AgentRunResult, run_agent_interaction


@dataclass(frozen=True)
class WeatherReport:
    location: str
    temperature_c: float
    condition: str
    humidity_pct: int

    def serialize(self, date: Optional[str] = None) -> str:
        """Return a human-readable summary used as tool output."""
        date_clause = f" on {date}" if date else ""
        return (
            f"Weather for {self.location}{date_clause}: {self.temperature_c:.1f}°C, "
            f"{self.condition}, humidity {self.humidity_pct}%"
        )


WEATHERSTACK_API_KEY = get_weatherstack_api_key()


def _extract_condition(descriptions: Optional[list[str]]) -> str:
    if not descriptions:
        return "Unknown conditions"
    return ", ".join(desc for desc in descriptions if desc) or "Unknown conditions"


def get_weatherstack_weather(location: str, date: Optional[str] = None) -> str:
    """Fetch live weather data via the Weatherstack current-conditions endpoint."""
    if not location:
        return "Missing location."

    params = {
        "access_key": WEATHERSTACK_API_KEY,
        "query": location,
        "units": "m",
    }

    fallback_note = ""
    if date:
        fallback_note = (
            "Historical dates are not supported on this plan; returning the latest conditions instead."
        )

    try:
        response = requests.get(
            WEATHERSTACK_API_URL,
            params=params,
            timeout=WEATHERSTACK_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except RequestException as exc:
        return f"Weather service request failed: {exc}"

    try:
        payload = response.json()
    except ValueError:
        return "Weather service returned an unreadable response."

    if error := payload.get("error"):
        code = error.get("code", "unknown")
        message = error.get("info", "Unknown error from Weatherstack.")
        return f"Weather service error ({code}): {message}"

    location_data = payload.get("location") or {}
    current = payload.get("current") or {}

    if not current:
        return "Weather service returned no current conditions for that query."

    descriptions = current.get("weather_descriptions")
    condition = _extract_condition(descriptions)
    temperature = current.get("temperature")
    humidity = current.get("humidity")

    if temperature is None or humidity is None:
        return "Weather service response was missing temperature or humidity data."

    report = WeatherReport(
        location=location_data.get("name") or location.title(),
        temperature_c=float(temperature),
        condition=condition,
        humidity_pct=int(humidity),
    )

    reported_date = date or location_data.get("localtime")
    summary = report.serialize(date=reported_date)
    extra_bits = ["Data source: Weatherstack live API."]
    if fallback_note:
        extra_bits.insert(0, fallback_note)
    if location_data.get("country"):
        extra_bits.append(f"Country: {location_data['country']}")
    return f"{summary}. {' '.join(extra_bits)}"


weather_tool_definition = {
    "type": "function",
    "function": {
        "name": "get_weatherstack_weather",
        "description": (
            "Return live weather information (temperature, conditions, humidity) for a city using the Weatherstack API."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Name of the city to look up weather for.",
                },
                "date": {
                    "type": "string",
                    "description": "Optional ISO date string for the requested forecast day.",
                },
            },
            "required": ["location"],
        },
    },
}


def run(
    user_input: str,
    *,
    additional_instructions: Optional[str] = None,
    auto_delete_agent: bool = False,
) -> AgentRunResult:
    config = AgentConfig(
        name="weather-assistant",
        instructions=(
            "You are a helpful weather assistant. Call the get_weatherstack_weather tool "
            "to provide real-time conditions from the Weatherstack API. Mention when "
            "historical dates are unavailable and clarify any assumptions you make."
        ),
        tools=[weather_tool_definition],
    )

    return run_agent_interaction(
        config=config,
        user_input=user_input,
        additional_instructions=(
            additional_instructions
            or "If no date is given, assume the request is for today and echo that assumption."
        ),
        handle_tool_calls=_handle_tool_calls,
        auto_delete_agent=auto_delete_agent,
    )


def _handle_tool_calls(project_client: Any, tool_calls: list[Any]) -> list[dict[str, str]]:
    outputs: list[dict[str, str]] = []
    for call in tool_calls:
        if getattr(call, "type", "") != "function":
            continue

        func = getattr(call, "function", None)
        if not func or getattr(func, "name", "") != "get_weatherstack_weather":
            continue

        try:
            arguments = json.loads(getattr(func, "arguments", "") or "{}")
        except json.JSONDecodeError:
            arguments = {}

        output = get_weatherstack_weather(
            location=arguments.get("location", ""),
            date=arguments.get("date"),
        )
        outputs.append(
            {
                "tool_call_id": getattr(call, "id", ""),
                "output": output,
            }
        )

    return outputs


if __name__ == "__main__":
    from utils.agent_runtime import print_thread_messages

    result = run("What's the weather like in Seattle today?")
    print_thread_messages(result)
