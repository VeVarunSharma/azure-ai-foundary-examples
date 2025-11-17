import json
import time
from dataclasses import dataclass
from typing import Optional

import requests
from requests import RequestException

from config.azure.ai_foundary_config import (
    MODEL_DEPLOYMENT_NAME,
    project_client_context,
)
from config.weatherstack_config import (
    WEATHERSTACK_API_URL,
    WEATHERSTACK_TIMEOUT_SECONDS,
    get_weatherstack_api_key,
)


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


def main() -> None:
    with project_client_context() as project_client:
        agent = project_client.agents.create_agent(
            model=MODEL_DEPLOYMENT_NAME,
            name="weather-assistant",
            instructions=(
                "You are a helpful weather assistant. Call the get_weatherstack_weather tool "
                "to provide real-time conditions from the Weatherstack API. Mention when "
                "historical dates are unavailable and clarify any assumptions you make."
            ),
            tools=[weather_tool_definition],
        )
        print(f"Created agent, ID: {agent.id}")

        thread = project_client.agents.threads.create()
        print(f"Created thread, ID: {thread.id}")

        message = project_client.agents.messages.create(
            thread_id=thread.id,
            role="user",
            content="What's the weather like in Seattle today?",
        )
        print(f"Created message, ID: {message['id']}")

        run = project_client.agents.runs.create(
            thread_id=thread.id,
            agent_id=agent.id,
            additional_instructions=(
                "If no date is given, assume the request is for today and echo that assumption."
            ),
        )
        print(f"Run created with status: {run.status}")

        while run.status not in {"completed", "failed", "cancelled"}:
            if run.status == "requires_action":
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []

                for call in tool_calls:
                    if call.type != "function":
                        print(f"Skipping unsupported tool type: {call.type}")
                        continue

                    if call.function.name != "get_weatherstack_weather":
                        print(f"Unknown function requested: {call.function.name}")
                        continue

                    try:
                        arguments = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        print("Failed to parse tool arguments; returning error message.")
                        arguments = {}

                    output = get_weatherstack_weather(
                        location=arguments.get("location", ""),
                        date=arguments.get("date"),
                    )
                    tool_outputs.append(
                        {
                            "tool_call_id": call.id,
                            "output": output,
                        }
                    )

                if tool_outputs:
                    run = project_client.agents.runs.submit_tool_outputs(
                        thread_id=thread.id,
                        run_id=run.id,
                        tool_outputs=tool_outputs,
                    )
                    print("Submitted mock weather tool outputs.")
                else:
                    raise RuntimeError("Run requires tool output but no outputs were generated.")
            else:
                time.sleep(1)
                run = project_client.agents.runs.get(
                    thread_id=thread.id,
                    run_id=run.id,
                )
                print(f"Polled run status: {run.status}")

        print(f"Run finished with status: {run.status}")
        if run.status == "failed":
            print(f"Run failed: {run.last_error}")

        messages = project_client.agents.messages.list(thread_id=thread.id)
        for msg in messages:
            print(f"Role: {msg.role}")
            print(f"Content: {msg.content}")
            print("---")

        # project_client.agents.delete_agent(agent.id)
        # print("Deleted agent")


if __name__ == "__main__":
    main()
