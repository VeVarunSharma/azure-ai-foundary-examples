import json
import time
from dataclasses import dataclass
from typing import Dict, Optional

from config.azure.ai_foundary_config import (
    MODEL_DEPLOYMENT_NAME,
    project_client_context,
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


MOCK_WEATHER_DATA: Dict[str, WeatherReport] = {
    "seattle": WeatherReport("Seattle", 12.5, "Overcast with light rain", 87),
    "new york": WeatherReport("New York", 18.0, "Partly cloudy", 72),
    "london": WeatherReport("London", 15.3, "Foggy", 90),
}


def get_mock_weather(location: str, date: Optional[str] = None) -> str:
    """Resolve mock weather data for the agent's tool call."""
    if not location:
        return "Missing location."

    report = MOCK_WEATHER_DATA.get(location.lower())
    if not report:
        return (
            "Weather data unavailable for the requested location. "
            "Try Seattle, New York, or London while using mock data."
        )

    return report.serialize(date=date)


weather_tool_definition = {
    "type": "function",
    "function": {
        "name": "get_mock_weather",
        "description": (
            "Return mock weather information (temperature, conditions, humidity) for a city."
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
                "You are a helpful weather assistant. Call the get_mock_weather tool to reply "
                "with weather updates. Explain to the user that the data is mock when applicable."
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

                    if call.function.name != "get_mock_weather":
                        print(f"Unknown function requested: {call.function.name}")
                        continue

                    try:
                        arguments = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        print("Failed to parse tool arguments; returning error message.")
                        arguments = {}

                    output = get_mock_weather(
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
