# Foundary SDK Python Examples

These scripts demonstrate how to work with Azure AI Foundry agents, including custom tools such as the Weatherstack-powered weather assistant.

## Prerequisites

1. **Python environment** – Python 3.10+ is recommended. Create and activate a virtual environment.
2. **Dependencies** – Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. **Azure configuration** – Set the environment variables that `config/azure/ai_foundary_config.py` expects:
   - `PROJECT_ENDPOINT`
   - `MODEL_DEPLOYMENT_NAME`
   - (any authentication variables required by `DefaultAzureCredential`, if applicable)
4. **Weatherstack configuration** – `config/weatherstack_config.py` loads these variables:
   - `WEATHERSTACK_API_KEY` (required)
   - `WEATHERSTACK_API_URL` (optional, defaults to `https://api.weatherstack.com/current`)
   - `WEATHERSTACK_TIMEOUT_SECONDS` (optional, defaults to `10`)

You can place these values in a local `.env` file; they are automatically loaded via `python-dotenv`.

## Weather Agent

Run the weather agent to chat with the Azure AI Foundry agent that calls Weatherstack for live data:

```bash
python weather_agent.py
```

The agent will:

- Create a temporary agent using your configured model deployment.
- Call the `get_weatherstack_weather` tool whenever it needs real-time conditions.
- Respond with the latest temperature, humidity, and descriptive conditions sourced from Weatherstack.

### Notes

- Historical dates are not supported on the free Weatherstack plan. When a user asks for a date, the tool clarifies that it returns current conditions instead.
- Consider deleting temporary agents once you're finished experimenting to avoid clutter within your Azure AI Foundry project.
- The script prints all agent/assistant messages for easy debugging.
