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

## CLI entry point

Use `main.py` to choose which agent to run and provide ad-hoc prompts from the terminal:

```bash
python main.py --agent weather --prompt "What's the weather like in Seattle today?"
```

If you omit `--prompt`, the script will interactively ask for your message. Pass `--additional-instructions` to send run-scoped directives (for example, "respond in Spanish") and `--auto-delete-agent` to delete the temporary agent after the run completes.

### Available agents

| Agent     | Description                                                           |
| --------- | --------------------------------------------------------------------- |
| `weather` | Calls the Weatherstack-powered function tool for live conditions.     |
| `math`    | Uses the Code Interpreter tool to solve and visualize math questions. |

Run individual agent modules directly if you prefer:

```bash
python weather_agent.py
python math_agent.py
```

Each module exposes a `run(prompt, ...)` helper that can be imported into other scripts.
