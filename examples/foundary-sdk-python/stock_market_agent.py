import json
import os
import time
from typing import Dict, Optional
import requests

from config.azure.ai_foundary_config import (
    MODEL_DEPLOYMENT_NAME,
    project_client_context,
)
ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY")
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"
DEFAULT_INTERVAL = "5min"
VALID_INTERVALS = {"1min", "5min", "15min", "30min", "60min"}
# Minimal mapping so the agent can resolve common company names to tickers.
KNOWN_COMPANIES: Dict[str, str] = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "meta": "META",
    "nvidia": "NVDA",
    "ibm": "IBM",
}
session = requests.Session()
def _resolve_symbol(company: Optional[str], symbol: Optional[str]) -> Optional[str]:
    if symbol:
        return symbol.strip().upper()
    if not company:
        return None
    normalized = company.strip().lower()
    if not normalized:
        return None
    mapped = KNOWN_COMPANIES.get(normalized)
    if mapped:
        return mapped
    fallback = company.strip().upper()
    if 1 <= len(fallback) <= 6 and fallback.replace(".", "").isalnum():
        return fallback
    return None

def fetch_intraday_stock_price(
    company: Optional[str] = None, symbol: Optional[str] = None, interval: str = DEFAULT_INTERVAL
) -> str:
    if not ALPHAVANTAGE_API_KEY:
        return "Missing ALPHAVANTAGE_API_KEY environment variable."
    chosen_interval = interval if interval in VALID_INTERVALS else DEFAULT_INTERVAL
    resolved_symbol = _resolve_symbol(company, symbol)
    if not resolved_symbol:
        return (
            "Unable to determine a ticker symbol from the provided company or symbol. "
            "Please specify the exchange ticker (e.g., AAPL)."
        )
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": resolved_symbol,
        "interval": chosen_interval,
        "apikey": ALPHAVANTAGE_API_KEY,
        "datatype": "json",
        "outputsize": "compact",
    }
    try:
        response = session.get(ALPHAVANTAGE_BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return f"Failed to reach Alpha Vantage: {exc}"
    except json.JSONDecodeError:
        return "Alpha Vantage returned a non-JSON response."
    series_key = next((key for key in data if key.startswith("Time Series")), None)
    if not series_key:
        diagnostic = data.get("Note") or data.get("Information") or data.get("Error Message")
        if diagnostic:
            return f"Alpha Vantage did not return intraday data: {diagnostic}"
        return "Alpha Vantage response did not include intraday time series data."
    series = data.get(series_key, {})
    if not series:
        return "Alpha Vantage returned an empty time series for that ticker."
    latest_timestamp = max(series.keys())
    latest_record = series[latest_timestamp]
    try:
        open_price = float(latest_record["1. open"])
        high_price = float(latest_record["2. high"])
        low_price = float(latest_record["3. low"])
        close_price = float(latest_record["4. close"])
        volume = int(float(latest_record["5. volume"]))
    except (KeyError, ValueError) as exc:
        return f"Failed to parse intraday record: {exc}"
    company_display = company or resolved_symbol
    return (
        f"Latest {chosen_interval} price for {company_display} ({resolved_symbol}) at {latest_timestamp} "
        f"is ${close_price:.2f} USD (open {open_price:.2f}, high {high_price:.2f}, "
        f"low {low_price:.2f}, volume {volume:,}). Data via Alpha Vantage."
    )

stock_price_tool_definition = {
    "type": "function",
    "function": {
        "name": "fetch_intraday_stock_price",
        "description": "Fetch the latest intraday stock price for a company using the Alpha Vantage API.",
        "parameters": {
            "type": "object",
            "properties": {
                "company": {
                    "type": "string",
                    "description": "Company name the user mentioned (e.g., Apple, Microsoft).",
                },
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol for the stock (e.g., AAPL, MSFT).",
                },
                "interval": {
                    "type": "string",
                    "description": "Intraday interval supported by Alpha Vantage.",
                    "enum": ["1min", "5min", "15min", "30min", "60min"],
                    "default": DEFAULT_INTERVAL,
                },
            },
        },
    },
}
def main() -> None:
    with project_client_context() as project_client:
        agent = project_client.agents.create_agent(
            model=MODEL_DEPLOYMENT_NAME,
            name="stock-market-assistant",
            instructions=(
                "You are a helpful stock market assistant. Call fetch_intraday_stock_price whenever the user "
                "asks for a current or very recent stock quote. Mention that prices come from Alpha Vantage and "
                "may be delayed. If the user supplies only a company name, pass it with the company parameter."
            ),
            tools=[stock_price_tool_definition],
        )
        print(f"Created agent, ID: {agent.id}")
        thread = project_client.agents.threads.create()
        print(f"Created thread, ID: {thread.id}")
        message = project_client.agents.messages.create(
            thread_id=thread.id,
            role="user",
            content="What is Apple's stock price right now?",
        )
        print(f"Created message, ID: {message['id']}")
        run = project_client.agents.runs.create(
            thread_id=thread.id,
            agent_id=agent.id,
            additional_instructions=(
                "Ask the user for the ticker if you cannot infer it. Use 5min interval when unsure."
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
                    if call.function.name != "fetch_intraday_stock_price":
                        print(f"Unknown function requested: {call.function.name}")
                        continue
                    try:
                        arguments = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        print("Failed to parse tool arguments; returning error message.")
                        arguments = {}
                    output = fetch_intraday_stock_price(
                        company=arguments.get("company"),
                        symbol=arguments.get("symbol"),
                        interval=arguments.get("interval", DEFAULT_INTERVAL),
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
                    print("Submitted stock price tool outputs.")
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