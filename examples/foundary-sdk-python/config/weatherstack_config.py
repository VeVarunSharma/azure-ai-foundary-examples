"""Configuration helpers for accessing the Weatherstack API."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from config.azure.ai_foundary_config import ConfigError

load_dotenv()


def _get_env(name: str) -> str:
	value = os.environ.get(name)
	if not value:
		raise ConfigError(f"Expected the {name} environment variable to be set.")
	return value


def _get_float_env(name: str, default: float) -> float:
	value = os.environ.get(name)
	if value is None:
		return default
	try:
		return float(value)
	except ValueError as exc:
		raise ConfigError(
			f"Expected the {name} environment variable to be numeric, got {value!r}."
		) from exc


WEATHERSTACK_API_URL = os.environ.get(
	"WEATHERSTACK_API_URL", "https://api.weatherstack.com/current"
)
WEATHERSTACK_TIMEOUT_SECONDS = _get_float_env("WEATHERSTACK_TIMEOUT_SECONDS", 10.0)


def get_weatherstack_api_key() -> str:
	"""Return the required Weatherstack API key from the environment."""

	return _get_env("WEATHERSTACK_API_KEY")


__all__ = [
	"WEATHERSTACK_API_URL",
	"WEATHERSTACK_TIMEOUT_SECONDS",
	"get_weatherstack_api_key",
]
