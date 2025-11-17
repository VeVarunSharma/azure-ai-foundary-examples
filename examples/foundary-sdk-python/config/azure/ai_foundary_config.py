"""Shared Azure AI Foundry configuration helpers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from azure.ai.projects import AIProjectClient
# from azure.identity import EnvironmentCredential
from azure.identity import DefaultAzureCredential

from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
	"""Raised when required configuration is missing."""


def _get_env(name: str) -> str:
	value = os.environ.get(name)
	if not value:
		raise ConfigError(f"Expected the {name} environment variable to be set.")
	return value


# AZURE_CLIENT_ID = _get_env("AZURE_CLIENT_ID")
# AZURE_TENANT_ID = _get_env("AZURE_TENANT_ID")
# AZURE_CLIENT_SECRET = _get_env("AZURE_CLIENT_SECRET")

# _AZURE_IDENTITY_VARS = (
# 	AZURE_CLIENT_ID,
# 	AZURE_TENANT_ID,
# 	AZURE_CLIENT_SECRET,
# )


def _ensure_env(names: tuple[str, ...]) -> None:
	for name in names:
		if not os.environ.get(name):
			raise ConfigError(f"Expected the {name} environment variable to be set.")

PROJECT_ENDPOINT = _get_env("PROJECT_ENDPOINT")
MODEL_DEPLOYMENT_NAME = _get_env("MODEL_DEPLOYMENT_NAME")

AZURE_CREDENTIAL = DefaultAzureCredential()

def create_project_client() -> AIProjectClient:
	"""Instantiate a new project client using shared configuration."""

	return AIProjectClient(
		endpoint=PROJECT_ENDPOINT,
		credential=AZURE_CREDENTIAL,
	)


@contextmanager
def project_client_context() -> Iterator[AIProjectClient]:
	"""Yield an opened project client that cleans up automatically."""

	client = create_project_client()
	with client:
		yield client


__all__ = [
	"ConfigError",
	"create_project_client",
	"AZURE_CREDENTIAL",
	"MODEL_DEPLOYMENT_NAME",
	"project_client_context",
	"PROJECT_ENDPOINT",
]
