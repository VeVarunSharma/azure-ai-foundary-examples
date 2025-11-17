"""Command-line entry point to interact with available agents."""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Mapping, Optional

from utils.agent_runtime import AgentRunResult, print_thread_messages

import math_agent
import weather_agent

def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Interact with Azure AI Foundry agents defined in this repository. "
			"Choose an agent and provide the user input via --prompt or stdin."
		)
	)
	parser.add_argument(
		"--agent",
		choices=sorted(AGENT_REGISTRY.keys()),
		required=True,
		help="Which agent implementation to run.",
	)
	parser.add_argument(
		"--prompt",
		"-p",
		help="User message to send to the agent. If omitted, you'll be prompted interactively.",
	)
	parser.add_argument(
		"--additional-instructions",
		"-i",
		help="Optional run-scoped instructions to pass to the agent.",
	)
	parser.add_argument(
		"--auto-delete-agent",
		action="store_true",
		help="Delete the temporary agent after the run completes.",
	)
	return parser


def main(argv: Optional[list[str]] = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)

	runner = AGENT_REGISTRY[args.agent]

	prompt = (args.prompt or _prompt_for_input()).strip()
	if not prompt:
		print("No prompt provided; exiting.")
		return 1

	result = runner(
		prompt,
		additional_instructions=args.additional_instructions,
		auto_delete_agent=args.auto_delete_agent,
	)
	print_thread_messages(result)
	return 0


def _prompt_for_input() -> str:
	try:
		return input("Enter your message for the agent: ")
	except KeyboardInterrupt:
		print()
		return ""


AGENT_REGISTRY: Mapping[str, Callable[..., AgentRunResult]] = {
	"math": math_agent.run,
	"weather": weather_agent.run,
}


if __name__ == "__main__":
	raise SystemExit(main())
