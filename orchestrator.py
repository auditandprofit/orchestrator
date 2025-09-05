import json
from typing import List, Dict, Any

from openai_utils import run_codex_cli, call_openai_api


def orchestrate(config: List[Dict[str, Any]]) -> str:
    """Execute a sequence of model calls defined in config.

    Each config item must contain:
        type: "codex" or "openai"
        prompt: The prompt string to send to the model

    The output of each step is appended to the prompt of the next step
    allowing sequential reasoning across models.

    Args:
        config: List of configuration dictionaries.

    Returns:
        The final model output as a string.
    """
    prev_output = ""

    for step in config:
        step_type = step.get("type")
        prompt = step.get("prompt", "")
        if prev_output:
            prompt = f"{prompt}\n{prev_output}".strip()

        if step_type == "codex":
            prev_output = run_codex_cli(prompt)
        elif step_type == "openai":
            response = call_openai_api(prompt)
            # Responses API returns dict; extract content if possible
            output_text = response.get("output", [{}])[0].get("content", [{}])[0].get("text", "")
            prev_output = output_text
        else:
            raise ValueError(f"Unknown step type: {step_type}")

    return prev_output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run model orchestration based on JSON config")
    parser.add_argument("config", help="Path to JSON configuration file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    result = orchestrate(config)
    print(result)
