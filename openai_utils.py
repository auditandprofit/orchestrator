import subprocess
import time
from typing import Optional

import openai
import requests


class CodexTimeoutError(Exception):
    """Custom exception for codex CLI timeouts."""


NETWORK_EXCEPTIONS = (
    requests.exceptions.RequestException,
    openai.error.APIConnectionError,
    openai.error.Timeout,
    openai.error.ServiceUnavailableError,
)


def run_codex_cli(prompt: str, max_retries: int = 3, timeout: Optional[int] = None) -> str:
    """Run the OpenAI codex CLI with retry logic on timeouts.

    Args:
        prompt: The prompt to pass to the codex CLI.
        max_retries: Maximum number of retries when a timeout occurs.
        timeout: Optional timeout for the subprocess call in seconds.

    Returns:
        The stdout from the codex CLI.

    Raises:
        subprocess.TimeoutExpired: If the command times out after all retries.
        Exception: Any non-timeout exceptions from subprocess.run are raised immediately.
    """
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ["openai", "codex", "--prompt", prompt],
                capture_output=True,
                check=True,
                text=True,
                timeout=timeout,
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            if attempt == max_retries - 1:
                raise
            time.sleep(1)
        except Exception:
            # Any non-timeout exception should fail fast.
            raise


def call_openai_api(prompt: str, max_retries: int = 3) -> dict:
    """Call the OpenAI API with retry logic on network errors.

    Args:
        prompt: Prompt string for the completion request.
        max_retries: Maximum number of retries on network-related errors.

    Returns:
        The OpenAI API response as a dictionary.

    Raises:
        openai.error.OpenAIError: If network errors persist after retries.
        Exception: Any other exception is raised immediately.
    """
    for attempt in range(max_retries):
        try:
            response = openai.Completion.create(engine="text-davinci-003", prompt=prompt)
            return response
        except NETWORK_EXCEPTIONS:
            if attempt == max_retries - 1:
                raise
            time.sleep(1)
        except Exception:
            # Fail fast for non-network errors.
            raise
