import subprocess
import time
import warnings
from typing import Optional

# Suppress urllib3 warning about unsupported SSL implementations
warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL",
)

from openai import (
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    OpenAI,
)
import requests


class CodexTimeoutError(Exception):
    """Custom exception for codex CLI timeouts."""


client = OpenAI()

NETWORK_EXCEPTIONS = (
    requests.exceptions.RequestException,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
)


def run_codex_cli(prompt: str, max_retries: int = 3, timeout: Optional[int] = None) -> str:
    """Run the Codex CLI non-interactively with retry logic on timeouts.

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
                ["codex", "exec", prompt],
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
    """Call the OpenAI Responses API with retry logic on network errors.

    Args:
        prompt: Prompt string for the response request.
        max_retries: Maximum number of retries on network-related errors.

    Returns:
        The Responses API response as a dictionary.

    Raises:
        openai.OpenAIError: If network errors persist after retries.
        Exception: Any other exception is raised immediately.
    """
    for attempt in range(max_retries):
        try:
            response = client.responses.create(model="gpt-4o-mini", input=prompt)
            return response.model_dump()
        except NETWORK_EXCEPTIONS:
            if attempt == max_retries - 1:
                raise
            time.sleep(1)
        except Exception:
            # Fail fast for non-network errors.
            raise
