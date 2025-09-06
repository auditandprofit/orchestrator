import subprocess
import time
import warnings
import tempfile
from pathlib import Path
from typing import Optional, Tuple

# Suppress urllib3 warning about unsupported SSL implementations
warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL",
)

# Directory for preserving Codex outputs
GENERATED_DIR = Path("generated")
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

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


def run_codex_cli(
    prompt: str,
    workdir: Path,
    max_retries: int = 3,
    timeout: Optional[int] = None,
) -> Tuple[str, Path]:
    """Run the Codex CLI and capture its final message via file output.

    The codex CLI supports writing its final message to a file. This helper
    invokes the CLI with that argument, then reads the file to obtain the
    message so it can be passed to subsequent steps.

    Args:
        prompt: The prompt to pass to the codex CLI.
        workdir: Directory to run the codex command from.
        max_retries: Maximum number of retries when a timeout occurs.
        timeout: Optional timeout for the subprocess call in seconds.

    Returns:
        A tuple of the final message string and the path to the file where it
        was stored.

    Raises:
        subprocess.TimeoutExpired: If the command times out after all retries.
        Exception: Any non-timeout exceptions from subprocess.run are raised
        immediately.
    """
    for attempt in range(max_retries):
        tmpdir = Path(tempfile.mkdtemp(prefix="codex_exec_", dir=GENERATED_DIR))
        output_path = tmpdir / "final_message.txt"
        try:
            subprocess.run(
                [
                    "codex",
                    "exec",
                    "--skip-git-repo-check",
                    "-C",
                    str(workdir),
                    "--output-file",
                    str(output_path),
                    prompt,
                ],
                capture_output=True,
                check=True,
                text=True,
                timeout=timeout,
            )
            return output_path.read_text(encoding="utf-8"), output_path
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
