import subprocess
import time
import warnings
import tempfile
import threading
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

try:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        APIStatusError,
        OpenAI,
    )
except ModuleNotFoundError:
    OpenAI = None  # type: ignore[assignment]

    class _MissingOpenAIError(Exception):
        """Fallback error type when the openai package is unavailable."""

    APIConnectionError = APITimeoutError = APIStatusError = _MissingOpenAIError  # type: ignore[assignment]
    _OPENAI_AVAILABLE = False
else:
    _OPENAI_AVAILABLE = True

try:
    import requests
except ModuleNotFoundError:
    class _RequestsFallback:
        class exceptions:  # type: ignore[no-redef]
            class RequestException(Exception):
                """Fallback RequestException when requests is unavailable."""

    requests = _RequestsFallback()  # type: ignore[assignment]


class CodexTimeoutError(Exception):
    """Custom exception for codex CLI timeouts."""


client = OpenAI() if _OPENAI_AVAILABLE else None

NETWORK_EXCEPTIONS = (
    requests.exceptions.RequestException,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
) if _OPENAI_AVAILABLE else (requests.exceptions.RequestException,)


def run_codex_cli(
    prompt: str,
    workdir: Path,
    output_dir: Path,
    max_retries: int = 3,
    timeout: Optional[int] = None,
) -> Tuple[str, Path]:
    """Run the Codex CLI and capture its final message via file output.

    The codex CLI supports writing its final message to a file. This helper
    invokes the CLI with that argument, streams the process's standard output to
    ``stdout.txt`` in the execution directory, then reads the final message file
    so it can be passed to subsequent steps.

    Args:
        prompt: The prompt to pass to the codex CLI.
        workdir: Directory to run the codex command from.
        output_dir: Base directory to store Codex output.
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
        tmpdir = Path(tempfile.mkdtemp(prefix="codex_exec_", dir=output_dir))
        output_path = tmpdir / "final_message.txt"
        stdout_path = tmpdir / "stdout.txt"
        time_path = tmpdir / "time.txt"
        try:
            start_time = time.time()
            with stdout_path.open("w", encoding="utf-8") as out_f:
                proc = subprocess.Popen(
                    [
                        "codex",
                        "exec",
                        "--skip-git-repo-check",
                        "-C",
                        str(workdir),
                        "--output-last-message",
                        str(output_path),
                        prompt,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                stderr_lines = []

                def stream_stdout():
                    for line in proc.stdout:  # type: ignore[arg-type]
                        out_f.write(line)
                        out_f.flush()

                def collect_stderr():
                    for line in proc.stderr:  # type: ignore[arg-type]
                        stderr_lines.append(line)

                t_out = threading.Thread(target=stream_stdout)
                t_err = threading.Thread(target=collect_stderr)
                t_out.start()
                t_err.start()

                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    t_out.join()
                    t_err.join()
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1)
                    continue

                t_out.join()
                t_err.join()

                duration = time.time() - start_time

                if proc.returncode != 0:
                    msg = "".join(stderr_lines) or str(proc.returncode)
                    raise subprocess.CalledProcessError(
                        proc.returncode, proc.args, stderr=msg
                    )

                if output_path.exists():
                    message = output_path.read_text(encoding="utf-8")
                elif stdout_path.exists():
                    message = stdout_path.read_text(encoding="utf-8")
                    output_path.write_text(message, encoding="utf-8")
                    time_path.write_text(
                        f"{proc.returncode}\n{duration}\n", encoding="utf-8"
                    )
                else:
                    raise FileNotFoundError(
                        "Codex CLI did not produce a final message file or stdout output"
                    )

                return message, output_path
        except subprocess.CalledProcessError as e:
            # Include stderr from the Codex CLI in the raised exception for logging.
            msg = e.stderr or str(e)
            raise Exception(msg) from e
        except Exception:
            # Any non-timeout exception should fail fast.
            raise


def call_openai_api(
    prompt: str,
    *,
    web_search: bool = False,
    max_retries: int = 3,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    service_tier: Optional[str] = None,
) -> dict:
    """Call the OpenAI Responses API with retry logic on network errors.

    Args:
        prompt: Prompt string for the response request.
        web_search: When ``True``, enable hosted web search for the response.
        max_retries: Maximum number of retries on network-related errors.

    Returns:
        The Responses API response as a dictionary.

    Raises:
        openai.OpenAIError: If network errors persist after retries.
        Exception: Any other exception is raised immediately.
    """
    if client is None:
        raise ModuleNotFoundError("openai package is required to call the OpenAI API")

    for attempt in range(max_retries):
        try:
            request_args = {"model": model or "gpt-4o-mini", "input": prompt}
            if reasoning_effort:
                request_args["reasoning"] = {"effort": reasoning_effort}
            if service_tier:
                request_args["service_tier"] = service_tier
            if web_search:
                request_args["tools"] = [{"type": "web_search"}]
            response = client.responses.create(**request_args)
            return response.model_dump()
        except NETWORK_EXCEPTIONS:
            if attempt == max_retries - 1:
                raise
            time.sleep(1)
        except Exception:
            # Fail fast for non-network errors.
            raise
