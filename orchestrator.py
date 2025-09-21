import itertools
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import tempfile
import traceback
import subprocess
import sys

from openai_utils import GENERATED_DIR, call_openai_api, run_codex_cli


class FlowCancelled(Exception):
    """Raised when a flow is cancelled due to exceeding failure limits."""


class MaxFlowFailuresExceeded(Exception):
    """Raised when the maximum number of flow failures has been reached."""


class FlowConfig(list):
    """Represents a generated flow along with metadata about its inputs."""

    def __init__(
        self,
        steps: Iterable[Dict[str, Any]],
        interpolated_paths: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(steps)
        self.interpolated_paths: Tuple[str, ...] = tuple(
            interpolated_paths or ()
        )


def _run_flow(
    config: List[Dict[str, Any]],
    step_counts: List[int],
    lock: threading.Lock,
    workdir: Path,
    flow_dir: Path,
    codex_timeout: Optional[int] = None,
    cancel_event: Optional[threading.Event] = None,
    openai_request_options: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple[List[Tuple[str, Optional[Path], Path]], bool]:
    """Execute a single flow defined in ``config``.

    When a step includes ``{"array": true}``, the step's output is parsed as a
    JSON array. The remaining steps are executed for each element in parallel,
    effectively branching the flow. The final result is a list of tuples
    containing each branch's output, the path to a final message file when the
    last step was a codex invocation, and the directory for that branch. A
    boolean flag is returned alongside the results indicating whether any part
    of the flow failed.
    """

    flow_failed = False

    def mark_failed() -> None:
        nonlocal flow_failed
        flow_failed = True

    def run_from(
        idx: int, prev_output: str, prev_path: Optional[Path], curr_dir: Path
    ) -> List[Tuple[str, Optional[Path], Path]]:
        if cancel_event and cancel_event.is_set():
            raise FlowCancelled()
        if idx >= len(config):
            return [(prev_output, prev_path, curr_dir)]

        step = config[idx]
        step_type = step.get("type")

        with lock:
            step_counts[idx] += 1

        try:
            prompt = step.get("prompt", "")
            prmpt_file = step.get("prmpt_file")
            if prmpt_file and not prompt:
                prompt = Path(prmpt_file).read_text(encoding="utf-8")
            if prev_output:
                prompt = f"{prompt}\n{prev_output}".strip()

            if step_type == "codex":
                output, path = run_codex_cli(
                    prompt, workdir, curr_dir, timeout=codex_timeout
                )
            elif step_type == "openai":
                web_search_enabled = step.get("web_search") is True
                openai_kwargs: Dict[str, str] = {}
                if openai_request_options:
                    for key in ("model", "service_tier", "reasoning_effort"):
                        value = openai_request_options.get(key)
                        if value:
                            openai_kwargs[key] = value
                response = call_openai_api(
                    prompt,
                    web_search=web_search_enabled,
                    **openai_kwargs,
                )
                output = (
                    response.get("output", [{}])[0]
                    .get("content", [{}])[0]
                    .get("text", "")
                )
                response_path = curr_dir / f"step_{idx}_openai_response.json"
                try:
                    response_path.write_text(
                        json.dumps(
                            response,
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        ),
                        encoding="utf-8",
                    )
                except TypeError:
                    # Fallback to storing a string representation if JSON encoding fails.
                    response_path.write_text(str(response), encoding="utf-8")
                path = curr_dir / f"step_{idx}_openai.txt"
                path.write_text(output, encoding="utf-8")
            elif "cmd" in step:
                completed = subprocess.run(
                    step["cmd"],
                    input=prev_output,
                    capture_output=True,
                    text=True,
                    shell=True,
                    check=True,
                )
                output = completed.stdout
                stdout_file = curr_dir / f"step_{idx}_cmd.txt"
                stdout_file.write_text(output, encoding="utf-8")
                path = None
            else:
                raise ValueError(f"Unknown step type: {step_type}")
        except Exception as e:
            called_process_error: Optional[subprocess.CalledProcessError] = None
            if isinstance(e, subprocess.CalledProcessError):
                called_process_error = e
            else:
                cause = getattr(e, "__cause__", None)
                if isinstance(cause, subprocess.CalledProcessError):
                    called_process_error = cause

            stderr_text: Optional[str] = None
            exit_code: Optional[int] = None

            if called_process_error is not None:
                exit_code = called_process_error.returncode
                if called_process_error.stderr:
                    stderr_text = str(called_process_error.stderr)
            else:
                exit_code = getattr(e, "returncode", None)
                stderr_attr = getattr(e, "stderr", None)
                if stderr_attr:
                    stderr_text = str(stderr_attr)

            stderr_to_log = stderr_text
            if not stderr_to_log and isinstance(e, subprocess.CalledProcessError):
                stderr_to_log = str(getattr(e, "stderr", "")) or None

            if stderr_to_log:
                try:
                    sys.stderr.write(stderr_to_log)
                    if not stderr_to_log.endswith("\n"):
                        sys.stderr.write("\n")
                    sys.stderr.flush()
                except Exception:
                    pass
            mark_failed()
            errors_base = curr_dir / "errors"
            errors_base.mkdir(parents=True, exist_ok=True)
            err_dir = Path(tempfile.mkdtemp(prefix="run_", dir=errors_base))
            error_file = err_dir / f"step_{idx}_{step_type}.txt"
            error_file.write_text(
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                encoding="utf-8",
            )
            if stderr_text is not None or exit_code is not None:
                stderr_file = err_dir / f"step_{idx}_{step_type}_stderr.txt"
                exit_code_str = "unknown" if exit_code is None else str(exit_code)
                stderr_content = stderr_text or ""
                if stderr_content and not stderr_content.endswith("\n"):
                    stderr_content += "\n"
                stderr_file.write_text(
                    f"exit_code: {exit_code_str}\n{stderr_content}",
                    encoding="utf-8",
                )
            output = ""
            path = error_file
            return [(output, path, curr_dir)]
        finally:
            with lock:
                step_counts[idx] -= 1

        if step.get("array"):
            try:
                items = json.loads(output)
                if not isinstance(items, list):
                    raise ValueError("Expected JSON array")
            except Exception as e:
                mark_failed()
                errors_base = curr_dir / "errors"
                errors_base.mkdir(parents=True, exist_ok=True)
                err_dir = Path(tempfile.mkdtemp(prefix="run_", dir=errors_base))
                error_file = err_dir / f"step_{idx}_array.txt"
                error_file.write_text(
                    f"JSON error: {e}\n{traceback.format_exc()}",
                    encoding="utf-8",
                )
                return [("", error_file, curr_dir)]

            results: List[Tuple[str, Optional[Path], Path]] = []
            threads: List[threading.Thread] = []
            res_lock = threading.Lock()

            cancelled = False

            for i, item in enumerate(items):
                branch_dir = curr_dir / f"branch_{i}"
                branch_dir.mkdir(parents=True, exist_ok=True)
                item_str = json.dumps(item) if not isinstance(item, str) else item

                def worker(s=item_str, bdir=branch_dir):
                    nonlocal cancelled
                    if cancel_event and cancel_event.is_set():
                        cancelled = True
                        mark_failed()
                        return
                    try:
                        branch_res = run_from(idx + 1, s, None, bdir)
                    except FlowCancelled:
                        cancelled = True
                        mark_failed()
                        return
                    with res_lock:
                        results.extend(branch_res)

                t = threading.Thread(target=worker)
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            if cancelled:
                raise FlowCancelled()

            return results

        return run_from(idx + 1, output, path, curr_dir)

    try:
        results = run_from(0, "", None, flow_dir)
    except FlowCancelled:
        mark_failed()
        failure_marker = flow_dir / "flow_failed.txt"
        failure_marker.write_text("Flow failed", encoding="utf-8")
        raise
    if flow_failed:
        failure_marker = flow_dir / "flow_failed.txt"
        failure_marker.write_text("Flow failed", encoding="utf-8")
    return results, flow_failed


def _generate_flow_configs(
    base_config: List[Dict[str, Any]],
    key_files: Dict[str, Path],
    append_filepath: bool = False,
) -> List[FlowConfig]:
    """Expand a base configuration into multiple flows via placeholder files.

    Placeholders in prompts must be wrapped with triple braces, e.g. ``{{{name}}}``.
    When ``append_filepath`` is ``True``, the path to each interpolated file is
    appended after its contents in the prompt.
    """

    if not key_files:
        return [FlowConfig((dict(step) for step in base_config))]

    loaded: Dict[str, List[Tuple[str, str]]] = {}

    for key, file_path in key_files.items():
        with file_path.open("r", encoding="utf-8") as f:
            paths = [line.strip() for line in f.readlines() if line.strip()]
        entries: List[Tuple[str, str]] = []
        for p in paths:
            text = Path(p).read_text(encoding="utf-8")
            if append_filepath:
                text = text.rstrip("\n") + f"\n{p}"
            entries.append((p, text))
        loaded[key] = entries

    keys = list(loaded.keys())
    values_product = itertools.product(*(loaded[k] for k in keys))
    flow_configs: List[FlowConfig] = []

    for combo in values_product:
        mapping: Dict[str, str] = {}
        combo_paths: List[str] = []
        for key, (path_str, value) in zip(keys, combo):
            mapping[key] = value
            combo_paths.append(path_str)

        flow: List[Dict[str, Any]] = []
        for step in base_config:
            new_step = dict(step)
            prompt = new_step.get("prompt", "")
            cmd_str = new_step.get("cmd")
            prmpt_file = new_step.get("prmpt_file")
            for key, value in mapping.items():
                placeholder = "{{{" + key + "}}}"
                prompt = prompt.replace(placeholder, value)
                if cmd_str is not None:
                    cmd_str = cmd_str.replace(placeholder, value)
                if prmpt_file is not None:
                    prmpt_file = prmpt_file.replace(placeholder, value)
            if prmpt_file is not None:
                prompt = Path(prmpt_file).read_text(encoding="utf-8")
                for key, value in mapping.items():
                    placeholder = "{{{" + key + "}}}"
                    prompt = prompt.replace(placeholder, value)
                new_step["prmpt_file"] = prmpt_file
            new_step["prompt"] = prompt
            if cmd_str is not None:
                new_step["cmd"] = cmd_str
            flow.append(new_step)
        flow_configs.append(FlowConfig(flow, combo_paths))

    return flow_configs


def orchestrate(
    base_config: List[Dict[str, Any]],
    flow_configs: List[List[Dict[str, Any]]],
    parallel: int = 1,
    workdir: Path = Path("."),
    codex_timeout: Optional[int] = None,
    max_flow_failures: int = 3,
    print_flow_paths: bool = True,
    list_codex_final_paths: bool = False,
    halt_on_max_failures: bool = True,
    openai_request_options: Optional[Dict[str, Optional[str]]] = None,
) -> List[Tuple[str, Optional[Path], Path]]:
    """Execute multiple flows with a concurrency cap while logging active counts.

    Returns a list of tuples containing each branch's final message, the path to
    the file holding that message when produced by a codex step, and the branch's
    output directory. Steps may define ``{"array": true}`` to branch a flow based
    on a JSON array output. Each step may optionally define a ``name`` field,
    which is used in the live progress output instead of the underlying step
    ``type``.

    Args:
        base_config: The original configuration defining step types and prompts.
        flow_configs: Expanded configurations for each flow.
        parallel: Maximum number of flows to run concurrently.
        workdir: Directory to run codex commands from.
        codex_timeout: Optional timeout in seconds for codex CLI invocations.
        max_flow_failures: Maximum number of flow-level failures allowed before
            cancelling remaining work.
        print_flow_paths: When ``True`` (default), emit the generated flow
            directory path for each flow to standard output.
        list_codex_final_paths: When ``True``, print the absolute path to each
            Codex final message file as flows finish, provided the final step is
            a Codex job and the flow completed successfully.
        halt_on_max_failures: When ``True`` (default), stop scheduling new
            flows and raise :class:`MaxFlowFailuresExceeded` once
            ``max_flow_failures`` is reached. When ``False``, the orchestrator
            continues running remaining flows and returns all results even if
            the failure threshold is exceeded.
        openai_request_options: Optional overrides for OpenAI steps, such as
            ``model``, ``service_tier``, and ``reasoning_effort``. When not
            provided, defaults built into :func:`call_openai_api` are used.

    Raises:
        MaxFlowFailuresExceeded: When the number of failed flows reaches the
            configured ``max_flow_failures`` threshold.
    """

    # Prefer a user-defined name for each step when displaying progress; fall back
    # to the step's type (e.g. "codex" or "openai") if no custom name is given.
    if max_flow_failures < 1:
        raise ValueError("max_flow_failures must be at least 1")

    step_names = [step.get("name") or step.get("type", "") for step in base_config]
    final_step_is_codex = bool(base_config) and base_config[-1].get("type") == "codex"
    step_counts = [0] * len(base_config)
    step_lock = threading.Lock()
    progress_lock = threading.Lock()
    results: List[Tuple[str, Optional[Path], Path]] = []
    finished = 0
    total = len(flow_configs)
    failed_flows = 0
    cancel_event = threading.Event()
    cancel_message_printed = False
    failed_flow_paths: List[Tuple[str, ...]] = []

    def worker(
        flow_conf: List[Dict[str, Any]],
        flow_dir: Path,
        interpolated_paths: Tuple[str, ...],
    ) -> None:
        nonlocal finished, failed_flows, cancel_message_printed
        try:
            branch_results, flow_failed = _run_flow(
                flow_conf,
                step_counts,
                step_lock,
                workdir,
                flow_dir,
                codex_timeout=codex_timeout,
                cancel_event=cancel_event,
                openai_request_options=openai_request_options,
            )
        except FlowCancelled:
            record_finished("failed", interpolated_paths)
            with progress_lock:
                finished += 1
            return

        status = "failed" if flow_failed else "done"
        success_paths: List[Path] = []
        if (
            list_codex_final_paths
            and final_step_is_codex
            and not flow_failed
        ):
            success_paths = [
                path
                for _, path, _ in branch_results
                if path is not None and path.name == "final_message.txt"
            ]

        trigger_message = False
        with progress_lock:
            results.extend(branch_results)
            finished += 1
            if flow_failed:
                failed_flows += 1
                if interpolated_paths:
                    failed_flow_paths.append(interpolated_paths)
                if halt_on_max_failures and failed_flows >= max_flow_failures:
                    cancel_event.set()
                    if not cancel_message_printed:
                        cancel_message_printed = True
                        trigger_message = True

        record_finished(status, interpolated_paths)
        for path in success_paths:
            try:
                print(path.resolve(), flush=True)
            except FileNotFoundError:
                # If the file was removed before printing, skip emitting it.
                continue

        if trigger_message:
            print("Maximum flow failures reached", flush=True)

    stop_event = threading.Event()

    def monitor():
        while not stop_event.is_set():
            with step_lock:
                parts = [f"{name}: {count}" for name, count in zip(step_names, step_counts)]
            with progress_lock:
                prog = f"{finished}/{total}"
            display = " -> ".join(parts)
            if display:
                display += f" | {prog}"
            else:
                display = prog
            print(display, end="\r", flush=True)
            time.sleep(0.5)
        with step_lock:
            parts = [f"{name}: {count}" for name, count in zip(step_names, step_counts)]
        with progress_lock:
            prog = f"{finished}/{total}"
        display = " -> ".join(parts)
        if display:
            display += f" | {prog}"
        else:
            display = prog
        print(display)

    run_dir = Path(tempfile.mkdtemp(prefix="run_", dir=GENERATED_DIR))

    finished_file = run_dir / "finished.txt"
    finished_file.touch()
    finished_lock = threading.Lock()

    def record_finished(status: str, interpolated_paths: Tuple[str, ...]) -> None:
        parts = ", ".join(interpolated_paths)
        line = status if not parts else f"{status} {parts}"
        with finished_lock:
            with finished_file.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    threads: List[threading.Thread] = []
    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()

    for flow_conf in flow_configs:
        if cancel_event.is_set():
            break
        flow_dir = Path(tempfile.mkdtemp(prefix="flow_", dir=run_dir))
        if print_flow_paths:
            print(flow_dir.resolve())
        interpolated_paths = getattr(flow_conf, "interpolated_paths", tuple())
        while True:
            if cancel_event.is_set():
                break
            with progress_lock:
                active = len([t for t in threads if t.is_alive()])
            if active < parallel:
                t = threading.Thread(
                    target=worker,
                    args=(flow_conf, flow_dir, tuple(interpolated_paths)),
                )
                threads.append(t)
                t.start()
                break
            time.sleep(0.1)

        if cancel_event.is_set():
            break

    for t in threads:
        t.join()

    stop_event.set()
    monitor_thread.join()

    if failed_flows:
        failed_file = run_dir / "failed_files"
        with failed_file.open("w", encoding="utf-8") as fh:
            for paths in failed_flow_paths:
                fh.write(",".join(paths))
                fh.write("\n")

    if (
        halt_on_max_failures
        and cancel_event.is_set()
        and failed_flows >= max_flow_failures
    ):
        if not cancel_message_printed:
            print("Maximum flow failures reached", flush=True)
        raise MaxFlowFailuresExceeded("Maximum flow failures reached")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run model orchestration based on JSON config"
    )
    parser.add_argument("config", help="Path to JSON configuration file")
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Maximum number of flows to run concurrently",
    )
    parser.add_argument(
        "--key",
        action="append",
        default=[],
        help="Placeholder interpolation in the form name:filelist.txt (use {{{name}}} in prompts)",
    )
    parser.add_argument(
        "--append-filepath",
        action="store_true",
        help="Append source file path after interpolated content",
    )
    parser.add_argument(
        "--max-flow-failures",
        type=int,
        default=3,
        help="Maximum number of flow failures allowed before cancelling execution",
    )
    parser.add_argument(
        "--ignore-max-failures",
        action="store_true",
        help=(
            "Continue running flows even after reaching the max failure "
            "threshold; disables the fail-fast behaviour"
        ),
    )
    parser.add_argument(
        "--workdir",
        required=True,
        help="Directory to run codex commands from",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for each codex CLI invocation",
    )
    parser.add_argument(
        "--openai-model",
        default=None,
        help=(
            "Model identifier to use for OpenAI steps. Defaults to the value "
            "configured in call_openai_api when not provided."
        ),
    )
    parser.add_argument(
        "--openai-service-tier",
        default=None,
        help="Service tier to request for OpenAI steps (for example, 'default' or 'scale')",
    )
    parser.add_argument(
        "--openai-reasoning-effort",
        default=None,
        help=(
            "Reasoning effort level for OpenAI steps (for example, 'medium' or 'high')"
        ),
    )
    parser.add_argument(
        "--hide-flow-paths",
        action="store_true",
        help="Suppress printing the generated flow directory paths",
    )
    parser.add_argument(
        "--list-final-message-paths",
        action="store_true",
        help=(
            "When the final step is a Codex job, print the absolute path to each "
            "final_message.txt as flows complete"
        ),
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    key_files = {}
    for item in args.key:
        if ":" not in item:
            raise ValueError("--key expects format name:filelist.txt")
        name, path = item.split(":", 1)
        key_files[name] = Path(path)

    flow_configs = _generate_flow_configs(
        config, key_files, append_filepath=args.append_filepath
    )

    openai_request_options = {
        key: value
        for key, value in {
            "model": args.openai_model,
            "service_tier": args.openai_service_tier,
            "reasoning_effort": args.openai_reasoning_effort,
        }.items()
        if value is not None
    }
    if not openai_request_options:
        openai_request_options = None

    try:
        results = orchestrate(
            config,
            flow_configs,
            parallel=args.parallel,
            workdir=Path(args.workdir),
            codex_timeout=args.timeout,
            max_flow_failures=args.max_flow_failures,
            print_flow_paths=not args.hide_flow_paths,
            list_codex_final_paths=args.list_final_message_paths,
            halt_on_max_failures=not args.ignore_max_failures,
            openai_request_options=openai_request_options,
        )
    except MaxFlowFailuresExceeded:
        sys.exit(1)
    for idx, (res, path, flow_dir) in enumerate(results):
        if path is None:
            filename = (
                "final_message.txt" if len(results) == 1 else f"final_message_{idx}.txt"
            )
            path = flow_dir / filename
            path.write_text(res, encoding="utf-8")
        print(path)
