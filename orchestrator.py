import itertools
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import tempfile
import traceback
import subprocess

from openai_utils import GENERATED_DIR, call_openai_api, run_codex_cli


def _run_flow(
    config: List[Dict[str, Any]],
    step_counts: List[int],
    lock: threading.Lock,
    workdir: Path,
    flow_dir: Path,
    codex_timeout: Optional[int] = None,
) -> List[Tuple[str, Optional[Path], Path]]:
    """Execute a single flow defined in ``config``.

    When a step includes ``{"array": true}``, the step's output is parsed as a
    JSON array. The remaining steps are executed for each element in parallel,
    effectively branching the flow. The final result is a list of tuples
    containing each branch's output, the path to a final message file when the
    last step was a codex invocation, and the directory for that branch.
    """

    def run_from(
        idx: int, prev_output: str, prev_path: Optional[Path], curr_dir: Path
    ) -> List[Tuple[str, Optional[Path], Path]]:
        if idx >= len(config):
            return [(prev_output, prev_path, curr_dir)]

        step = config[idx]
        step_type = step.get("type")

        with lock:
            step_counts[idx] += 1

        try:
            prompt = step.get("prompt", "")
            prmpt_file = step.get("prmpt_file")
            if prmpt_file:
                prompt = Path(prmpt_file).read_text(encoding="utf-8")
            if prev_output:
                prompt = f"{prompt}\n{prev_output}".strip()

            if step_type == "codex":
                output, path = run_codex_cli(
                    prompt, workdir, curr_dir, timeout=codex_timeout
                )
            elif step_type == "openai":
                response = call_openai_api(prompt)
                output = (
                    response.get("output", [{}])[0]
                    .get("content", [{}])[0]
                    .get("text", "")
                )
                path = None
            elif "cmd" in step:
                completed = subprocess.run(
                    step["cmd"],
                    input=prev_output,
                    capture_output=True,
                    text=True,
                    shell=True,
                    cwd=workdir,
                    check=True,
                )
                output = completed.stdout
                path = None
            else:
                raise ValueError(f"Unknown step type: {step_type}")
        except Exception as e:
            errors_base = curr_dir / "errors"
            errors_base.mkdir(parents=True, exist_ok=True)
            err_dir = Path(tempfile.mkdtemp(prefix="run_", dir=errors_base))
            error_file = err_dir / f"step_{idx}_{step_type}.txt"
            error_file.write_text(
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
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

            for i, item in enumerate(items):
                branch_dir = curr_dir / f"branch_{i}"
                branch_dir.mkdir(parents=True, exist_ok=True)
                item_str = json.dumps(item) if not isinstance(item, str) else item

                def worker(s=item_str, bdir=branch_dir):
                    branch_res = run_from(idx + 1, s, None, bdir)
                    with res_lock:
                        results.extend(branch_res)

                t = threading.Thread(target=worker)
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            return results

        return run_from(idx + 1, output, path, curr_dir)

    return run_from(0, "", None, flow_dir)


def _generate_flow_configs(
    base_config: List[Dict[str, Any]],
    key_files: Dict[str, Path],
    append_filepath: bool = False,
) -> List[List[Dict[str, Any]]]:
    """Expand a base configuration into multiple flows via placeholder files.

    Placeholders in prompts must be wrapped with triple braces, e.g. ``{{{name}}}``.
    When ``append_filepath`` is ``True``, the path to each interpolated file is
    appended after its contents in the prompt.
    """

    if not key_files:
        return [base_config]

    loaded: Dict[str, List[str]] = {}

    for key, file_path in key_files.items():
        with file_path.open("r", encoding="utf-8") as f:
            paths = [line.strip() for line in f.readlines() if line.strip()]
        contents: List[str] = []
        for p in paths:
            text = Path(p).read_text(encoding="utf-8")
            if append_filepath:
                text = text.rstrip("\n") + f"\n{p}"
            contents.append(text)
        loaded[key] = contents

    keys = list(loaded.keys())
    values_product = itertools.product(*(loaded[k] for k in keys))
    flow_configs: List[List[Dict[str, Any]]] = []

    for combo in values_product:
        mapping = dict(zip(keys, combo))
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
        flow_configs.append(flow)

    return flow_configs


def orchestrate(
    base_config: List[Dict[str, Any]],
    flow_configs: List[List[Dict[str, Any]]],
    parallel: int = 1,
    workdir: Path = Path("."),
    codex_timeout: Optional[int] = None,
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
    """

    # Prefer a user-defined name for each step when displaying progress; fall back
    # to the step's type (e.g. "codex" or "openai") if no custom name is given.
    step_names = [step.get("name") or step.get("type", "") for step in base_config]
    step_counts = [0] * len(base_config)
    step_lock = threading.Lock()
    progress_lock = threading.Lock()
    results: List[Tuple[str, Optional[Path], Path]] = []
    finished = 0
    total = len(flow_configs)

    def worker(flow_conf: List[Dict[str, Any]], flow_dir: Path):
        nonlocal finished
        branch_results = _run_flow(
            flow_conf, step_counts, step_lock, workdir, flow_dir, codex_timeout
        )
        with progress_lock:
            results.extend(branch_results)
            finished += 1

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

    threads: List[threading.Thread] = []
    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()

    for flow_conf in flow_configs:
        flow_dir = Path(tempfile.mkdtemp(prefix="flow_", dir=GENERATED_DIR))
        while True:
            with progress_lock:
                active = len([t for t in threads if t.is_alive()])
            if active < parallel:
                t = threading.Thread(target=worker, args=(flow_conf, flow_dir))
                threads.append(t)
                t.start()
                break
            time.sleep(0.1)

    for t in threads:
        t.join()

    stop_event.set()
    monitor_thread.join()

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

    results = orchestrate(
        config,
        flow_configs,
        parallel=args.parallel,
        workdir=Path(args.workdir),
        codex_timeout=args.timeout,
    )
    for idx, (res, path, flow_dir) in enumerate(results):
        if path is None:
            filename = (
                "final_message.txt" if len(results) == 1 else f"final_message_{idx}.txt"
            )
            path = flow_dir / filename
            path.write_text(res, encoding="utf-8")
        print(path)
