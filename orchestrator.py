import itertools
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import tempfile

from openai_utils import GENERATED_DIR, call_openai_api, run_codex_cli


def _run_flow(
    config: List[Dict[str, Any]],
    step_counts: List[int],
    lock: threading.Lock,
    workdir: Path,
) -> Tuple[str, Optional[Path]]:
    """Execute a single flow defined in config, updating step counts.

    Returns a tuple of the final output and the path to the final message file
    when the last step was a codex invocation. For non-codex final steps, the
    path is ``None``.
    """

    prev_output = ""
    prev_path: Optional[Path] = None

    for idx, step in enumerate(config):
        step_type = step.get("type")
        prompt = step.get("prompt", "")
        if prev_output:
            prompt = f"{prompt}\n{prev_output}".strip()

        with lock:
            step_counts[idx] += 1

        try:
            if step_type == "codex":
                prev_output, prev_path = run_codex_cli(prompt, workdir)
            elif step_type == "openai":
                response = call_openai_api(prompt)
                # Responses API returns dict; extract content if possible
                output_text = (
                    response.get("output", [{}])[0]
                    .get("content", [{}])[0]
                    .get("text", "")
                )
                prev_output = output_text
                prev_path = None
            else:
                raise ValueError(f"Unknown step type: {step_type}")
        finally:
            with lock:
                step_counts[idx] -= 1

    return prev_output, prev_path


def _generate_flow_configs(
    base_config: List[Dict[str, Any]],
    key_files: Dict[str, Path],
) -> List[List[Dict[str, Any]]]:
    """Expand a base configuration into multiple flows via placeholder files.

    Placeholders in prompts must be wrapped with triple braces, e.g. ``{{{name}}}``.
    """

    if not key_files:
        return [base_config]

    loaded: Dict[str, List[str]] = {}

    for key, file_path in key_files.items():
        with file_path.open("r", encoding="utf-8") as f:
            paths = [line.strip() for line in f.readlines() if line.strip()]
        contents = [Path(p).read_text(encoding="utf-8") for p in paths]
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
            for key, value in mapping.items():
                placeholder = "{{{" + key + "}}}"
                prompt = prompt.replace(placeholder, value)
            new_step["prompt"] = prompt
            flow.append(new_step)
        flow_configs.append(flow)

    return flow_configs


def orchestrate(
    base_config: List[Dict[str, Any]],
    flow_configs: List[List[Dict[str, Any]]],
    parallel: int = 1,
    workdir: Path = Path("."),
) -> List[Tuple[str, Optional[Path]]]:
    """Execute multiple flows with a concurrency cap while logging active counts.

    Returns a list of tuples containing each flow's final message and the path to
    the file holding that message when produced by a codex step. Each step may
    optionally define a ``name`` field, which is used in the live progress output
    instead of the underlying step ``type``.
    """

    # Prefer a user-defined name for each step when displaying progress; fall back
    # to the step's type (e.g. "codex" or "openai") if no custom name is given.
    step_names = [step.get("name") or step.get("type", "") for step in base_config]
    step_counts = [0] * len(base_config)
    step_lock = threading.Lock()
    progress_lock = threading.Lock()
    results: List[Tuple[str, Optional[Path]]] = [("", None)] * len(flow_configs)
    finished = 0
    total = len(flow_configs)

    def worker(idx: int, flow_conf: List[Dict[str, Any]]):
        nonlocal finished
        result, path = _run_flow(flow_conf, step_counts, step_lock, workdir)
        results[idx] = (result, path)
        with progress_lock:
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

    for idx, flow_conf in enumerate(flow_configs):
        while True:
            with progress_lock:
                active = len([t for t in threads if t.is_alive()])
            if active < parallel:
                t = threading.Thread(target=worker, args=(idx, flow_conf))
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
        "--workdir",
        required=True,
        help="Directory to run codex commands from",
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

    flow_configs = _generate_flow_configs(config, key_files)

    results = orchestrate(
        config,
        flow_configs,
        parallel=args.parallel,
        workdir=Path(args.workdir),
    )
    for idx, (res, path) in enumerate(results):
        if path is None:
            tmpdir = Path(tempfile.mkdtemp(prefix="codex_run_", dir=GENERATED_DIR))
            filename = (
                "final_message.txt" if len(results) == 1 else f"final_message_{idx}.txt"
            )
            path = tmpdir / filename
            path.write_text(res, encoding="utf-8")
        print(path)
