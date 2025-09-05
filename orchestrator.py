import itertools
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from openai_utils import call_openai_api, run_codex_cli


def _run_flow(
    config: List[Dict[str, Any]],
    step_counts: List[int],
    lock: threading.Lock,
) -> str:
    """Execute a single flow defined in config, updating step counts."""

    prev_output = ""

    for idx, step in enumerate(config):
        step_type = step.get("type")
        prompt = step.get("prompt", "")
        if prev_output:
            prompt = f"{prompt}\n{prev_output}".strip()

        with lock:
            step_counts[idx] += 1

        try:
            if step_type == "codex":
                prev_output = run_codex_cli(prompt)
            elif step_type == "openai":
                response = call_openai_api(prompt)
                # Responses API returns dict; extract content if possible
                output_text = (
                    response.get("output", [{}])[0]
                    .get("content", [{}])[0]
                    .get("text", "")
                )
                prev_output = output_text
            else:
                raise ValueError(f"Unknown step type: {step_type}")
        finally:
            with lock:
                step_counts[idx] -= 1

    return prev_output


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
) -> List[str]:
    """Execute multiple flows with a concurrency cap while logging active counts."""

    step_names = [step.get("type", "") for step in base_config]
    step_counts = [0] * len(base_config)
    step_lock = threading.Lock()
    progress_lock = threading.Lock()
    results: List[str] = [""] * len(flow_configs)
    finished = 0
    total = len(flow_configs)

    def worker(idx: int, flow_conf: List[Dict[str, Any]]):
        nonlocal finished
        result = _run_flow(flow_conf, step_counts, step_lock)
        results[idx] = result
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

    results = orchestrate(config, flow_configs, parallel=args.parallel)
    for res in results:
        print(res)
