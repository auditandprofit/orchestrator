import json
from typing import List, Dict, Any

from openai_utils import run_codex_cli, call_openai_api
import threading
import time


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


def orchestrate(config: List[Dict[str, Any]], parallel: int = 1) -> List[str]:
    """Execute multiple flows in parallel while logging active step counts."""

    step_names = [step.get("type", "") for step in config]
    step_counts = [0] * len(config)
    lock = threading.Lock()
    results: List[str] = []

    def flow_wrapper():
        result = _run_flow(config, step_counts, lock)
        results.append(result)

    stop_event = threading.Event()

    def monitor():
        while not stop_event.is_set():
            with lock:
                parts = [f"{name}: {count}" for name, count in zip(step_names, step_counts)]
            print(" -> ".join(parts), end="\r", flush=True)
            time.sleep(0.5)
        with lock:
            parts = [f"{name}: {count}" for name, count in zip(step_names, step_counts)]
        print(" -> ".join(parts))

    threads = [threading.Thread(target=flow_wrapper) for _ in range(parallel)]
    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()

    for t in threads:
        t.start()
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
        help="Number of flows to run in parallel",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    results = orchestrate(config, parallel=args.parallel)
    for res in results:
        print(res)
