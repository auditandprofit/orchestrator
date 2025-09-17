import shlex
import sys
import threading

import pytest

import orchestrator


def test_flow_failure_marks_directory(tmp_path):
    config = [{"type": "cmd", "cmd": "false"}]

    results, failed = orchestrator._run_flow(
        config, [0], threading.Lock(), tmp_path, tmp_path
    )

    assert failed
    assert results[0][1] is not None
    assert (tmp_path / "flow_failed.txt").exists()
    assert "errors" in str(results[0][1])


def _copy_flow(base_config):
    return [dict(step) for step in base_config]


def test_orchestrate_stops_after_max_failures(tmp_path, capsys):
    base_config = [{"type": "cmd", "cmd": "false"}]
    flow_configs = [_copy_flow(base_config) for _ in range(4)]

    with pytest.raises(orchestrator.MaxFlowFailuresExceeded):
        orchestrator.orchestrate(
            base_config,
            flow_configs,
            parallel=2,
            workdir=tmp_path,
            max_flow_failures=2,
        )

    captured = capsys.readouterr()
    assert "Maximum flow failures reached" in captured.out


def test_orchestrate_can_disable_flow_path_output(tmp_path, capsys, monkeypatch):
    base_config = [{"type": "cmd", "cmd": "printf hi"}]
    flow_configs = [_copy_flow(base_config)]

    monkeypatch.setattr(orchestrator, "GENERATED_DIR", tmp_path)

    orchestrator.orchestrate(
        base_config,
        flow_configs,
        parallel=1,
        workdir=tmp_path,
        print_flow_paths=False,
    )

    captured = capsys.readouterr()
    assert "flow_" not in captured.out


def test_orchestrate_groups_flows_in_run_directory(tmp_path, monkeypatch):
    base_config = [{"type": "cmd", "cmd": "printf hi"}]
    flow_configs = [_copy_flow(base_config) for _ in range(2)]

    monkeypatch.setattr(orchestrator, "GENERATED_DIR", tmp_path)

    results = orchestrator.orchestrate(
        base_config,
        flow_configs,
        parallel=2,
        workdir=tmp_path,
        print_flow_paths=False,
    )

    flow_dirs = {res[2] for res in results}
    assert len(flow_dirs) == len(flow_configs)

    run_dirs = {flow_dir.parent for flow_dir in flow_dirs}
    assert len(run_dirs) == 1

    run_dir = run_dirs.pop()
    assert run_dir.parent == tmp_path
    assert run_dir.name.startswith("run_")

    for flow_dir in flow_dirs:
        assert flow_dir.name.startswith("flow_")


def test_cmd_failure_forwards_stderr(tmp_path, capsys):
    script = tmp_path / "failing_script.py"
    script.write_text(
        "import sys\nsys.stderr.write('boom\\n')\nsys.exit(1)\n",
        encoding="utf-8",
    )
    cmd = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"
    config = [{"type": "cmd", "cmd": cmd}]

    orchestrator._run_flow(config, [0], threading.Lock(), tmp_path, tmp_path)

    captured = capsys.readouterr()
    assert "boom" in captured.err
