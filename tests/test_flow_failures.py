import json
import shlex
import subprocess
import sys
import threading
from pathlib import Path

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


def test_orchestrate_honors_max_flows(tmp_path, monkeypatch):
    base_config = [{"type": "cmd", "cmd": "printf hi"}]
    flow_configs = [_copy_flow(base_config) for _ in range(5)]

    monkeypatch.setattr(orchestrator, "GENERATED_DIR", tmp_path)

    results = orchestrator.orchestrate(
        base_config,
        flow_configs,
        parallel=2,
        workdir=tmp_path,
        print_flow_paths=False,
        max_flows=2,
    )

    assert len(results) == 2

    flow_dirs = {res[2] for res in results}
    assert len(flow_dirs) == 2

    run_dirs = {flow_dir.parent for flow_dir in flow_dirs}
    assert len(run_dirs) == 1
    run_dir = run_dirs.pop()

    executed_flow_dirs = [
        path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("flow_")
    ]
    assert len(executed_flow_dirs) == 2


def test_orchestrate_rejects_negative_max_flows(tmp_path):
    base_config = [{"type": "cmd", "cmd": "printf hi"}]
    flow_configs = [_copy_flow(base_config)]

    with pytest.raises(ValueError):
        orchestrator.orchestrate(
            base_config,
            flow_configs,
            parallel=1,
            workdir=tmp_path,
            max_flows=-1,
        )


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


def test_cmd_failure_writes_stderr_file(tmp_path):
    script = tmp_path / "failing_script.py"
    script.write_text(
        "import sys\nsys.stderr.write('boom\\n')\nsys.exit(3)\n",
        encoding="utf-8",
    )
    cmd = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"
    config = [{"type": "cmd", "cmd": cmd}]

    results, failed = orchestrator._run_flow(
        config, [0], threading.Lock(), tmp_path, tmp_path
    )

    assert failed
    error_file = results[0][1]
    assert error_file is not None
    stderr_file = error_file.with_name(f"{error_file.stem}_stderr.txt")
    assert stderr_file.exists()
    content = stderr_file.read_text(encoding="utf-8")
    assert "exit_code: 3" in content
    assert "boom" in content


def test_flow_exits_on_empty_response(tmp_path, capsys):
    sentinel = tmp_path / "sentinel.txt"
    base_config = [
        {
            "type": "cmd",
            "cmd": "printf ''",
            "exit_on_empty_response": True,
            "name": "empty_step",
        },
        {
            "type": "cmd",
            "cmd": f"printf done > {shlex.quote(str(sentinel))}",
        },
    ]

    results, failed = orchestrator._run_flow(
        base_config,
        [0 for _ in base_config],
        threading.Lock(),
        tmp_path,
        tmp_path,
    )

    captured = capsys.readouterr()

    assert not failed
    assert not sentinel.exists()
    assert "empty response" in captured.out.lower()

    assert len(results) == 1
    output, log_path, flow_dir = results[0]
    assert output == ""
    assert flow_dir == tmp_path
    assert log_path is not None
    assert log_path.read_text(encoding="utf-8").strip().endswith(
        "empty_step produced an empty response."
    )

def test_flow_exits_on_response_signal(tmp_path, capsys):
    sentinel = tmp_path / "sentinel.txt"
    base_config = [
        {
            "type": "cmd",
            "cmd": "printf 'continue kill-signal'",
            "exit_on_response_contains": "kill-signal",
            "name": "kill_step",
        },
        {
            "type": "cmd",
            "cmd": f"printf done > {shlex.quote(str(sentinel))}",
        },
    ]

    results, failed = orchestrator._run_flow(
        base_config,
        [0 for _ in base_config],
        threading.Lock(),
        tmp_path,
        tmp_path,
    )

    captured = capsys.readouterr()

    assert not failed
    assert not sentinel.exists()
    assert "exit signal" in captured.out.lower()

    assert len(results) == 1
    output, log_path, flow_dir = results[0]
    assert output == ""
    assert flow_dir == tmp_path
    assert log_path is not None
    log_content = log_path.read_text(encoding="utf-8").strip()
    assert "kill_step" in log_content
    assert "kill-signal" in log_content


def test_codex_failure_writes_stderr_file(tmp_path, monkeypatch):
    def fake_run_codex_cli(prompt, workdir, output_dir, max_retries=3, timeout=None):
        err = subprocess.CalledProcessError(2, ["codex", "exec"], stderr="codex boom\n")
        raise Exception("codex boom") from err

    monkeypatch.setattr(orchestrator, "run_codex_cli", fake_run_codex_cli)

    config = [{"type": "codex", "prompt": ""}]

    results, failed = orchestrator._run_flow(
        config, [0], threading.Lock(), tmp_path, tmp_path
    )

    assert failed
    error_file = results[0][1]
    assert error_file is not None
    stderr_file = error_file.with_name(f"{error_file.stem}_stderr.txt")
    assert stderr_file.exists()
    content = stderr_file.read_text(encoding="utf-8")
    assert "exit_code: 2" in content
    assert "codex boom" in content


def test_orchestrate_can_ignore_max_failures(tmp_path, monkeypatch):
    base_config = [{"type": "cmd", "cmd": "false"}]
    flow_configs = [_copy_flow(base_config) for _ in range(3)]

    monkeypatch.setattr(orchestrator, "GENERATED_DIR", tmp_path)

    results = orchestrator.orchestrate(
        base_config,
        flow_configs,
        parallel=2,
        workdir=tmp_path,
        max_flow_failures=1,
        halt_on_max_failures=False,
    )

    assert len(results) == len(flow_configs)
    for _, path, flow_dir in results:
        assert path is not None
        assert "errors" in str(path)
        assert (flow_dir / "flow_failed.txt").exists()


def test_failed_files_logs_interpolated_paths(tmp_path, monkeypatch):
    file_a = tmp_path / "a.txt"
    file_a.write_text("a", encoding="utf-8")
    file_b = tmp_path / "b.txt"
    file_b.write_text("b", encoding="utf-8")

    file_list = tmp_path / "files.txt"
    file_list.write_text(f"{file_a}\n{file_b}\n", encoding="utf-8")

    base_config = [{"type": "cmd", "cmd": "false"}]
    flows = orchestrator._generate_flow_configs(base_config, {"name": file_list})

    monkeypatch.setattr(orchestrator, "GENERATED_DIR", tmp_path)

    with pytest.raises(orchestrator.MaxFlowFailuresExceeded):
        orchestrator.orchestrate(
            base_config,
            flows,
            parallel=1,
            workdir=tmp_path,
            max_flow_failures=1,
        )

    run_dirs = list(tmp_path.glob("run_*"))
    assert len(run_dirs) == 1
    failed_file = run_dirs[0] / "failed_files"
    assert failed_file.exists()

    lines = failed_file.read_text(encoding="utf-8").splitlines()
    assert lines == [str(file_a)]


def test_failed_files_multiple_keys(tmp_path, monkeypatch):
    first_path = tmp_path / "first.txt"
    first_path.write_text("first", encoding="utf-8")
    second_path = tmp_path / "second.txt"
    second_path.write_text("second", encoding="utf-8")

    first_list = tmp_path / "first_list.txt"
    first_list.write_text(f"{first_path}\n", encoding="utf-8")
    second_list = tmp_path / "second_list.txt"
    second_list.write_text(f"{second_path}\n", encoding="utf-8")

    base_config = [{"type": "cmd", "cmd": "false"}]
    flows = orchestrator._generate_flow_configs(
        base_config,
        {"alpha": first_list, "beta": second_list},
    )

    monkeypatch.setattr(orchestrator, "GENERATED_DIR", tmp_path)

    orchestrator.orchestrate(
        base_config,
        flows,
        parallel=1,
        workdir=tmp_path,
        max_flow_failures=1,
        halt_on_max_failures=False,
    )

    run_dirs = list(tmp_path.glob("run_*"))
    assert len(run_dirs) == 1
    failed_file = run_dirs[0] / "failed_files"
    assert failed_file.exists()

    lines = failed_file.read_text(encoding="utf-8").splitlines()
    assert lines == [f"{first_path},{second_path}"]


def test_finished_file_records_flow_status(tmp_path, monkeypatch):
    success_cmd = tmp_path / "success_cmd.txt"
    success_cmd.write_text("printf success", encoding="utf-8")
    fail_cmd = tmp_path / "fail_cmd.txt"
    fail_cmd.write_text("false", encoding="utf-8")

    cmd_list = tmp_path / "cmds.txt"
    cmd_list.write_text(f"{success_cmd}\n{fail_cmd}\n", encoding="utf-8")

    base_config = [{"type": "cmd", "cmd": "{{{cmd}}}"}]
    flows = orchestrator._generate_flow_configs(base_config, {"cmd": cmd_list})

    monkeypatch.setattr(orchestrator, "GENERATED_DIR", tmp_path)

    orchestrator.orchestrate(
        base_config,
        flows,
        parallel=1,
        workdir=tmp_path,
        print_flow_paths=False,
        max_flow_failures=2,
    )

    run_dirs = list(tmp_path.glob("run_*"))
    assert len(run_dirs) == 1
    finished_file = run_dirs[0] / "finished.txt"
    assert finished_file.exists()
    lines = finished_file.read_text(encoding="utf-8").splitlines()
    assert lines == [f"done {success_cmd}", f"failed {fail_cmd}"]


def test_finished_file_is_cleared_between_runs(tmp_path, monkeypatch):
    base_config = [{"type": "cmd", "cmd": "printf hi"}]
    flow_configs = [_copy_flow(base_config)]

    run_dir = tmp_path / "run_shared"
    first_flow_dir = run_dir / "flow_first"
    second_flow_dir = run_dir / "flow_second"
    mkdtemp_paths = iter(
        [
            str(run_dir),
            str(first_flow_dir),
            str(run_dir),
            str(second_flow_dir),
        ]
    )

    def fake_mkdtemp(prefix="", dir=None):
        path = Path(next(mkdtemp_paths))
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    monkeypatch.setattr(orchestrator.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(orchestrator, "GENERATED_DIR", tmp_path)

    orchestrator.orchestrate(
        base_config,
        flow_configs,
        parallel=1,
        workdir=tmp_path,
        print_flow_paths=False,
    )

    finished_file = run_dir / "finished.txt"
    assert finished_file.read_text(encoding="utf-8").splitlines() == ["done"]

    orchestrator.orchestrate(
        base_config,
        flow_configs,
        parallel=1,
        workdir=tmp_path,
        print_flow_paths=False,
    )

    assert finished_file.read_text(encoding="utf-8").splitlines() == ["done"]


def test_cancelled_branch_flow_is_marked_failed(tmp_path, monkeypatch):
    array_script = tmp_path / "array_script.py"
    array_script.write_text(
        "import json\nprint(json.dumps(['branch']))\n", encoding="utf-8"
    )

    slow_script = tmp_path / "slow_script.py"
    slow_script.write_text(
        (
            "import sys\nimport time\n"
            "time.sleep(0.3)\n"
            "print('slow-done')\n"
        ),
        encoding="utf-8",
    )

    fail_script = tmp_path / "fail_script.py"
    fail_script.write_text(
        "import time\nimport sys\ntime.sleep(0.1)\nsys.exit(1)\n",
        encoding="utf-8",
    )

    array_cmd = f"{shlex.quote(sys.executable)} {shlex.quote(str(array_script))}"
    slow_cmd = f"{shlex.quote(sys.executable)} {shlex.quote(str(slow_script))}"
    fail_cmd = f"{shlex.quote(sys.executable)} {shlex.quote(str(fail_script))}"

    base_config = [
        {"type": "cmd", "cmd": array_cmd, "array": True},
        {"type": "cmd", "cmd": slow_cmd},
    ]

    flow_fail = _copy_flow(base_config)
    flow_fail[1]["cmd"] = fail_cmd
    flow_slow = _copy_flow(base_config)

    monkeypatch.setattr(orchestrator, "GENERATED_DIR", tmp_path)

    with pytest.raises(orchestrator.MaxFlowFailuresExceeded):
        orchestrator.orchestrate(
            base_config,
            [flow_fail, flow_slow],
            parallel=2,
            workdir=tmp_path,
            max_flow_failures=1,
        )

    run_dirs = list(tmp_path.glob("run_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    finished_file = run_dir / "finished.txt"
    assert finished_file.exists()
    lines = finished_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(line.startswith("failed") for line in lines)

    flow_dirs = list(run_dir.glob("flow_*"))
    assert len(flow_dirs) == 2
    for flow_dir in flow_dirs:
        assert (flow_dir / "flow_failed.txt").exists()


def test_cli_ignore_max_failures_flag(tmp_path):
    cmd = f"{shlex.quote(sys.executable)} -c {shlex.quote('import sys; sys.stderr.write("boom\\n"); sys.exit(1)')}"
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps([{"type": "cmd", "cmd": cmd}]),
        encoding="utf-8",
    )

    script_path = Path(orchestrator.__file__).resolve()
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(config_path),
            "--workdir",
            str(tmp_path),
            "--max-flow-failures",
            "1",
            "--ignore-max-failures",
            "--hide-flow-paths",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert "boom" in result.stderr
    assert "errors" in result.stdout
    assert "Maximum flow failures reached" not in result.stdout
