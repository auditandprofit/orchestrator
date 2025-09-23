import threading
from pathlib import Path
import orchestrator

def test_cmd_output_saved(tmp_path):
    config = [
        {"type": "cmd", "cmd": "printf '[\"a\",\"b\"]'", "array": True},
        {"type": "cmd", "cmd": "cat"},
    ]
    results, failed = orchestrator._run_flow(
        config, [0, 0], threading.Lock(), tmp_path, tmp_path
    )

    assert (tmp_path / "step_0_cmd.txt").read_text() == '["a","b"]'
    assert (tmp_path / "branch_0" / "step_1_cmd.txt").read_text() == "a"
    assert (tmp_path / "branch_1" / "step_1_cmd.txt").read_text() == "b"
    assert sorted(r[0] for r in results) == ["a", "b"]
    assert not failed


def test_cmd_respects_process_cwd(tmp_path, monkeypatch):
    workdir = tmp_path / "workdir"
    workdir.mkdir()

    default_dir = tmp_path / "default"
    default_dir.mkdir()
    data_file = default_dir / "data.txt"
    data_file.write_text("hello", encoding="utf-8")

    monkeypatch.chdir(default_dir)

    config = [{"type": "cmd", "cmd": "cat data.txt"}]

    results, failed = orchestrator._run_flow(
        config, [0], threading.Lock(), workdir, tmp_path
    )

    assert not failed
    assert results[0][0] == "hello"


def test_cmd_inputs_can_select_named_steps(tmp_path):
    config = [
        {"type": "cmd", "cmd": "printf alpha", "name": "first"},
        {"type": "cmd", "cmd": "printf beta", "name": "second"},
        {"type": "cmd", "cmd": "cat", "inputs": ["second", "first"]},
    ]

    results, failed = orchestrator._run_flow(
        config, [0, 0, 0], threading.Lock(), tmp_path, tmp_path
    )

    assert not failed
    assert results[0][0] == "beta\nalpha"


def test_cmd_inputs_accept_step_indexes(tmp_path):
    config = [
        {"type": "cmd", "cmd": "printf foo"},
        {"type": "cmd", "cmd": "printf bar"},
        {"type": "cmd", "cmd": "cat", "inputs": [1, 0]},
    ]

    results, failed = orchestrator._run_flow(
        config, [0, 0, 0], threading.Lock(), tmp_path, tmp_path
    )

    assert not failed
    assert results[0][0] == "bar\nfoo"
