import threading
from pathlib import Path
import orchestrator

def test_cmd_output_saved(tmp_path):
    config = [
        {"type": "cmd", "cmd": "printf '[\"a\",\"b\"]'", "array": True},
        {"type": "cmd", "cmd": "cat"},
    ]
    results = orchestrator._run_flow(config, [0, 0], threading.Lock(), tmp_path, tmp_path)

    assert (tmp_path / "step_0_cmd.txt").read_text() == '["a","b"]'
    assert (tmp_path / "branch_0" / "step_1_cmd.txt").read_text() == "a"
    assert (tmp_path / "branch_1" / "step_1_cmd.txt").read_text() == "b"
    assert sorted(r[0] for r in results) == ["a", "b"]
