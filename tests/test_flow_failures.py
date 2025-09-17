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
