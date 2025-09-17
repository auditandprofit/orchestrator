from pathlib import Path

import orchestrator


def test_orchestrate_lists_codex_final_message_paths(tmp_path, monkeypatch, capsys):
    base_config = [{"type": "codex"}]
    flow_configs = [[dict(step) for step in base_config]]

    monkeypatch.setattr(orchestrator, "GENERATED_DIR", tmp_path)

    def fake_run_codex_cli(
        prompt: str,
        workdir: Path,
        output_dir: Path,
        timeout=None,
        max_retries: int = 3,
    ):
        exec_dir = output_dir / "codex_exec_test"
        exec_dir.mkdir(parents=True, exist_ok=True)
        final_path = exec_dir / "final_message.txt"
        final_path.write_text("result", encoding="utf-8")
        return "result", final_path

    monkeypatch.setattr(orchestrator, "run_codex_cli", fake_run_codex_cli)

    results = orchestrator.orchestrate(
        base_config,
        flow_configs,
        parallel=1,
        workdir=tmp_path,
        print_flow_paths=False,
        list_codex_final_paths=True,
    )

    captured = capsys.readouterr()
    assert len(results) == 1
    final_path = results[0][1]
    assert final_path is not None
    assert str(final_path.resolve()) in captured.out


def test_orchestrate_skips_listing_when_final_step_not_codex(
    tmp_path, capsys, monkeypatch
):
    base_config = [{"type": "cmd", "cmd": "printf hi"}]
    flow_configs = [[dict(step) for step in base_config]]

    monkeypatch.setattr(orchestrator, "GENERATED_DIR", tmp_path)

    orchestrator.orchestrate(
        base_config,
        flow_configs,
        parallel=1,
        workdir=tmp_path,
        print_flow_paths=False,
        list_codex_final_paths=True,
    )

    captured = capsys.readouterr()
    assert "final_message.txt" not in captured.out
