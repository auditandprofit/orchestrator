import io

import openai_utils


def test_run_codex_cli_falls_back_to_stdout(tmp_path, monkeypatch):
    prompts = []

    def fake_popen(cmd, stdout, stderr, text):
        prompts.append(cmd[-1])

        class DummyProcess:
            def __init__(self) -> None:
                self.args = cmd
                self.returncode = 0
                self.stdout = io.StringIO("final output\n")
                self.stderr = io.StringIO("")

            def wait(self, timeout=None):
                return 0

            def kill(self):
                return None

        return DummyProcess()

    monkeypatch.setattr(openai_utils.subprocess, "Popen", fake_popen)

    message, path = openai_utils.run_codex_cli("Test prompt", tmp_path, tmp_path)

    assert prompts == ["Test prompt"]
    assert path.read_text(encoding="utf-8") == "final output\n"
    assert message == "final output\n"

    exec_dirs = list(tmp_path.glob("codex_exec_*"))
    assert len(exec_dirs) == 1
    stdout_path = exec_dirs[0] / "stdout.txt"
    assert stdout_path.read_text(encoding="utf-8") == "final output\n"
