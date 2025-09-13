import threading
from pathlib import Path

import orchestrator


def fake_api(prompt: str) -> dict:
    return {"output": [{"content": [{"text": prompt}]}]}


def test_prmpt_file_placeholders(tmp_path, monkeypatch):
    template = tmp_path / "template.txt"
    template.write_text("Hello {{{name}}}!", encoding="utf-8")

    names_file = tmp_path / "names.txt"
    names_file.write_text(str(tmp_path / "name.txt") + "\n", encoding="utf-8")

    (tmp_path / "name.txt").write_text("World", encoding="utf-8")

    base_config = [{"type": "openai", "prmpt_file": str(template)}]
    key_files = {"name": names_file}
    flows = orchestrator._generate_flow_configs(base_config, key_files)

    monkeypatch.setattr(orchestrator, "call_openai_api", fake_api)

    res = orchestrator._run_flow(flows[0], [0], threading.Lock(), tmp_path, tmp_path)
    assert res[0][0] == "Hello World!"
