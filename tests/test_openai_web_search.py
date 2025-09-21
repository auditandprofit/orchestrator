import threading

import orchestrator


def test_openai_web_search_flag(tmp_path, monkeypatch):
    calls = []

    def fake_api(prompt: str, *, web_search: bool = False, max_retries: int = 3) -> dict:
        calls.append((prompt, web_search))
        return {"output": [{"content": [{"text": prompt}]}]}

    monkeypatch.setattr(orchestrator, "call_openai_api", fake_api)

    flow_dir = tmp_path / "flow"
    flow_dir.mkdir()

    res, failed = orchestrator._run_flow(
        [
            {"type": "openai", "prompt": "First", "web_search": True},
            {"type": "openai", "prompt": "Second"},
        ],
        [0, 0],
        threading.Lock(),
        tmp_path,
        flow_dir,
    )

    assert not failed
    assert len(res) == 1
    assert len(calls) == 2
    assert calls[0][1] is True
    assert calls[1][1] is False
