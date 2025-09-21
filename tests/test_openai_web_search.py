import threading

import orchestrator


def test_openai_web_search_flag(tmp_path, monkeypatch):
    calls = []

    def fake_api(
        prompt: str,
        *,
        web_search: bool = False,
        max_retries: int = 3,
        model=None,
        reasoning_effort=None,
        service_tier=None,
    ) -> dict:
        calls.append(
            {
                "prompt": prompt,
                "web_search": web_search,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "service_tier": service_tier,
            }
        )
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
    assert calls[0]["web_search"] is True
    assert calls[1]["web_search"] is False
    assert calls[0]["model"] is None
    assert calls[0]["reasoning_effort"] is None
    assert calls[0]["service_tier"] is None


def test_openai_request_options_forwarding(tmp_path, monkeypatch):
    calls = []

    def fake_api(
        prompt: str,
        *,
        web_search: bool = False,
        max_retries: int = 3,
        model=None,
        reasoning_effort=None,
        service_tier=None,
    ) -> dict:
        calls.append(
            {
                "prompt": prompt,
                "web_search": web_search,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "service_tier": service_tier,
            }
        )
        return {"output": [{"content": [{"text": prompt}]}]}

    monkeypatch.setattr(orchestrator, "call_openai_api", fake_api)

    flow_dir = tmp_path / "flow"
    flow_dir.mkdir()

    res, failed = orchestrator._run_flow(
        [{"type": "openai", "prompt": "Hello"}],
        [0],
        threading.Lock(),
        tmp_path,
        flow_dir,
        openai_request_options={
            "model": "gpt-test",
            "service_tier": "scale",
            "reasoning_effort": "medium",
        },
    )

    assert not failed
    assert res[0][0] == "Hello"
    assert len(calls) == 1
    call = calls[0]
    assert call["model"] == "gpt-test"
    assert call["service_tier"] == "scale"
    assert call["reasoning_effort"] == "medium"
