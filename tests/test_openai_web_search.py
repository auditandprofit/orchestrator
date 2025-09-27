import json
import threading
from types import SimpleNamespace

import orchestrator
import openai_utils


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
    assert res[0][1] == flow_dir / "step_1_openai.txt"
    assert len(calls) == 2
    assert calls[0]["web_search"] is True
    assert calls[1]["web_search"] is False
    assert calls[0]["model"] is None
    assert calls[0]["reasoning_effort"] is None
    assert calls[0]["service_tier"] is None

    first_text_path = flow_dir / "step_0_openai.txt"
    second_text_path = flow_dir / "step_1_openai.txt"
    assert first_text_path.read_text(encoding="utf-8") == "First"
    assert second_text_path.read_text(encoding="utf-8") == "Second\nFirst"

    first_response_path = flow_dir / "step_0_openai_response.json"
    second_response_path = flow_dir / "step_1_openai_response.json"
    first_response = json.loads(first_response_path.read_text(encoding="utf-8"))
    second_response = json.loads(second_response_path.read_text(encoding="utf-8"))
    assert first_response["output"][0]["content"][0]["text"] == "First"
    assert second_response["output"][0]["content"][0]["text"] == "Second\nFirst"


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
    assert res[0][1] == flow_dir / "step_0_openai.txt"
    assert len(calls) == 1
    call = calls[0]
    assert call["model"] == "gpt-test"
    assert call["service_tier"] == "scale"
    assert call["reasoning_effort"] == "medium"


def test_openai_response_buckets_parsing(tmp_path, monkeypatch):
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
        call = {
            "prompt": prompt,
            "web_search": web_search,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "service_tier": service_tier,
        }
        calls.append(call)
        if len(calls) == 1:
            bucket_payload = {"summary": "Primary", "code": "print('ok')"}
            text = json.dumps(bucket_payload)
        else:
            text = prompt
        return {"output": [{"content": [{"text": text}]}]}

    monkeypatch.setattr(orchestrator, "call_openai_api", fake_api)

    flow_dir = tmp_path / "flow"
    flow_dir.mkdir()

    res, failed = orchestrator._run_flow(
        [
            {
                "type": "openai",
                "prompt": "Describe",
                "response_buckets": {"summary": {}, "code": {}},
                "primary_bucket": "summary",
            },
            {
                "type": "openai",
                "prompt": "Followup",
                "inputs": ["step_0.summary", "step_0.code"],
            },
        ],
        [0, 0],
        threading.Lock(),
        tmp_path,
        flow_dir,
    )

    assert not failed
    assert len(calls) == 2
    assert res[0][0] == "Followup\nPrimary\nprint('ok')"

    first_output_path = flow_dir / "step_0_openai.txt"
    assert first_output_path.read_text(encoding="utf-8") == "Primary"

    summary_file = flow_dir / "step_0_openai_bucket_summary.txt"
    code_file = flow_dir / "step_0_openai_bucket_code.txt"
    assert summary_file.read_text(encoding="utf-8") == "Primary"
    assert code_file.read_text(encoding="utf-8") == "print('ok')"

    followup_prompt = calls[1]["prompt"]
    assert followup_prompt == "Followup\nPrimary\nprint('ok')"


def test_openai_response_buckets_fallback(tmp_path, monkeypatch):
    def fake_api(
        prompt: str,
        *,
        web_search: bool = False,
        max_retries: int = 3,
        model=None,
        reasoning_effort=None,
        service_tier=None,
    ) -> dict:
        return {"output": [{"content": [{"text": "Not JSON"}]}]}

    monkeypatch.setattr(orchestrator, "call_openai_api", fake_api)

    flow_dir = tmp_path / "flow"
    flow_dir.mkdir()

    res, failed = orchestrator._run_flow(
        [
            {
                "type": "openai",
                "prompt": "Describe",
                "response_buckets": ["summary", "code"],
            }
        ],
        [0],
        threading.Lock(),
        tmp_path,
        flow_dir,
    )

    assert not failed
    assert res[0][0] == "Not JSON"
    bucket_path = flow_dir / "step_0_openai_bucket_summary.txt"
    assert not bucket_path.exists()


class _DummyResponse:
    def __init__(self, data: dict) -> None:
        self._data = data

    def model_dump(self) -> dict:
        return self._data


class _DummyResponsesClient:
    def __init__(self) -> None:
        self.last_kwargs: dict = {}

    def create(self, **kwargs):  # type: ignore[no-untyped-def]
        self.last_kwargs = kwargs
        return _DummyResponse({"output": []})


def test_call_openai_api_uses_preview_tool_with_type(monkeypatch):
    dummy_responses = _DummyResponsesClient()
    dummy_client = SimpleNamespace(responses=dummy_responses)
    monkeypatch.setattr(openai_utils, "client", dummy_client)

    class DummyWebSearchTool:
        def __init__(self, *, type: str) -> None:
            self.type = type

    monkeypatch.setattr(openai_utils, "WebSearchTool", DummyWebSearchTool)

    openai_utils.call_openai_api("Prompt", web_search=True)

    tools = dummy_responses.last_kwargs.get("tools")
    assert isinstance(tools, list)
    assert len(tools) == 1
    assert getattr(tools[0], "type") == "web_search_preview"


def test_call_openai_api_uses_preview_tool_without_type(monkeypatch):
    dummy_responses = _DummyResponsesClient()
    dummy_client = SimpleNamespace(responses=dummy_responses)
    monkeypatch.setattr(openai_utils, "client", dummy_client)
    monkeypatch.setattr(openai_utils, "WebSearchTool", None)

    openai_utils.call_openai_api("Prompt", web_search=True)

    tools = dummy_responses.last_kwargs.get("tools")
    assert tools == [{"type": "web_search_preview"}]
