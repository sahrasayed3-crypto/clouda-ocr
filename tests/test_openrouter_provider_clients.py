import pytest
import requests

from pdfword import openrouter_client
from pdfword.models import ModelEndpointError, OpenRouterEmptyContentError
from pdfword.provider_client import (
    OpenAICompatibleProviderClient,
    OpenRouterClient,
    get_provider_client,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="") -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeSession:
    def __init__(self, *, get_responses=None, post_responses=None) -> None:
        self.get_responses = list(get_responses or [])
        self.post_responses = list(post_responses or [])
        self.posts: list[dict] = []

    def get(self, *_args, **_kwargs):
        response = self.get_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def post(self, *_args, **kwargs):
        self.posts.append(kwargs)
        response = self.post_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_extract_text_variants_and_errors():
    assert (
        openrouter_client._extract_text_from_response(
            {"choices": [{"message": {"content": "  hello  "}}]}
        )
        == "hello"
    )
    assert (
        openrouter_client._extract_text_from_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"text": "part one"},
                                {"content": "part two"},
                                "part three",
                            ]
                        }
                    }
                ]
            }
        )
        == "part one\npart two\npart three"
    )
    assert (
        openrouter_client._extract_text_from_response({"choices": [{"text": "legacy"}]})
        == "legacy"
    )
    with pytest.raises(RuntimeError):
        openrouter_client._extract_text_from_response({"error": "bad request"})
    with pytest.raises(OpenRouterEmptyContentError):
        openrouter_client._extract_text_from_response(
            {"choices": [{"message": {"content": ""}}]}
        )


def test_discover_models_cache_success_and_fallback(monkeypatch, tmp_path):
    cache_path = tmp_path / "models.json"
    monkeypatch.setattr(openrouter_client, "MODEL_CACHE_PATH", cache_path)
    payload = {
        "data": [
            {
                "id": "vision/good",
                "name": "Vision Good",
                "architecture": {
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                },
                "pricing": {"prompt": "0.001", "completion": "0.002", "image": "0.01"},
                "context_length": 8000,
            },
            {"name": "missing id"},
        ]
    }
    monkeypatch.setattr(
        openrouter_client,
        "_get_http_session",
        lambda: FakeSession(get_responses=[FakeResponse(payload=payload)]),
    )

    models = openrouter_client.discover_openrouter_models(force=True)
    assert models[0]["id"] == "vision/good"
    assert models[0]["supports_vision"] is True
    assert cache_path.is_file()

    monkeypatch.setattr(
        openrouter_client,
        "_get_http_session",
        lambda: FakeSession(get_responses=[requests.ConnectionError("offline")]),
    )
    assert (
        openrouter_client.discover_openrouter_models(force=True)[0]["id"]
        == "vision/good"
    )

    cache_path.write_text("{bad json", encoding="utf-8")
    assert openrouter_client.discover_openrouter_models(force=False) == []


def test_model_selection_and_costs(monkeypatch):
    catalog = [
        {
            "id": "vision/fast",
            "supports_vision": True,
            "prompt_price": "0.1",
            "completion_price": "0.2",
            "image_price": "0.3",
        },
        {"id": "deepseek/text", "supports_vision": True},
        {"id": "text/only", "supports_vision": False},
    ]
    monkeypatch.setattr(
        openrouter_client, "discover_openrouter_models", lambda *a, **k: catalog
    )
    assert openrouter_client.get_openrouter_model_ids() == {
        "vision/fast",
        "deepseek/text",
        "text/only",
    }
    assert openrouter_client.verified_vision_model_ids(
        ["deepseek/text", "text/only", "vision/fast"]
    ) == ["vision/fast"]
    assert openrouter_client.estimate_model_cost("vision/fast", 2, 3) == pytest.approx(
        0.8
    )
    assert openrouter_client.estimate_model_cost("missing", 2, 3) is None
    assert openrouter_client.estimate_vision_request_cost(
        "vision/fast", 10, estimated_prompt_tokens=20
    ) == pytest.approx(5.375)


def test_vision_route_rejects_missing_pricing_before_request(monkeypatch):
    monkeypatch.setattr(
        openrouter_client,
        "discover_openrouter_models",
        lambda *a, **k: [
            {
                "id": "vision/missing-price",
                "supports_vision": True,
                "prompt_price": "0.1",
                "completion_price": "0.2",
            }
        ],
    )
    called = False

    def fake_post(*_args, **_kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(openrouter_client, "_post_with_retries", fake_post)

    with pytest.raises(ModelEndpointError):
        openrouter_client.openrouter_chat_with_image(
            "key",
            "vision/missing-price",
            "system",
            "user",
            "abc",
            max_tokens=10,
        )
    assert called is False


def test_post_with_retries_telemetry_and_endpoint_error(monkeypatch):
    session = FakeSession(
        post_responses=[
            FakeResponse(status_code=429),
            FakeResponse(
                payload={
                    "id": "req1",
                    "model": "vision/fast",
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3},
                    "choices": [{"message": {"content": "ok"}}],
                }
            ),
        ]
    )
    monkeypatch.setattr(openrouter_client, "_get_http_session", lambda: session)
    monkeypatch.setattr(openrouter_client.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(openrouter_client, "estimate_model_cost", lambda *_args: 1.25)
    events = []
    with openrouter_client.capture_openrouter_telemetry(events.append):
        data = openrouter_client._post_with_retries(
            {"model": "vision/fast"}, {"Authorization": "Bearer test"}
        )
    assert data["id"] == "req1"
    assert events[0]["cost"] == 1.25
    assert events[0]["cost_is_estimated"] is True

    monkeypatch.setattr(
        openrouter_client,
        "_get_http_session",
        lambda: FakeSession(
            post_responses=[
                FakeResponse(status_code=404, text="no endpoints found for model")
            ]
        ),
    )
    with pytest.raises(ModelEndpointError):
        openrouter_client._post_with_retries({"model": "missing"}, {})


def test_openrouter_chat_retries_empty_content(monkeypatch):
    calls = []

    def fake_post(payload, headers):
        calls.append((payload, headers))
        if len(calls) == 1:
            return {"choices": [{"message": {"content": ""}}]}
        return {"choices": [{"message": {"content": "done"}}]}

    monkeypatch.setattr(openrouter_client, "_post_with_retries", fake_post)
    monkeypatch.setattr(openrouter_client.time, "sleep", lambda *_args: None)
    assert (
        openrouter_client.openrouter_chat_text(
            "key", "model", "system", "user", max_tokens=5
        )
        == "done"
    )
    assert calls[0][0]["messages"][1]["content"] == "user"


def test_openai_compatible_provider_success_and_retry(monkeypatch):
    responses = [
        FakeResponse(status_code=500),
        FakeResponse(
            payload={"choices": [{"message": {"content": [{"text": "provider ok"}]}}]}
        ),
    ]
    posts = []

    def fake_post(*_args, **kwargs):
        posts.append(kwargs)
        return responses.pop(0)

    monkeypatch.setattr("pdfword.provider_client.requests.post", fake_post)
    monkeypatch.setattr("pdfword.provider_client.time.sleep", lambda *_args: None)
    client = OpenAICompatibleProviderClient(
        name="local",
        api_key="key",  # pragma: allowlist secret
        api_url="https://example.invalid",
    )
    assert (
        client.chat_with_image(
            model="m",
            system_prompt="s",
            user_text="u",
            image_b64="abc",
            max_tokens=8,
        )
        == "provider ok"
    )
    assert len(posts) == 2
    assert posts[-1]["json"]["messages"][1]["content"][1]["image_url"][
        "url"
    ].startswith("data:image/png;base64,")

    monkeypatch.setattr(
        "pdfword.provider_client.requests.post",
        lambda *_args, **_kwargs: FakeResponse(payload={"choices": []}),
    )
    with pytest.raises(RuntimeError):
        client.chat_text(model="m", system_prompt="s", user_text="u", max_tokens=8)


def test_openrouter_client_wrapper_and_provider_factory(monkeypatch):
    monkeypatch.setattr(
        "pdfword.provider_client.resolve_models",
        lambda: ("fast", "accurate", False),
    )
    monkeypatch.setattr(
        "pdfword.provider_client.openrouter_chat_text",
        lambda **kwargs: f"text:{kwargs['model']}",
    )
    monkeypatch.setattr(
        "pdfword.provider_client.openrouter_chat",
        lambda **kwargs: f"image:{kwargs['model']}",
    )
    client = OpenRouterClient(api_key="primary")
    assert client.resolve_models("df", "da") == ("fast", "accurate", False)
    assert (
        client.chat_text(model="m", system_prompt="s", user_text="u", max_tokens=3)
        == "text:m"
    )
    assert (
        client.chat(model="m", system_prompt="s", image_b64="i", max_tokens=3)
        == "image:m"
    )

    assert get_provider_client("openrouter", "primary").name == "openrouter"
    together = get_provider_client("together", "primary")
    assert together.name == "together"
    assert get_provider_client("unknown", "primary").name == "openrouter"
