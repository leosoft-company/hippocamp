"""Reflection must not reach a cloud model unless explicitly allowed."""

import io
import json

import pytest

from hippocamp import reflect
from hippocamp.cli import _resolve_llm


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("HIPPOCAMP_ALLOW_CLOUD_REFLECT", "HIPPOCAMP_LLM_URL",
              "HIPPOCAMP_LLM_MODEL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)


@pytest.mark.parametrize("name", [None, "auto", "local"])
@pytest.mark.parametrize("api_key", [None, "sk-test"])
def test_default_resolution_is_always_local(monkeypatch, name, api_key):
    """No argument, 'auto' and 'local' all mean the local adapter — with or
    without an Anthropic key in the environment."""
    if api_key:
        monkeypatch.setenv("ANTHROPIC_API_KEY", api_key)
    llm = _resolve_llm(name)
    assert type(llm).__name__ == "_LocalLLM"
    assert llm.base_url == reflect.DEFAULT_LOCAL_URL


@pytest.mark.parametrize("name", ["anthropic", "claude", "claude-cli", "cli"])
def test_cloud_adapters_refuse_without_opt_in(monkeypatch, name):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with pytest.raises(RuntimeError, match="HIPPOCAMP_ALLOW_CLOUD_REFLECT"):
        _resolve_llm(name)


def test_cloud_adapter_functions_refuse_when_called_directly():
    with pytest.raises(RuntimeError, match="cloud model"):
        reflect.default_anthropic_llm()
    with pytest.raises(RuntimeError, match="cloud model"):
        reflect.claude_cli_llm()


def test_claude_cli_adapter_builds_only_with_opt_in(monkeypatch):
    monkeypatch.setenv("HIPPOCAMP_ALLOW_CLOUD_REFLECT", "1")
    monkeypatch.setattr(reflect.shutil, "which", lambda name: "/usr/bin/true")
    assert reflect.claude_cli_llm() is not None


@pytest.mark.parametrize("url", [
    "https://api.anthropic.com", "http://192.168.0.12:1234",
    "http://example.com:1234", "http://127.0.0.1.evil.com:1234",
])
def test_local_adapter_refuses_non_loopback(url, monkeypatch):
    with pytest.raises(RuntimeError, match="loopback"):
        reflect.local_llm(base_url=url)
    monkeypatch.setenv("HIPPOCAMP_LLM_URL", url)
    with pytest.raises(RuntimeError, match="loopback"):
        reflect.local_llm()


@pytest.mark.parametrize("url", ["http://127.0.0.1:1234", "http://localhost:8080/", "http://[::1]:1234"])
def test_local_adapter_accepts_loopback(url):
    assert reflect.local_llm(base_url=url) is not None


def test_local_adapter_talks_only_to_the_local_server(monkeypatch):
    """Picks the first chat (non-embedding) model and posts the prompt there."""
    import urllib.request

    seen = []

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data.decode()) if req.data else None
        seen.append((req.full_url, body))
        if req.full_url.endswith("/v1/models"):
            payload = {"data": [{"id": "text-embedding-nomic"}, {"id": "gemma-local"}]}
        else:
            payload = {"choices": [{"message": {"content": "{\"facts\": []}"}}]}
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    llm = reflect.local_llm()
    assert llm.complete("episodes…", system="be terse") == '{"facts": []}'
    assert [u for u, _ in seen] == [
        "http://127.0.0.1:1234/v1/models",
        "http://127.0.0.1:1234/v1/chat/completions",
    ]
    sent = seen[1][1]
    assert sent["model"] == "gemma-local"
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]


def test_local_adapter_reports_a_server_that_is_down(monkeypatch):
    import urllib.error
    import urllib.request

    def refuse(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with pytest.raises(RuntimeError, match="did not answer"):
        reflect.local_llm(model="m").complete("x")
