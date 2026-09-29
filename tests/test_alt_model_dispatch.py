"""Regression tests for the dead native-Ollama dispatch and alt-client fallback.

Background (2026-09-29): the deepseek-v4-flash:cloud alt model was retired
upstream (HTTP 410) and replaced with kimi-k2.6:cloud. Kimi is a reasoning
model; through the OpenAI-compatible path it burns the entire token budget on
`reasoning` and returns empty `content` (finish_reason=length). The native
Ollama path (which passes `think: False` and works for kimi) was unreachable:

1. ``_call_once`` compared the client's normalized ``base_url`` (always
   trailing-slash, e.g. ``http://host/v1/``) against the raw ``LLM_BASE_URL``
   env value (no trailing slash) — never equal, so ``is_ollama`` was always
   False and every call went through the OpenAI-compatible path.
2. ``make_client_alt()`` required ``LLM_BASE_URL_2``; when unset it returned
   None and priority docs silently reused the primary client regardless.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from resources import summarization

# ── _call_once dispatch ──────────────────────────────────────────────────────


def _mock_client_for(base_url: str) -> MagicMock:
    client = MagicMock()
    client.base_url = base_url
    return client


class TestCallOnceDispatch:
    def test_trailing_slash_base_url_dispatches_to_native_ollama(self, monkeypatch):
        """OpenAI SDK normalizes base_url with a trailing slash; the env value
        has none. dispatch must still detect the Ollama endpoint."""
        monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
        client = _mock_client_for("http://127.0.0.1:11434/v1/")
        called = {}

        def fake_native(messages, model, base_url, *, max_tokens, timeout):
            called["base_url"] = base_url
            return "native-ok"

        with patch.object(summarization, "_call_once_native_ollama", side_effect=fake_native):
            result = summarization._call_once(
                [{"role": "user", "content": "hi"}], "m:cloud", client, max_tokens=100
            )

        assert result == "native-ok"
        assert called["base_url"].rstrip("/") == "http://127.0.0.1:11434/v1"

    def test_non_ollama_base_url_still_uses_openai_client(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
        client = _mock_client_for("https://api.openai.com/v1/")
        called = {}

        def fake_native(*a, **k):  # pragma: no cover — must not be reached
            called["oops"] = True
            return "native-wrong"

        with patch.object(summarization, "_call_once_native_ollama", side_effect=fake_native):
            summarization._call_once(
                [{"role": "user", "content": "hi"}], "m", client, max_tokens=100
            )

        assert "oops" not in called
        client.chat.completions.create.assert_called_once()

    def test_missing_env_var_still_uses_openai_client(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        client = _mock_client_for("http://127.0.0.1:11434/v1/")
        with patch.object(summarization, "_call_once_native_ollama") as fake_native:
            summarization._call_once(
                [{"role": "user", "content": "hi"}], "m:cloud", client, max_tokens=100
            )
        fake_native.assert_not_called()
        client.chat.completions.create.assert_called_once()


# ── make_client_alt fallback ─────────────────────────────────────────────────


class TestMakeClientAlt:
    def test_falls_back_to_primary_base_url_when_base_url_2_unset(self, monkeypatch):
        """LLM_BASE_URL_2 absent → alt client should reuse the primary base URL
        (same Ollama proxy serves both models at this deployment)."""
        monkeypatch.delenv("LLM_BASE_URL_2", raising=False)
        monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
        client = summarization.make_client_alt()
        assert client is not None
        assert "127.0.0.1:11434/v1" in str(client.base_url)

    def test_explicit_base_url_2_still_wins(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
        monkeypatch.setenv("LLM_BASE_URL_2", "http://alt-host:9999/v1")
        client = summarization.make_client_alt()
        assert client is not None
        assert "alt-host:9999/v1" in str(client.base_url)

    def test_returns_none_when_neither_url_set(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL_2", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        assert summarization.make_client_alt() is None


# ── reasoning-model handling on the OpenAI-compatible path ───────────────────


class TestReasoningModelCompat:
    def test_reasoning_in_separate_field_does_not_hit_empty_content(self):
        """Kimi returns content plus a separate reasoning attr. The empty-content
        guard must look at ``content`` only — which it does — but a bare mock
        returning MagicMock for message would look non-empty; pin the contract."""
        response = MagicMock()
        response.choices = [
            MagicMock(
                finish_reason="stop",
                message=MagicMock(content="A summary.", reasoning="long thinking…"),
            )
        ]
        client = _mock_client_for("https://api.openai.com/v1/")
        client.chat.completions.create.return_value = response
        out = summarization._call_once(
            [{"role": "user", "content": "x"}], "kimi-k2.6:cloud", client, max_tokens=100
        )
        assert out == "A summary."

    def test_reasoning_effort_none_added_for_non_thinkable_models(self):
        """Reasoning models reached through the compat path must get
        reasoning_effort=none so thinking tokens cannot drain the budget."""
        response, client = MagicMock(), _mock_client_for("https://api.openai.com/v1/")
        response.choices = [
            MagicMock(finish_reason="stop", message=MagicMock(content="S", reasoning=""))
        ]
        client.chat.completions.create.return_value = response
        summarization._call_once(
            [{"role": "user", "content": "x"}], "kimi-k2.6:cloud", client, max_tokens=100
        )
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"]["reasoning_effort"] == "none"

    def test_reasoning_effort_none_not_duplicated(self):
        response, client = MagicMock(), _mock_client_for("https://api.openai.com/v1/")
        response.choices = [MagicMock(finish_reason="stop", message=MagicMock(content="S"))]
        client.chat.completions.create.return_value = response
        summarization._call_once(
            [{"role": "user", "content": "x"}],
            "kimi-k2.6:cloud",
            client,
            max_tokens=100,
        )
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"].get("reasoning_effort") == "none"
