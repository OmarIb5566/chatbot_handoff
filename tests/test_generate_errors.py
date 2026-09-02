"""Failures at the model boundary arrive as types the UI can act on.

Every test here fakes the HTTP layer: the point is the mapping from what the
server did to what the reader is told, and that mapping must hold whether or
not Ollama is running on the machine executing the tests.
"""
from __future__ import annotations

import pytest
import requests

import chatbot
from errors import (GenerationFailed, ModelNotFound, ModelTimeout,
                    ModelUnavailable, PipelineError)


class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _post(monkeypatch, result):
    def fake(*a, **kw):
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(chatbot.requests, "post", fake)


def test_a_timeout_is_not_the_same_as_a_dead_server(monkeypatch):
    """They send the reader to opposite places: wait or raise the budget,
    versus go and start Ollama."""
    _post(monkeypatch, requests.Timeout("timed out"))
    with pytest.raises(ModelTimeout):
        chatbot.generate("hi")


def test_a_connection_error_is_reported_as_unavailable(monkeypatch):
    _post(monkeypatch, requests.ConnectionError("refused"))
    with pytest.raises(ModelUnavailable):
        chatbot.generate("hi")


def test_a_missing_model_says_so(monkeypatch):
    """The most likely failure on a fresh machine, and the one that used to
    surface as a generic 'something went wrong'."""
    _post(monkeypatch, FakeResponse(404, text="model not found"))
    with pytest.raises(ModelNotFound) as e:
        chatbot.generate("hi", model="never-pulled:9b")
    assert "never-pulled:9b" in str(e.value)


def test_a_server_error_carries_the_status(monkeypatch):
    _post(monkeypatch, FakeResponse(500, text="boom"))
    with pytest.raises(GenerationFailed) as e:
        chatbot.generate("hi")
    assert "500" in str(e.value)


def test_a_200_that_is_not_json_does_not_become_an_empty_answer(monkeypatch):
    """A proxy or captive portal in front of the host. Returning "" here would
    render as an empty answer with no explanation anywhere."""
    _post(monkeypatch, FakeResponse(200, payload=None))
    with pytest.raises(GenerationFailed):
        chatbot.generate("hi")


def test_a_good_response_is_stripped_and_returned(monkeypatch):
    _post(monkeypatch, FakeResponse(200, payload={"response": "  answer  "}))
    assert chatbot.generate("hi") == "answer"


def test_every_expected_failure_carries_a_user_message():
    """The UI renders one string and does not re-derive wording per type."""
    for cls in (ModelUnavailable, ModelNotFound, ModelTimeout,
                GenerationFailed, PipelineError):
        assert cls.user_message and cls.user_message[0].isupper()
