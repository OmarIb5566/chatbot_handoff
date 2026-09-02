"""Configuration reads the environment, and a bad value does not take the
process down before there is any logging to say why."""
from __future__ import annotations

import importlib
import logging

import pytest

import config as config_module


def _reload(monkeypatch, **env):
    """Re-import config under a given environment.

    config reads os.environ once at import, deliberately - a process gets one
    consistent configuration for its lifetime - so testing the reading means
    re-importing rather than mutating attributes.
    """
    for k in [k for k in dict(__import__("os").environ) if k.startswith("RME_")]:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return importlib.reload(config_module)


def test_defaults_are_the_tuned_values(monkeypatch):
    c = _reload(monkeypatch)
    assert c.TOP_K == 6
    assert c.CONTEXT_BUDGET_CHARS == 18_000
    assert c.NUM_CTX == 8192            # 4096 silently truncates the prompt
    assert c.POLICY == "flag"
    assert c.LOG_CONTENT is False       # user content is opt-in, never default


def test_the_environment_overrides(monkeypatch):
    c = _reload(monkeypatch, RME_OLLAMA_HOST="http://gpu-box:11434",
                RME_MODEL="qwen3.6:27b", RME_TOP_K="9")
    assert c.OLLAMA_HOST == "http://gpu-box:11434"
    assert c.MODEL == "qwen3.6:27b"
    assert c.TOP_K == 9


def test_the_rewrite_model_follows_the_main_model_unless_set(monkeypatch):
    assert _reload(monkeypatch, RME_MODEL="a:1b").REWRITE_MODEL == "a:1b"
    c = _reload(monkeypatch, RME_MODEL="a:1b", RME_REWRITE_MODEL="b:2b")
    assert c.REWRITE_MODEL == "b:2b"


def test_a_non_numeric_value_falls_back_and_is_remembered(monkeypatch):
    """It must not raise at import, and it must not be silent either -
    startup logs BAD_ENV so an ignored typo is visible."""
    c = _reload(monkeypatch, RME_TOP_K="six")
    assert c.TOP_K == 6
    assert ("RME_TOP_K", "six", 6) in c.BAD_ENV


def test_blank_values_are_treated_as_unset(monkeypatch):
    c = _reload(monkeypatch, RME_MODEL="   ")
    assert c.MODEL == "qwen3:14b"


@pytest.mark.parametrize("raw,expected",
                         [("1", True), ("true", True), ("yes", True),
                          ("0", False), ("false", False), ("", False)])
def test_content_logging_switch(monkeypatch, raw, expected):
    assert _reload(monkeypatch, RME_LOG_CONTENT=raw).LOG_CONTENT is expected


def test_the_answer_log_line_records_shape_not_content(monkeypatch, caplog):
    import logs
    _reload(monkeypatch)
    log = logging.getLogger("rme.test")
    log.propagate = True
    with caplog.at_level(logging.INFO, logger="rme.test"):
        logs.log_answer({"question": "how many leave days do I get",
                         "answer": "21 days", "lang": "en", "model": "m",
                         "hits": [1, 2, 3], "validator": {"verdict": "clean"},
                         "incomplete_workflows": [1], "variant_conflicts": [],
                         "total_s": 74.2}, log)
    line = caplog.text
    assert "hits=3" in line and "verdict=clean" in line and "incomplete_wf=1" in line
    assert "question_chars=28" in line
    assert "leave days" not in line          # the question itself stays out
    assert "21 days" not in line
