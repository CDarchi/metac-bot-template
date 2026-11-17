"""Offline tests for bot_config.py. They need no API keys and make no network calls."""
import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SETTING_NAMES = [
    "BOT_FORECAST_MODEL", "BOT_REASONING_EFFORT", "BOT_PREDICTIONS_PER_QUESTION",
    "BOT_RESEARCHER", "BOT_SUMMARIZER_MODEL", "BOT_PARSER_MODEL",
    "BOT_MAX_COST_PER_RUN_USD", "BOT_DRY_RUN",
    "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
]


@pytest.fixture()
def cfg(monkeypatch):
    for name in SETTING_NAMES:
        monkeypatch.delenv(name, raising=False)
    import bot_config
    return importlib.reload(bot_config)


def test_defaults_without_any_key_use_toolkit(cfg):
    assert cfg.forecast_model_name() is None
    assert cfg.build_llms() is None
    assert cfg.predictions_per_question() == 5
    assert cfg.max_cost_per_run_usd() == 5.0
    assert cfg.dry_run() is False


def test_budget_tier_is_default_with_openrouter_key(cfg, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    llms = cfg.build_llms()
    assert llms["default"].model == "openrouter/anthropic/claude-haiku-5.5"
    assert set(llms) == {"default"}  # other roles are filled in by the toolkit


def test_budget_tier_with_anthropic_key_only(cfg, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert cfg.forecast_model_name() == "anthropic/claude-haiku-5-5"


def test_empty_github_variables_count_as_unset(cfg, monkeypatch):
    # GitHub passes "" for a repository variable that does not exist.
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    for name in SETTING_NAMES[:8]:
        monkeypatch.setenv(name, "")
    assert cfg.forecast_model_name() == "openrouter/anthropic/claude-haiku-5.5"
    assert cfg.predictions_per_question() == 5
    assert cfg.reasoning_effort() is None
    assert cfg.dry_run() is False


def test_toolkit_default_can_be_requested(cfg, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setenv("BOT_FORECAST_MODEL", "toolkit-default")
    assert cfg.build_llms() is None


def test_standard_tier_settings_match_metaculus_shape(cfg, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setenv("BOT_FORECAST_MODEL", "openrouter/google/gemini-4-argon")
    monkeypatch.setenv("BOT_REASONING_EFFORT", "high")
    kwargs = cfg.forecast_llm_kwargs(cfg.forecast_model_name(), cfg.reasoning_effort())
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["temperature"] is None
    assert cfg.build_llms()["default"].model == "openrouter/google/gemini-4-argon"


def test_direct_claude_uses_adaptive_thinking(cfg):
    kwargs = cfg.forecast_llm_kwargs("anthropic/claude-haiku-5-5", "high")
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "high"}


def test_bad_values_fall_back_to_defaults(cfg, monkeypatch):
    monkeypatch.setenv("BOT_PREDICTIONS_PER_QUESTION", "many")
    monkeypatch.setenv("BOT_MAX_COST_PER_RUN_USD", "-3")
    monkeypatch.setenv("BOT_REASONING_EFFORT", "extreme")
    assert cfg.predictions_per_question() == 5
    assert cfg.max_cost_per_run_usd() == 5.0
    assert cfg.reasoning_effort() is None


def test_optional_roles_and_dry_run(cfg, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setenv("BOT_RESEARCHER", "asknews/news-summaries")
    monkeypatch.setenv("BOT_PREDICTIONS_PER_QUESTION", "3")
    monkeypatch.setenv("BOT_DRY_RUN", "true")
    llms = cfg.build_llms()
    assert llms["researcher"] == "asknews/news-summaries"
    assert cfg.predictions_per_question() == 3
    assert cfg.dry_run() is True
    assert "dry run=True" in cfg.describe_settings()
