"""
Settings for the forecasting bot that can be changed WITHOUT editing code.

Where to change them on GitHub:
    Settings -> Secrets and variables -> Actions -> "Variables" tab -> New repository variable

Variable name                   Default        Meaning
------------------------------  -------------  ---------------------------------------------
BOT_FORECAST_MODEL              (budget tier)  Model that makes the forecasts. See TIERS below.
BOT_REASONING_EFFORT            (none)         low / medium / high. Only for models that support it.
BOT_PREDICTIONS_PER_QUESTION    5              How many forecasts are averaged per question (1-9).
BOT_RESEARCHER                  (auto)         e.g. asknews/news-summaries
BOT_SUMMARIZER_MODEL            (auto)         Cheap model used only for comment summaries.
BOT_PARSER_MODEL                (auto)         Cheap model that turns text into numbers.
BOT_MAX_COST_PER_RUN_USD        5              Soft spending cap for one run. 0 = no cap.
BOT_DRY_RUN                     false          true = forecast but do not publish to Metaculus.

Why the default is a cheap model:
    The upstream template defaults to a top-tier model. Metaculus' own estimate for
    that model at high thinking effort is about $2.40 per question, so a $100 credit
    grant could be used up quickly. This file defaults to a budget tier instead, and
    lets the owner step up one variable at a time once more credits are granted.

TIERS (model IDs and per-question cost estimates are taken from Metaculus' own bot
configuration file, forecasting-tools/run_bots.py, as of November 2026; the estimates
assume 5 forecasts per question at high thinking effort, so they are upper-end
figures for the plain settings used here and must be checked against real usage):
    budget    openrouter/anthropic/claude-haiku-5.5            about $0.06 / question
    standard  openrouter/google/gemini-4-argon  (effort=high)  about $0.38 / question
    top       toolkit-default (currently Claude Opus 5.5)      up to about $2.40 / question
"""

from __future__ import annotations

import logging
import os
from typing import Any

from forecasting_tools import GeneralLlm

logger = logging.getLogger(__name__)

TOOLKIT_DEFAULT = "toolkit-default"
BUDGET_MODEL_OPENROUTER = "openrouter/anthropic/claude-haiku-5.5"
BUDGET_MODEL_ANTHROPIC = "anthropic/claude-haiku-5-5"
VALID_EFFORTS = ("low", "medium", "high")


def _get(name: str, default: str = "") -> str:
    """Read a setting. Empty or missing values fall back to the default."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _get_number(name: str, default: float, low: float, high: float) -> float:
    raw = _get(name)
    if not raw:
        return default
    try:
        number = float(raw)
    except ValueError:
        logger.warning(f"{name}={raw!r} is not a number. Using default {default}.")
        return default
    if number < low or number > high:
        logger.warning(
            f"{name}={number} is outside {low}..{high}. Using default {default}."
        )
        return default
    return number


def predictions_per_question() -> int:
    return int(_get_number("BOT_PREDICTIONS_PER_QUESTION", 5, 1, 9))


def max_cost_per_run_usd() -> float:
    """Soft cap in USD for one run. 0 means no cap."""
    return _get_number("BOT_MAX_COST_PER_RUN_USD", 5.0, 0.0, 1000.0)


def dry_run() -> bool:
    return _get("BOT_DRY_RUN").lower() in ("1", "true", "yes", "on")


def reasoning_effort() -> str | None:
    effort = _get("BOT_REASONING_EFFORT").lower()
    if not effort or effort == "none":
        return None
    if effort not in VALID_EFFORTS:
        logger.warning(
            f"BOT_REASONING_EFFORT={effort!r} is not one of {VALID_EFFORTS}. Ignoring it."
        )
        return None
    return effort


def forecast_model_name() -> str | None:
    """
    Returns the model to forecast with, or None to let the toolkit choose
    (the upstream template behaviour).
    """
    requested = _get("BOT_FORECAST_MODEL")
    if requested.lower() == TOOLKIT_DEFAULT:
        return None
    if requested:
        return requested
    # Nothing requested: use the budget tier that matches the available key.
    if _get("OPENROUTER_API_KEY"):
        return BUDGET_MODEL_OPENROUTER
    if _get("ANTHROPIC_API_KEY"):
        return BUDGET_MODEL_ANTHROPIC
    return None


def forecast_llm_kwargs(model: str, effort: str | None) -> dict[str, Any]:
    """
    Model settings. The shapes below copy what Metaculus uses for its own bots:
    - Claude called directly: adaptive thinking with an effort level
    - other reasoning models: reasoning_effort, and no temperature
    - no effort requested: a plain call with temperature 0.3
    """
    if effort is None:
        return {"temperature": 0.3, "timeout": 160, "allowed_tries": 2}
    if model.startswith("anthropic/"):
        return {
            "temperature": 1,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
            "allowed_openai_params": ["output_config"],
            "max_tokens": 64000,
            "timeout": 160,
            "allowed_tries": 2,
        }
    return {
        "reasoning_effort": effort,
        "temperature": None,
        "timeout": 300,
        "allowed_tries": 2,
    }


def build_llms() -> dict[str, Any] | None:
    """
    Returns the `llms` argument for the bot, or None to use toolkit defaults.
    Roles that are not set here are filled in by the toolkit automatically.
    """
    model = forecast_model_name()
    if model is None:
        return None
    llms: dict[str, Any] = {
        "default": GeneralLlm(model=model, **forecast_llm_kwargs(model, reasoning_effort()))
    }
    for role, variable in (
        ("researcher", "BOT_RESEARCHER"),
        ("summarizer", "BOT_SUMMARIZER_MODEL"),
        ("parser", "BOT_PARSER_MODEL"),
    ):
        value = _get(variable)
        if value:
            llms[role] = value
    return llms


def describe_settings() -> str:
    model = forecast_model_name() or "toolkit default"
    cap = max_cost_per_run_usd()
    return (
        f"forecast model={model}, effort={reasoning_effort() or 'none'}, "
        f"forecasts per question={predictions_per_question()}, "
        f"cost cap per run={'none' if cap == 0 else f'${cap:g}'}, "
        f"dry run={dry_run()}"
    )
