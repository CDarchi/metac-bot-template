"""
Offline end-to-end check: runs one binary question through the bot with the LLM
mocked, publishing off, and the spending cap active. No keys, no network.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_binary_question_flows_through_bot(monkeypatch):
    monkeypatch.setenv("METACULUS_TOKEN", "dummy-token-for-offline-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-key-for-offline-test")
    monkeypatch.setenv("BOT_PREDICTIONS_PER_QUESTION", "3")
    monkeypatch.setenv("BOT_MAX_COST_PER_RUN_USD", "1")

    import main
    from bot_config import build_llms, max_cost_per_run_usd, predictions_per_question
    from forecasting_tools import (
        BinaryPrediction, BinaryQuestion, GeneralLlm, MonetaryCostManager,
    )

    calls = {"llm": 0}

    async def fake_invoke(self, prompt, *args, **kwargs):
        calls["llm"] += 1
        return "Base rate is low; status quo persists.\n\nProbability: 23%"

    async def fake_research(self, question):
        return "No relevant news found (offline test)."

    async def fake_structure_output(*args, **kwargs):
        return BinaryPrediction(prediction_in_decimal=0.23)

    monkeypatch.setattr(GeneralLlm, "invoke", fake_invoke)
    monkeypatch.setattr(main.FallTemplateBot2026, "run_research", fake_research)
    monkeypatch.setattr(main, "structure_output", fake_structure_output)

    bot = main.FallTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=predictions_per_question(),
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=False,
        extra_metadata_in_explanation=True,
        llms=build_llms(),
    )
    assert bot.get_llm("default", "llm").model == "openrouter/anthropic/claude-haiku-5.5"

    question = BinaryQuestion(
        question_text="Will the offline test question resolve Yes?",
        id_of_post=1,
        id_of_question=1,
        page_url="https://www.metaculus.com/questions/1/",
        background_info="Synthetic question used only for an offline test.",
        resolution_criteria="Resolves Yes if the test passes.",
        fine_print="",
    )

    with MonetaryCostManager(max_cost_per_run_usd()) as cost_manager:
        report = asyncio.run(bot.forecast_question(question))

    assert abs(report.prediction - 0.23) < 1e-9
    assert calls["llm"] >= 3  # one reasoning call per forecast
    assert cost_manager.current_usage == 0  # mocked calls cost nothing
