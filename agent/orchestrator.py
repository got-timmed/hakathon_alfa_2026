from __future__ import annotations

from pathlib import Path

from agent.autoresearch import (
    ExperimentEvaluator,
    HypothesisProvider,
    run_autoresearch,
)
from agent.budget import Budget
from agent.contracts import Hypothesis, ResearchOutcome, TaskType
from agent.experiment_log import ExperimentLog
from config import (
    DAILY_API_BUDGET_RUB,
    FINALIZATION_RESERVE_SECONDS,
    MAX_EXPERIMENTS,
    TOTAL_TIME_LIMIT_SECONDS,
)


async def run_research(
    *,
    task_type: TaskType,
    initial_hypothesis: Hypothesis,
    evaluator: ExperimentEvaluator,
    log_path: str | Path,
    hypothesis_provider: HypothesisProvider | None = None,
    budget: Budget | None = None,
    max_experiments: int = MAX_EXPERIMENTS,
) -> ResearchOutcome:
    active_budget = budget or Budget(
        time_limit_seconds=TOTAL_TIME_LIMIT_SECONDS,
        api_budget_rub=DAILY_API_BUDGET_RUB,
        finalization_reserve_seconds=FINALIZATION_RESERVE_SECONDS,
    )
    return await run_autoresearch(
        task_type=task_type,
        initial_hypothesis=initial_hypothesis,
        evaluator=evaluator,
        budget=active_budget,
        experiment_log=ExperimentLog(log_path),
        hypothesis_provider=hypothesis_provider,
        max_experiments=max_experiments,
    )
