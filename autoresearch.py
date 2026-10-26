from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from datetime import datetime, timezone
from time import monotonic
from typing import Protocol

from budget import Budget, BudgetExceeded
from contracts import (
    ExperimentResult,
    ExperimentStatus,
    Hypothesis,
    ResearchOutcome,
    TaskType,
    is_better,
    metric_for_task,
)
from experiment_log import ExperimentLog


class ExperimentEvaluator(Protocol):
    async def __call__(
        self,
        hypothesis: Hypothesis,
        task_type: TaskType,
        timeout_seconds: float,
    ) -> float: ...


class HypothesisProvider(Protocol):
    """Each proposal is charged its maximum possible cost before the request."""

    def estimate_max_cost_rub(
        self,
        results: Sequence[ExperimentResult],
    ) -> float: ...

    async def __call__(
        self,
        results: Sequence[ExperimentResult],
        timeout_seconds: float,
    ) -> Hypothesis | None: ...


async def run_autoresearch(
    *,
    task_type: TaskType,
    initial_hypothesis: Hypothesis,
    evaluator: ExperimentEvaluator,
    budget: Budget,
    experiment_log: ExperimentLog,
    hypothesis_provider: HypothesisProvider | None = None,
    max_experiments: int = 20,
) -> ResearchOutcome:
    """Evaluate a seed solution and refine it through an optional provider.

    The seed can be a baseline or a manually supplied candidate. Hypothesis
    generation is deliberately injected so this loop works before the LLM
    proposal step is available.
    """
    if max_experiments <= 0:
        raise ValueError("max_experiments must be positive")

    metric = metric_for_task(task_type)
    results: list[ExperimentResult] = []
    seen_ids: set[str] = set()
    hypothesis: Hypothesis | None = initial_hypothesis
    best: ExperimentResult | None = None
    stop_reason: str | None = None

    while hypothesis is not None and len(results) < max_experiments:
        if hypothesis.id in seen_ids:
            raise ValueError(f"Hypothesis provider returned duplicate id: {hypothesis.id}")
        seen_ids.add(hypothesis.id)

        try:
            timeout_seconds = budget.require_research_time()
        except BudgetExceeded:
            stop_reason = "Research time limit reached"
            break

        started_clock = monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            metric_value = await asyncio.wait_for(
                evaluator(hypothesis, task_type, timeout_seconds),
                timeout=timeout_seconds,
            )
            if not math.isfinite(metric_value):
                raise ValueError("Evaluator returned a non-finite metric")
            result = ExperimentResult(
                hypothesis=hypothesis,
                task_type=task_type,
                metric=metric,
                status=ExperimentStatus.SUCCEEDED,
                metric_value=float(metric_value),
                elapsed_seconds=max(0.0, monotonic() - started_clock),
                started_at=started_at,
            )
        except Exception as error:
            result = ExperimentResult(
                hypothesis=hypothesis,
                task_type=task_type,
                metric=metric,
                status=ExperimentStatus.FAILED,
                metric_value=None,
                elapsed_seconds=max(0.0, monotonic() - started_clock),
                started_at=started_at,
                error=f"{type(error).__name__}: {error}",
            )

        experiment_log.append(result)
        results.append(result)
        if is_better(result, best):
            best = result

        if (
            hypothesis_provider is None
            or len(results) >= max_experiments
        ):
            break
        if budget.remaining_api_budget_rub <= 0:
            stop_reason = "API budget exhausted"
            break

        try:
            timeout_seconds = budget.require_research_time()
        except BudgetExceeded:
            stop_reason = "Research time limit reached"
            break
        try:
            estimated_cost_rub = hypothesis_provider.estimate_max_cost_rub(tuple(results))
            budget.charge_api_estimate(estimated_cost_rub)
        except BudgetExceeded:
            stop_reason = "Estimated API cost exceeds the remaining budget"
            break
        except Exception as error:
            stop_reason = f"API cost estimate failed: {type(error).__name__}: {error}"
            break
        try:
            hypothesis = await asyncio.wait_for(
                hypothesis_provider(tuple(results), timeout_seconds),
                timeout=timeout_seconds,
            )
        except Exception as error:
            stop_reason = f"Hypothesis proposal failed: {type(error).__name__}: {error}"
            break
        if hypothesis is None:
            stop_reason = "Hypothesis provider returned no further proposal"

    return ResearchOutcome(
        experiments=tuple(results),
        best=best,
        stop_reason=stop_reason,
    )
