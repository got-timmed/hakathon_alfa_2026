from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from time import monotonic
from typing import Protocol

from budget import Budget, BudgetExceeded
from contracts import (
    ExperimentResult,
    ExperimentStatus,
    EvaluationReport,
    Hypothesis,
    ResearchOutcome,
    SandboxOutput,
    TaskType,
    is_better,
    metric_for_task,
    MetricName,
)
from experiment_log import ExperimentLog


class ExperimentEvaluator(Protocol):
    async def __call__(
        self,
        hypothesis: Hypothesis,
        task_type: TaskType,
        timeout_seconds: float,
    ) -> float | EvaluationReport: ...


class HypothesisAgent(Protocol):
    async def propose(
        self,
        previous_cycle_output: str,
        history: Sequence[ExperimentResult],
        timeout_seconds: float,
    ) -> Hypothesis | None: ...


class HypothesisAgentFactory(Protocol):
    """Create a new, stateless LLM agent for each hypothesis proposal."""

    def estimate_max_cost_rub(
        self,
        results: Sequence[ExperimentResult],
    ) -> float: ...

    def create_agent(self) -> HypothesisAgent: ...


def format_previous_cycle_output(result: ExperimentResult) -> str:
    """Serialize the last run, including sandbox diagnostics, for the next prompt."""
    payload = {
        "hypothesis": {
            "id": result.hypothesis.id,
            "description": result.hypothesis.description,
            "config": dict(result.hypothesis.config),
            "code": result.hypothesis.code,
        },
        "task_type": result.task_type.value,
        "metric": result.metric.value,
        "metric_value": result.metric_value,
        "status": result.status.value,
        "error": result.error,
        "elapsed_seconds": result.elapsed_seconds,
        "sandbox_output": (
            asdict(result.sandbox_output) if result.sandbox_output is not None else None
        ),
    }
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)


async def run_autoresearch(
    *,
    task_type: TaskType,
    initial_hypothesis: Hypothesis,
    evaluator: ExperimentEvaluator,
    budget: Budget,
    experiment_log: ExperimentLog,
    hypothesis_agent_factory: HypothesisAgentFactory | None = None,
    max_experiments: int = 20,
    metric: MetricName | None = None,
) -> ResearchOutcome:
    """Evaluate a seed solution and refine it through an optional provider.

    The seed can be a baseline or a manually supplied candidate. Hypothesis
    generation is deliberately injected so this loop works before the LLM
    proposal step is available.
    """
    if max_experiments <= 0:
        raise ValueError("max_experiments must be positive")

    metric = metric or metric_for_task(task_type)
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
        sandbox_output: SandboxOutput | None = None
        try:
            evaluation = await asyncio.wait_for(
                evaluator(hypothesis, task_type, timeout_seconds),
                timeout=timeout_seconds,
            )
            evaluation_error: str | None = None
            if isinstance(evaluation, EvaluationReport):
                metric_value = evaluation.metric_value
                sandbox_output = evaluation.sandbox_output
                evaluation_error = evaluation.error
                if sandbox_output is not None:
                    if sandbox_output.status != "success":
                        evaluation_error = evaluation_error or (
                            f"Sandbox finished with status {sandbox_output.status}"
                        )
                    elif sandbox_output.missing_outputs:
                        evaluation_error = evaluation_error or (
                            "Sandbox did not produce expected outputs: "
                            + ", ".join(sandbox_output.missing_outputs)
                        )
            else:
                metric_value = evaluation
            if evaluation_error is not None:
                raise RuntimeError(evaluation_error)
            if metric_value is None:
                raise ValueError("Evaluator returned no metric value")
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
                sandbox_output=sandbox_output,
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
                sandbox_output=sandbox_output,
            )

        experiment_log.append(result)
        results.append(result)
        if is_better(result, best):
            best = result

        if (
            hypothesis_agent_factory is None
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
            estimated_cost_rub = hypothesis_agent_factory.estimate_max_cost_rub(
                tuple(results)
            )
            budget.charge_api_estimate(estimated_cost_rub)
        except BudgetExceeded:
            stop_reason = "Estimated API cost exceeds the remaining budget"
            break
        except Exception as error:
            stop_reason = f"API cost estimate failed: {type(error).__name__}: {error}"
            break
        try:
            agent = hypothesis_agent_factory.create_agent()
            previous_cycle_output = format_previous_cycle_output(result)
            hypothesis = await asyncio.wait_for(
                agent.propose(
                    previous_cycle_output,
                    tuple(results),
                    timeout_seconds,
                ),
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
