from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autoresearch import run_autoresearch
from budget import Budget, BudgetExceeded
from contracts import (
    EvaluationReport,
    ExperimentStatus,
    Hypothesis,
    MetricName,
    SandboxOutput,
    TaskType,
)
from experiment_log import ExperimentLog
from validation import score_predictions


class _Agent:
    def __init__(self, provider):
        self._provider = provider

    async def propose(self, previous_cycle_output, history, timeout_seconds):
        return await self._provider._propose(
            previous_cycle_output, history, timeout_seconds
        )


class _Provider:
    def __init__(self, propose, estimated_cost_rub: float = 1.0):
        self._propose = propose
        self._estimated_cost_rub = estimated_cost_rub
        self.created_agents = 0

    def estimate_max_cost_rub(self, results):
        return self._estimated_cost_rub

    def create_agent(self):
        self.created_agents += 1
        return _Agent(self)


class ValidationTests(unittest.TestCase):
    def test_rmse_is_root_mean_squared_error(self) -> None:
        score = score_predictions(TaskType.REGRESSION, [1, 2], [2, 4])
        self.assertAlmostEqual(score, 2.5**0.5)

    def test_multiclass_classification_uses_macro_f1(self) -> None:
        score = score_predictions(
            TaskType.MULTICLASS_CLASSIFICATION,
            ["a", "a", "b", "c"],
            ["a", "b", "b", "b"],
        )
        self.assertAlmostEqual(score, 7 / 18)

    def test_validation_rejects_different_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            score_predictions(TaskType.REGRESSION, [1], [1, 2])


class BudgetTests(unittest.TestCase):
    def test_research_deadline_preserves_finalization_reserve(self) -> None:
        now = [100.0]
        budget = Budget(
            time_limit_seconds=100,
            api_budget_rub=600,
            finalization_reserve_seconds=10,
            clock=lambda: now[0],
        )
        now[0] += 85
        self.assertEqual(budget.remaining_research_seconds, 5)

        now[0] += 5
        self.assertEqual(budget.remaining_research_seconds, 0)
        with self.assertRaises(BudgetExceeded):
            budget.require_research_time()

    def test_api_budget_rejects_an_over_budget_estimate(self) -> None:
        budget = Budget(time_limit_seconds=100, finalization_reserve_seconds=5)
        with self.assertRaises(BudgetExceeded):
            budget.ensure_can_spend_api(600.01)


class AutoResearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_metric_direction_and_logs_each_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log = ExperimentLog(Path(temp_dir) / "nested" / "experiments.jsonl")
            hypotheses = [
                Hypothesis("seed", "baseline"),
                Hypothesis("candidate-1", "first change"),
                Hypothesis("candidate-2", "second change"),
            ]
            values = iter([0.6, 0.8, 0.7])
            previous_cycle_outputs = []

            async def evaluate(hypothesis, task_type, timeout_seconds):
                return EvaluationReport(
                    metric_value=next(values),
                    sandbox_output=SandboxOutput(
                        status="success",
                        stdout=f"stdout from {hypothesis.id}",
                        output_files=("validation_predictions.csv",),
                    ),
                )

            async def propose(previous_cycle_output, results, timeout_seconds):
                previous_cycle_outputs.append(previous_cycle_output)
                return hypotheses[len(results)] if len(results) < len(hypotheses) else None

            provider = _Provider(propose)
            outcome = await run_autoresearch(
                task_type=TaskType.MULTICLASS_CLASSIFICATION,
                initial_hypothesis=hypotheses[0],
                evaluator=evaluate,
                hypothesis_agent_factory=provider,
                budget=Budget(time_limit_seconds=100, finalization_reserve_seconds=5),
                experiment_log=log,
                max_experiments=3,
            )

            self.assertEqual(len(outcome.experiments), 3)
            self.assertEqual(outcome.best.hypothesis.id, "candidate-1")
            self.assertEqual(outcome.best.metric, MetricName.MACRO_F1)
            self.assertEqual(len(log.read()), 3)
            self.assertEqual(provider.created_agents, 2)
            self.assertEqual(len(previous_cycle_outputs), 2)
            self.assertIn('"metric_value": 0.6', previous_cycle_outputs[0])
            logged_output = log.read()[0].sandbox_output
            self.assertEqual(logged_output.stdout, "stdout from seed")
            self.assertEqual(
                logged_output.output_files,
                ("validation_predictions.csv",),
            )

    async def test_regression_selects_lowest_rmse(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            hypotheses = [
                Hypothesis("seed", "baseline"),
                Hypothesis("candidate", "improved"),
            ]
            values = iter([1.5, 1.2])

            async def evaluate(hypothesis, task_type, timeout_seconds):
                return next(values)

            async def propose(previous_cycle_output, results, timeout_seconds):
                return hypotheses[1] if len(results) == 1 else None

            provider = _Provider(propose)
            outcome = await run_autoresearch(
                task_type=TaskType.REGRESSION,
                initial_hypothesis=hypotheses[0],
                evaluator=evaluate,
                hypothesis_agent_factory=provider,
                budget=Budget(time_limit_seconds=100, finalization_reserve_seconds=5),
                experiment_log=ExperimentLog(Path(temp_dir) / "experiments.jsonl"),
                max_experiments=2,
            )

            self.assertEqual(outcome.best.hypothesis.id, "candidate")
            self.assertEqual(outcome.best.metric, MetricName.RMSE)

    async def test_failed_experiment_does_not_become_best(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log = ExperimentLog(Path(temp_dir) / "experiments.jsonl")

            async def evaluate(hypothesis, task_type, timeout_seconds):
                if hypothesis.id == "bad":
                    raise RuntimeError("training failed")
                return 1.25

            outcome = await run_autoresearch(
                task_type=TaskType.REGRESSION,
                initial_hypothesis=Hypothesis("bad", "broken candidate"),
                evaluator=evaluate,
                budget=Budget(time_limit_seconds=100, finalization_reserve_seconds=5),
                experiment_log=log,
            )

            self.assertIsNone(outcome.best)
            self.assertEqual(outcome.experiments[0].status, ExperimentStatus.FAILED)
            self.assertIn("training failed", outcome.experiments[0].error)

    async def test_research_stops_before_consuming_finalization_reserve(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            now = [0.0]
            budget = Budget(
                time_limit_seconds=100,
                finalization_reserve_seconds=10,
                clock=lambda: now[0],
            )

            async def evaluate(hypothesis, task_type, timeout_seconds):
                now[0] = 95.0
                return 0.5

            async def propose(previous_cycle_output, results, timeout_seconds):
                self.fail("must preserve time for finalization")

            outcome = await run_autoresearch(
                task_type=TaskType.REGRESSION,
                initial_hypothesis=Hypothesis("seed", "baseline"),
                evaluator=evaluate,
                hypothesis_agent_factory=_Provider(propose),
                budget=budget,
                experiment_log=ExperimentLog(Path(temp_dir) / "experiments.jsonl"),
            )

            self.assertEqual(len(outcome.experiments), 1)
            self.assertEqual(outcome.best.hypothesis.id, "seed")

    async def test_budget_blocks_a_proposal_above_remaining_api_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            budget = Budget(
                time_limit_seconds=100,
                api_budget_rub=10,
                finalization_reserve_seconds=5,
            )

            async def evaluate(hypothesis, task_type, timeout_seconds):
                return 1.0

            async def propose(previous_cycle_output, results, timeout_seconds):
                self.fail("must not make an over-budget model request")

            outcome = await run_autoresearch(
                task_type=TaskType.REGRESSION,
                initial_hypothesis=Hypothesis("seed", "baseline"),
                evaluator=evaluate,
                hypothesis_agent_factory=_Provider(propose, estimated_cost_rub=10.01),
                budget=budget,
                experiment_log=ExperimentLog(Path(temp_dir) / "experiments.jsonl"),
            )

            self.assertEqual(len(outcome.experiments), 1)
            self.assertEqual(budget.api_charged_rub, 0)

    async def test_proposal_error_preserves_best_completed_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            budget = Budget(
                time_limit_seconds=100,
                api_budget_rub=10,
                finalization_reserve_seconds=5,
            )

            async def evaluate(hypothesis, task_type, timeout_seconds):
                return 0.75

            async def propose(previous_cycle_output, results, timeout_seconds):
                raise RuntimeError("provider unavailable")

            outcome = await run_autoresearch(
                task_type=TaskType.MULTICLASS_CLASSIFICATION,
                initial_hypothesis=Hypothesis("seed", "baseline"),
                evaluator=evaluate,
                hypothesis_agent_factory=_Provider(propose, estimated_cost_rub=2),
                budget=budget,
                experiment_log=ExperimentLog(Path(temp_dir) / "experiments.jsonl"),
            )

            self.assertEqual(outcome.best.hypothesis.id, "seed")
            self.assertIn("provider unavailable", outcome.stop_reason)
            self.assertEqual(budget.api_charged_rub, 2)


if __name__ == "__main__":
    unittest.main()
