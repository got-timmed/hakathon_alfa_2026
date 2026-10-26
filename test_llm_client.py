from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from contracts import (
    ExperimentResult,
    ExperimentStatus,
    Hypothesis,
    MetricName,
    TaskType,
)
from llm_client import QwenHypothesisAgentFactory, QwenSettings


class _FakeCompletions:
    def __init__(self, content: str):
        self.content = content
        self.request = None

    async def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=self.content))
            ]
        )


class _FakeClient:
    def __init__(self, content: str):
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(content),
        )
        self.closed = False

    async def close(self):
        self.closed = True


class QwenHypothesisAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_fresh_agent_uses_previous_output_and_parses_code(self) -> None:
        clients = []

        def create_client(**kwargs):
            client = _FakeClient(
                json.dumps(
                    {
                        "description": "Try regularization",
                        "config": {"alpha": 0.1},
                        "code": "print('candidate')",
                    }
                )
            )
            clients.append((kwargs, client))
            return client

        settings = QwenSettings(
            model="Qwen3.8-27B",
            request_reserve_rub=0.022,
            max_input_tokens=10_000,
            max_output_tokens=500,
        )
        factory = QwenHypothesisAgentFactory(
            settings,
            client_factory=create_client,
            prompt_template=(
                "task={{TASK_TYPE}} metric={{METRIC}} direction={{DIRECTION}}\n"
                "history={{HISTORY}}\nprevious={{PREVIOUS_CYCLE_OUTPUT}}"
            ),
        )
        previous_result = ExperimentResult(
            hypothesis=Hypothesis(
                "seed",
                "baseline",
                code="print('seed')",
            ),
            task_type=TaskType.REGRESSION,
            metric=MetricName.RMSE,
            status=ExperimentStatus.FAILED,
            metric_value=None,
            elapsed_seconds=0.2,
            started_at="2026-10-26T10:00:00+00:00",
            error="RuntimeError: training failed",
        )
        output = '{"status":"failed","stderr":"missing feature"}'

        first_agent = factory.create_agent()
        second_agent = factory.create_agent()
        self.assertIsNot(first_agent, second_agent)
        proposal = await first_agent.propose(output, [previous_result], 10)

        self.assertEqual(proposal.description, "Try regularization")
        self.assertEqual(proposal.config, {"alpha": 0.1})
        self.assertEqual(proposal.code, "print('candidate')")
        kwargs, client = clients[0]
        self.assertEqual(kwargs, {})
        request = client.chat.completions.request
        self.assertEqual(request["model"], "Qwen3.8-27B")
        self.assertEqual(request["max_tokens"], 500)
        self.assertIn(output, request["messages"][1]["content"])
        self.assertIn("minimize", request["messages"][1]["content"])
        self.assertTrue(client.closed)

    async def test_factory_reserves_worst_case_cost(self) -> None:
        settings = QwenSettings(
            model="Qwen3.8-27B",
            request_reserve_rub=0.022,
            max_input_tokens=10_000,
            max_output_tokens=500,
        )
        factory = QwenHypothesisAgentFactory(
            settings,
            client_factory=lambda **kwargs: _FakeClient("{}"),
            prompt_template="test",
        )
        self.assertAlmostEqual(factory.estimate_max_cost_rub([]), 0.022)

    def test_settings_use_platform_environment_and_default_model(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_BASE_URL": "https://proxy.example/v1",
            },
            clear=True,
        ):
            settings = QwenSettings.from_env()

        self.assertEqual(settings.model, "qwen/qwen3.8-27b")
        self.assertEqual(settings.request_reserve_rub, 30)


if __name__ == "__main__":
    unittest.main()
