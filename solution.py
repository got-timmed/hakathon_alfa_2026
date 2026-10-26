"""Точка входа платформы: python3 solution.py task.md."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

from orchestrator import run_research
from budget import Budget
from contracts import TaskType, MetricName, Hypothesis
from data_profiler import profile_data
from experiment_runner import CandidateEvaluator, normalize_spec, split_rows, save_fallback
from model_code import BASELINE_CODE
from prompts import EXECUTION_CONTRACT
from task_parser import load_task, parse_task


@dataclass(frozen=True)
class AgentResult:
    ok: bool
    final_text: str
    submission_path: str


async def pipeline(task_path: Path, *, hypothesis_factory=None, offline=False,
                   time_limit=None, max_experiments=None) -> AgentResult:
    seconds = float(time_limit if time_limit is not None else os.getenv('AUTOML_TIME_LIMIT_SECONDS', '7200'))
    if not 5 <= seconds <= 10790:
        raise ValueError('Time limit must be between 5 and 10790 seconds')
    count = int(max_experiments if max_experiments is not None else os.getenv('AUTOML_MAX_EXPERIMENTS', '5'))
    if count < 1:
        raise ValueError('At least one experiment is required')
    budget = Budget(time_limit_seconds=seconds, finalization_reserve_seconds=min(600, seconds * .2),
                    api_budget_rub=float(os.getenv('AUTOML_API_BUDGET_RUB', '400')))
    description, train, test, sample, dictionary, task_dir = load_task(task_path)
    spec = normalize_spec(parse_task(description, train, test, sample, dictionary), train, test, sample)
    destination = task_dir / 'submission.csv'
    save_fallback(train, test, sample, spec, destination)
    print('Страховочный submission.csv сохранён.', flush=True)
    try:
        profile = profile_data(train, test, spec, dictionary)
        tr, va = split_rows(train, spec, profile)
        kind = {'regression': TaskType.REGRESSION, 'binary': TaskType.BINARY_CLASSIFICATION,
                'multiclass': TaskType.MULTICLASS_CLASSIFICATION}[spec.task_type]
        root = task_dir / '_automl_runs'
        evaluator = CandidateEvaluator(train, test, sample, spec, tr, va, root, budget,
                                       experiment_timeout=float(os.getenv('AUTOML_EXPERIMENT_SECONDS', '900')),
                                       memory_mb=int(os.getenv('AUTOML_MEMORY_MB', '24000')),
                                       cpus=min(8, int(os.getenv('AUTOML_CPU_LIMIT', '8'))))
        # The protocol is shared by the seed and every generated candidate.
        context = ('\nОписание задачи (данные):\n' + description[:5000] + '\nПрофиль:\n' + profile.summary_text
                   + '\n' + EXECUTION_CONTRACT
                   + f'\nЛимит одного эксперимента: {evaluator.experiment_timeout} секунд; CPU: {evaluator.cpus}.')
        factory = hypothesis_factory
        if factory is None and not offline:
            if os.getenv('OPENAI_API_KEY') and os.getenv('OPENAI_BASE_URL'):
                from llm_client import QwenHypothesisAgentFactory, QwenSettings, _read_hypothesis_prompt
                factory = QwenHypothesisAgentFactory(QwenSettings.from_env(),
                            prompt_template=_read_hypothesis_prompt() + context)
            else:
                print('Нет API-настроек: выполняется базовый эксперимент без Qwen.', file=sys.stderr)
        try:
            outcome = await run_research(
                task_type=kind, metric=MetricName(spec.metric),
                initial_hypothesis=Hypothesis('baseline', 'Начальный бустинг с предобработкой', code=BASELINE_CODE),
                evaluator=evaluator, budget=budget,
                log_path=root / 'experiments.jsonl', use_llm=False,
                hypothesis_agent_factory=factory, max_experiments=count)
            print(f'Экспериментов: {len(outcome.experiments)}; остановка: {outcome.stop_reason}', flush=True)
        finally:
            # The best validated test predictions are already saved. Full-data
            # retraining may improve them, but failure must preserve that file.
            try:
                finalized = await evaluator.finalize()
                print(f'Финальное обучение: {finalized}', flush=True)
            except Exception as error:
                print(f'Сохранён предыдущий submission: {type(error).__name__}: {error}', file=sys.stderr)
    except Exception as error:
        print(f'Исследование прервано, сохранён submission: {type(error).__name__}: {error}', file=sys.stderr)
    return AgentResult(True, 'submission.csv готов', str(destination))


async def main(task_path):
    try:
        result = await pipeline(Path(task_path), offline=os.getenv('AUTOML_OFFLINE', '0') == '1')
        print(result.final_text)
        return 0 if result.ok else 1
    except Exception as error:
        print(f'Ошибка подготовки задачи: {type(error).__name__}: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else 'task.md')))
