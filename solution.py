"""Точка входа. Запуск: python solution.py task.md
Переменные окружения: OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL
"""

import asyncio
import os
import sys
from pathlib import Path

# Предполагаемые модули (реализуются отдельно)
from preprocessing import PreparedTask, prepare
from agent.sandbox import ExecutionResult, run_experiment
from llm.prompts import build_first_iteration_prompt, extract_python_code

TOTAL_TIMEOUT_SEC = 10790


def read_task(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_config() -> AgentConfig:
    """Читает переменные окружения, заданные платформой."""
    return AgentConfig(
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.environ["OPENAI_MODEL"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
        max_steps=100,
    )


async def pipeline(task_path: Path) -> AgentResult:
    task_path = task_path.resolve()
    task_dir = task_path.parent

    #Чтение задачи
    task_text = await asyncio.to_thread(read_task, task_path)

    # Препроцессинг
    prepared: PreparedTask = await asyncio.to_thread(prepare, task_text, task_dir)
    for warning in prepared.warnings:
        print(f"[preprocessing] {warning}", file=sys.stderr)

    first_prompt_answer = await build_first_iteration_prompt(
        df_train=prepared.df_train,
        target_col=prepared.target_col,
        task_type=prepared.task_type,
        target_metric=prepared.target_metric,
        val_ids=prepared.val_ids,
        data_dictionary=prepared.data_dictionary,
    )
    code = extract_python_code(first_prompt_answer)
    result: ExecutionResult = await asyncio.to_thread(run_experiment, code, prepared.root, prepared.scratch, prepared.env,
        prepared.cpu_limit, prepared.memory_limit_mb, prepared.deadline, prepared.max_output_bytes, prepared.outputs)

async def main(task_path: Path) -> int:
    try:
        result = await asyncio.wait_for(pipeline(task_path), timeout=TOTAL_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        print("Превышен общий лимит времени", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Ошибка пайплайна: {error!r}", file=sys.stderr)
        return 1

    print(result.final_text or "Агент завершил работу.")
    return 0 if result.ok else 1


if __name__ == "__main__":
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "task.md")
    sys.exit(asyncio.run(main(path)))