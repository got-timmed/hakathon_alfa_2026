"""Точка входа. Запуск: python solution.py task.md
Переменные окружения: OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL
"""

import asyncio
import sys
from pathlib import Path

from contracts import TaskSpec
from sandbox import ExecutionResult, run_experiment
from llm.prompts import build_first_iteration_prompt, extract_python_code
from llm.qwen_client import call_qwen
from task_parser import load_task, parse_task

TOTAL_TIMEOUT_SEC = 10790



async def pipeline(task_path: Path) -> AgentResult:
    task_path = task_path.resolve()

    # Препроцессинг
    description, train, test, sample_submission, data_dictionary, task_dir = load_task(task_path)
    prepared: TaskSpec = parse_task(description, train, test, sample_submission, data_dictionary)

    for warning in prepared.warnings:
        print(f"[preprocessing] {warning}", file=sys.stderr)

    first_prompt = build_first_iteration_prompt(
        df_train=train,
        target_col=prepared.target_col,
        task_type=prepared.task_type,
        target_metric=prepared.metric,
        val_ids=prepared.val_ids, #не понял что это, вопрос Владу
        data_dictionary=data_dictionary,
    ) #первый запрос к LLM, который возвращает код для эксперимента
    first_prompt_answer = await asyncio.to_thread(call_qwen, first_prompt) #запрос к LLM
    code = extract_python_code(first_prompt_answer) #извлечение кода из ответа LLM
    #Запуск эксперимента в песочнице
    result: ExecutionResult = await asyncio.to_thread(run_experiment, code, ...) #не доделано
    
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