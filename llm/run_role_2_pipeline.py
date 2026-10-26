import pandas as pd
from llm.prompts import build_first_iteration_prompt, extract_python_code
from llm.qwen_client import call_qwen


def run_role_2_pipeline(train_path: str, data_dict_path: str, val_ids: list, task_type: str, metric: str,
                        target_col: str):
    """
    Твоя точка входа. Принимает данные от Роли 1, возвращает код для Роли 3.
    """
    # 1. Читаем данные
    df_train = pd.read_csv(train_path)
    df_dict = pd.read_csv(data_dict_path) if data_dict_path else None

    # 2. Формируем промпт
    prompt = build_first_iteration_prompt(
        df_train=df_train,
        target_col=target_col,
        task_type=task_type,
        target_metric=metric,
        val_ids=val_ids,
        data_dictionary=df_dict
    )

    # 3. Отправляем в Qwen
    response = call_qwen(prompt)

    # 4. Извлекаем код
    code = extract_python_code(response)

    return {
        "raw_response": response,  # Для дебага
        "generated_code": code  # Отдаем Роли 3
    }