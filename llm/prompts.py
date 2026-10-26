import pandas as pd
import re
from typing import List, Any


def build_first_iteration_prompt(
        df_train: pd.DataFrame,
        target_col: str,
        task_type: str,  # 'regression' или 'multiclass'
        target_metric: str,  # 'RMSE' или 'Macro F1'
        val_ids: List[Any],  # Список id, которые попали в валидацию
        data_dictionary: pd.DataFrame = None  # Опционально, если Роль 1 передает
) -> str:
    """
    Формирует System Prompt для Qwen для первой итерации.
    """
    # 1. Собираем информацию о признаках (исключая таргет и id)
    feature_info = []
    for col in df_train.columns:
        if col in [target_col, 'id']:
            continue

        dtype = str(df_train[col].dtype)
        desc = ""
        if data_dictionary is not None and 'feature' in data_dictionary.columns:
            match = data_dictionary[data_dictionary['feature'] == col]
            if not match.empty:
                desc = f" (Описание: {match.iloc[0].get('description_ru', '')})"

        feature_info.append(f"- {col}: {dtype}{desc}")

    features_str = "\n".join(feature_info)

    # 2. Форматируем val_ids (если их много, лучше сослаться на файл)
    if len(val_ids) > 50:
        val_ids_str = f"Возьми список id для валидации из файла `val_ids.csv` (колонка 'id')."
    else:
        val_ids_str = f"Используй строго следующие id для валидации: {val_ids}"

    # 3. Базовые правила
    base_rules = f"""
Ты — экспертный ML-инженер. Твоя задача — сгенерировать Python-код для решения задачи машинного обучения.

КОНТЕКСТ:
- Целевая переменная: '{target_col}'
- Тип задачи: {task_type}
- Метрика: {target_metric}
- Данные: файлы train.csv и test.csv находятся в текущей рабочей папке.

ДОСТУПНЫЕ ПРИЗНАКИ:
{features_str}

СТРОГИЕ ПРАВИЛА ВАЛИДАЦИИ:
- {val_ids_str}
- Обучи модель на всех остальных строках из train.csv.
- Сделай предсказания для валидационных id и сохрани их в файл `validation_predictions.csv` с колонками `id` и `{target_col}` (ориентируйся на sample_submission.csv).
- НЕ ВЫВОДИ метрики в stdout (например, print(rmse)). Оценка будет произведена внешним модулем. Твой код должен только сохранить файл.

ПРАВИЛА ГЕНЕРАЦИИ КОДА:
1. Код должен быть самодостаточным скриптом (Python 3.12).
2. Используй только библиотеки из списка: pandas, numpy, scikit-learn, lightgbm, xgboost, catboost.
3. Не используй предобученные модели и не ходи в интернет.
4. Обработай пропуски и закодируй категориальные признаки.
5. Сначала напиши краткое обоснование гипотезы (1-2 предложения), затем блок кода в ```python ... ```.
"""

    # 4. Специфика по типу задачи
    if task_type == 'regression':
        task_specific = f"""
СПЕЦИФИКА РЕГРЕССИИ (Метрика: {target_metric}):
- Рекомендуемые модели: CatBoostRegressor, LGBMRegressor, XGBRegressor.
- Попробуй создать 1-2 новых признака на основе логики (например, отношения, разности).
"""
    elif task_type == 'multiclass':
        task_specific = f"""
СПЕЦИФИКА КЛАССИФИКАЦИИ (Метрика: {target_metric}):
- Рекомендуемые модели: CatBoostClassifier, LGBMClassifier, XGBClassifier.
- Для категориальных признаков используй OneHotEncoder (handle_unknown='ignore') или нативный cat_features в CatBoost.
- Macro F1 чувствителен к редким классам. Если есть дисбаланс, используй class_weight='balanced'.
"""
    else:
        raise ValueError(f"Неизвестный тип задачи: {task_type}")

    return base_rules + task_specific


def extract_python_code(qwen_response: str) -> str:
    """
    Извлекает код из ответа Qwen (между ```python и ```).
    """
    match = re.search(r"```python\s*(.*?)\s*```", qwen_response, re.DOTALL)
    if match:
        return match.group(1)

    # Fallback, если Qwen забыл markdown
    if "import pandas" in qwen_response or "import numpy" in qwen_response:
        lines = qwen_response.split("\n")
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith(("import ", "from ")):
                start_idx = i
                break
        return "\n".join(lines[start_idx:])

    raise ValueError("Не удалось извлечь Python-код из ответа Qwen")