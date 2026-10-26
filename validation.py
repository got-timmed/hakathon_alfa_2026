from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import f1_score, mean_squared_error

from contracts import MetricName, TaskType, metric_for_task


def score_predictions(
    task_type: TaskType,
    y_true: Sequence[object],
    y_pred: Sequence[object],
) -> float:
    if len(y_true) == 0:
        raise ValueError("Validation data must not be empty")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    metric = metric_for_task(task_type)
    if metric is MetricName.RMSE:
        try:
            actual = np.asarray(y_true, dtype=float)
            predicted = np.asarray(y_pred, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("RMSE requires numeric targets and predictions") from error
        if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
            raise ValueError("RMSE targets and predictions must be finite")
        return float(math.sqrt(mean_squared_error(actual, predicted)))

    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
