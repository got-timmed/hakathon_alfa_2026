"""Независимая оценка предсказаний и строгое сопоставление строк CSV."""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import metrics
from contracts import TaskType, MetricName, metric_for_task


def score_predictions(task_type, y_true, y_pred, metric=None):
    if len(y_true) == 0:
        raise ValueError('Validation data must not be empty')
    if len(y_true) != len(y_pred):
        raise ValueError('y_true and y_pred must have the same length')
    metric = MetricName(metric or metric_for_task(task_type))
    if np.asarray(y_true).ndim != 1 or np.asarray(y_pred).ndim != 1:
        raise ValueError('Only one target column is supported')
    if pd.isna(y_true).any() or pd.isna(y_pred).any():
        raise ValueError('Missing labels or predictions')
    regression = task_type is TaskType.REGRESSION
    if regression or metric in (MetricName.ROC_AUC, MetricName.LOGLOSS):
        predicted = np.asarray(y_pred, dtype=float)
        if not np.isfinite(predicted).all():
            raise ValueError('Predictions must be finite')
    if regression:
        actual = np.asarray(y_true, dtype=float)
        if not np.isfinite(actual).all():
            raise ValueError('Targets must be finite')
        fn = {MetricName.RMSE: lambda a,b: math.sqrt(metrics.mean_squared_error(a,b)),
              MetricName.MSE: metrics.mean_squared_error, MetricName.MAE: metrics.mean_absolute_error,
              MetricName.R2: metrics.r2_score}[metric]
        value = fn(actual, predicted)
    elif metric is MetricName.ROC_AUC:
        value = metrics.roc_auc_score(y_true, predicted)
    elif metric is MetricName.LOGLOSS:
        value = metrics.log_loss(y_true, predicted, labels=[0,1])
    elif metric is MetricName.ACCURACY:
        value = metrics.accuracy_score(y_true, y_pred)
    else:
        average = {MetricName.MACRO_F1:'macro', MetricName.WEIGHTED_F1:'weighted', MetricName.F1:'binary'}[metric]
        value = metrics.f1_score(y_true, y_pred, average=average, zero_division=0)
    if not math.isfinite(value):
        raise ValueError('Metric is not finite')
    return float(value)


def read_predictions(path, ids, id_col, target_col, *, numeric=False, classes=None, probability=False):
    frame = pd.read_csv(path, dtype={id_col: str, **({} if numeric else {target_col:str})})
    if list(frame.columns) != [id_col, target_col]:
        raise ValueError(f'Expected columns {[id_col, target_col]}')
    if frame.isna().any().any() or frame[id_col].duplicated().any():
        raise ValueError('Missing values or duplicate prediction IDs')
    expected = pd.Index([str(x) for x in ids])
    if expected.has_duplicates or set(frame[id_col]) != set(expected):
        raise ValueError('Prediction IDs do not match expected IDs')
    aligned = frame.set_index(id_col).loc[expected, target_col]
    if numeric:
        aligned = pd.to_numeric(aligned, errors='raise')
        if not np.isfinite(aligned.to_numpy()).all():
            raise ValueError('Non-finite predictions')
        if probability and not aligned.between(0,1).all():
            raise ValueError('Probabilities must lie in [0,1]')
    elif classes is not None and not aligned.isin([str(x) for x in classes]).all():
        raise ValueError('Unknown predicted class')
    return aligned.to_numpy()
