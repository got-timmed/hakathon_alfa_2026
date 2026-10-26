from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class TaskType(str, Enum):
    REGRESSION = "regression"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"


class MetricName(str, Enum):
    RMSE = "rmse"
    MACRO_F1 = "macro_f1"


class MetricDirection(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ExperimentStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def metric_for_task(task_type: TaskType) -> MetricName:
    if task_type is TaskType.REGRESSION:
        return MetricName.RMSE
    if task_type is TaskType.MULTICLASS_CLASSIFICATION:
        return MetricName.MACRO_F1
    raise ValueError(f"Unsupported task type: {task_type}")


def direction_for_metric(metric: MetricName) -> MetricDirection:
    if metric is MetricName.RMSE:
        return MetricDirection.MINIMIZE
    if metric is MetricName.MACRO_F1:
        return MetricDirection.MAXIMIZE
    raise ValueError(f"Unsupported metric: {metric}")


@dataclass(frozen=True)
class Hypothesis:
    id: str
    description: str
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Hypothesis id must not be empty")
        if not self.description.strip():
            raise ValueError("Hypothesis description must not be empty")


@dataclass(frozen=True)
class ExperimentResult:
    hypothesis: Hypothesis
    task_type: TaskType
    metric: MetricName
    status: ExperimentStatus
    metric_value: float | None
    elapsed_seconds: float
    started_at: str
    error: str | None = None

    def __post_init__(self) -> None:
        expected_metric = metric_for_task(self.task_type)
        if self.metric is not expected_metric:
            raise ValueError(
                f"{self.task_type.value} requires {expected_metric.value}, "
                f"not {self.metric.value}"
            )
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        if self.status is ExperimentStatus.SUCCEEDED:
            if self.metric_value is None:
                raise ValueError("Successful experiments must have a metric value")
            if not math.isfinite(self.metric_value):
                raise ValueError("Successful experiment metric must be finite")
        elif self.metric_value is not None:
            raise ValueError("Failed experiments must not have a metric value")


@dataclass(frozen=True)
class ResearchOutcome:
    experiments: tuple[ExperimentResult, ...]
    best: ExperimentResult | None
    stop_reason: str | None = None


def is_better(candidate: ExperimentResult, incumbent: ExperimentResult | None) -> bool:
    if candidate.status is not ExperimentStatus.SUCCEEDED:
        return False
    if incumbent is None or incumbent.status is not ExperimentStatus.SUCCEEDED:
        return True
    if candidate.metric is not incumbent.metric:
        raise ValueError("Cannot compare experiments with different metrics")
    assert candidate.metric_value is not None
    assert incumbent.metric_value is not None
    direction = direction_for_metric(candidate.metric)
    if direction is MetricDirection.MINIMIZE:
        return candidate.metric_value < incumbent.metric_value
    return candidate.metric_value > incumbent.metric_value
