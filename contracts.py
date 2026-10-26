from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Mapping, Optional


class TaskType(str, Enum):
    REGRESSION = "regression"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    BINARY_CLASSIFICATION = "binary"


class MetricName(str, Enum):
    RMSE = "rmse"
    MACRO_F1 = "macro_f1"
    MAE = "mae"
    MSE = "mse"
    R2 = "r2"
    ACCURACY = "accuracy"
    F1 = "f1"
    WEIGHTED_F1 = "weighted_f1"
    ROC_AUC = "roc_auc"
    LOGLOSS = "logloss"


class MetricDirection(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ExperimentStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class SandboxOutput:
    status: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_sec: float = 0.0
    peak_memory_mb: float | None = None
    error: str | None = None
    output_files: tuple[str, ...] = ()
    missing_outputs: tuple[str, ...] = ()
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    def __post_init__(self) -> None:
        if not self.status.strip():
            raise ValueError("Sandbox status must not be empty")
        if not math.isfinite(self.duration_sec) or self.duration_sec < 0:
            raise ValueError("Sandbox duration must be finite and non-negative")
        if self.peak_memory_mb is not None and (
            not math.isfinite(self.peak_memory_mb) or self.peak_memory_mb < 0
        ):
            raise ValueError("Peak memory must be finite and non-negative")


@dataclass(frozen=True)
class EvaluationReport:
    metric_value: float | None
    sandbox_output: SandboxOutput | None = None
    error: str | None = None


def metric_for_task(task_type: TaskType) -> MetricName:
    if task_type is TaskType.REGRESSION:
        return MetricName.RMSE
    if task_type is TaskType.BINARY_CLASSIFICATION:
        return MetricName.ROC_AUC
    if task_type is TaskType.MULTICLASS_CLASSIFICATION:
        return MetricName.MACRO_F1
    raise ValueError(f"Unsupported task type: {task_type}")


def direction_for_metric(metric: MetricName) -> MetricDirection:
    if metric in (MetricName.RMSE, MetricName.MAE, MetricName.MSE, MetricName.LOGLOSS):
        return MetricDirection.MINIMIZE
    return MetricDirection.MAXIMIZE



@dataclass(frozen=True)
class Hypothesis:
    id: str
    description: str
    config: Mapping[str, Any] = field(default_factory=dict)
    code: str | None = None

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
    sandbox_output: SandboxOutput | None = None

    def __post_init__(self) -> None:
        regression_metrics = {MetricName.RMSE, MetricName.MAE, MetricName.MSE, MetricName.R2}
        if (self.task_type is TaskType.REGRESSION) != (self.metric in regression_metrics):
            raise ValueError('Metric does not match task type')
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


@dataclass
class TaskSpec:
    task_type: str = "regression"   
    target_col: str = "target"
    id_col: str = "id"
    metric: str = "rmse"               # rmse | mae | r2 | macro_f1 | f1 | accuracy | roc_auc | logloss
    higher_is_better: bool = False
    description: str = ""
    classes: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class DataProfile:
    n_rows: int = 0
    n_test_rows: int = 0
    n_cols: int = 0
    numeric_cols: list = field(default_factory=list)
    categorical_cols: list = field(default_factory=list)
    datetime_cols: list = field(default_factory=list)
    text_cols: list = field(default_factory=list)
    id_like_cols: list = field(default_factory=list)
    high_missing_cols: list = field(default_factory=list)
    constant_cols: list = field(default_factory=list)
    high_cardinality_cols: list = field(default_factory=list)
    target_stats: dict = field(default_factory=dict)
    group_col: Optional[str] = None
    time_col: Optional[str] = None
    shift: dict = field(default_factory=dict) 
    summary_text: str = ""                    

    def to_dict(self):
        return asdict(self)


def stub_task_spec() -> TaskSpec: #заглушка
    return TaskSpec(task_type="regression", target_col="site_eui", id_col="id",
                    metric="rmse", higher_is_better=False, description="stub")


def stub_profile() -> DataProfile:
    return DataProfile(n_rows=1000, n_test_rows=500, n_cols=10,
                       numeric_cols=["f1", "f2"], categorical_cols=["c1"],
                       summary_text="STUB: 1000 rows, 10 cols, regression.")