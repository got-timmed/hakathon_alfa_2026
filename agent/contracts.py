from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional


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
