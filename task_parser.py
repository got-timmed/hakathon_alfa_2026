from __future__ import annotations
import re
import pandas as pd
from .contracts import TaskSpec
from pathlib import Path


class C:
    CLASSIF_MAX_UNIQUE = 15        
    CLASSIF_MAX_UNIQUE_RATIO = 0.05 

_METRIC_RULES = [
    (r"macro[\s_\-]*f1|макро[\s_\-]*f1", "macro_f1", True),
    (r"weighted[\s_\-]*f1", "weighted_f1", True),
    (r"roc[\s_\-]*auc|\bauc\b", "roc_auc", True),
    (r"log[\s_\-]*loss|logloss", "logloss", False),
    (r"\brmse\b|root mean squared", "rmse", False),
    (r"\bmae\b|mean absolute", "mae", False),
    (r"\bmse\b", "mse", False),
    (r"\br2\b|r\^2|r-squared", "r2", True),
    (r"accuracy|точност", "accuracy", True),
    (r"\bf1\b", "f1", True),
]
_DEFAULT_METRIC = {"regression": ("rmse", False), "binary": ("roc_auc", True),
                   "multiclass": ("macro_f1", True)}


def _find_id_and_target(train, test, sample_sub):
    id_col, target_col = None, None
    if sample_sub is not None and len(sample_sub.columns) >= 2:
        sub_cols = list(sample_sub.columns)
        if test is not None:
            common = [c for c in sub_cols if c in test.columns]
            if common:
                id_col = common[0]
        if id_col is None:
            id_col = sub_cols[0]
        rest = [c for c in sub_cols if c != id_col]
        if rest:
            target_col = rest[0]
    if target_col is None and train is not None and test is not None:
        diff = [c for c in train.columns if c not in test.columns]
        if diff:
            target_col = diff[0]
    if id_col is None:
        for c in (train.columns if train is not None else []):
            if c.lower() == "id" or c.lower().endswith("_id"):
                id_col = c
                break
    return id_col or "id", target_col or train.columns[-1]


def _task_type(y: pd.Series):
    y = y.dropna()
    n_unique = y.nunique()
    if n_unique <= 1:
        return "regression", []
    numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
    if not numeric:
        return ("binary" if n_unique == 2 else "multiclass"), sorted(map(str, y.unique()))[:200]
    is_int_like = bool((y.round() == y).all())
    if is_int_like and n_unique <= C.CLASSIF_MAX_UNIQUE and n_unique / max(len(y), 1) < C.CLASSIF_MAX_UNIQUE_RATIO:
        return ("binary" if n_unique == 2 else "multiclass"), sorted(y.unique().tolist())
    return "regression", []


def _metric_from_text(text: str):
    t = (text or "").lower()
    for pattern, name, hib in _METRIC_RULES:
        if re.search(pattern, t):
            return name, hib
    return None


def parse_task(description: str, train: pd.DataFrame, test: pd.DataFrame = None,
               sample_submission: pd.DataFrame = None, data_dictionary: pd.DataFrame = None) -> TaskSpec:
    spec = TaskSpec(description=description or "")
    try:
        spec.id_col, spec.target_col = _find_id_and_target(train, test, sample_submission)
    except Exception:
        pass
    try:
        spec.task_type, spec.classes = _task_type(train[spec.target_col])
    except Exception:
        pass
    try:
        found = _metric_from_text(description)
        if found is None:
            found = _DEFAULT_METRIC[spec.task_type]
        reg_metrics = {"rmse", "mae", "mse", "r2"}
        if (spec.task_type == "regression") != (found[0] in reg_metrics):
            found = _DEFAULT_METRIC[spec.task_type]
        spec.metric, spec.higher_is_better = found
    except Exception:
        pass
    return spec


def _find(d: Path, name: str):
    if (d / name).exists():
        return d / name
    hits = sorted(d.glob(f"**/{name}"), key=lambda p: len(p.parts))
    return hits[0] if hits else None


def _read(d: Path, name: str):
    p = _find(d, name)
    return pd.read_csv(p, encoding="utf-8") if p else None


def load_task(task_md_path):
    task_md = Path(task_md_path).resolve()
    d = task_md.parent
    description = task_md.read_text(encoding="utf-8")
    return (description, _read(d, "train.csv"), _read(d, "test.csv"),
            _read(d, "sample_submission.csv"), _read(d, "data_dictionary.csv"), d)