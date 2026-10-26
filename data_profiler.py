from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from .contracts import DataProfile, TaskSpec


class C:
    HIGH_MISSING_THRESHOLD = 0.95  
    ID_UNIQUE_RATIO = 0.95          
    TEXT_MIN_AVG_LEN = 40           
    HIGH_CARDINALITY = 50            
    GROUP_MAX_UNIQUE_RATIO = 0.5 
    SUMMARY_MAX_TOKENS = 1500       
    MAX_COLS_LISTED = 25            


_GROUP_HINTS = ("id", "building", "customer", "client", "user", "store", "shop", "site",
                "device", "account", "group", "patient", "driver")
_TIME_HINTS = ("date", "time", "timestamp", "datetime", "дата")


def _approx_tokens(s: str) -> int:
    return int(len(s) / 3.2) + 1


def _looks_datetime(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    sample = s.dropna().astype(str).head(200)
    if len(sample) == 0 or sample.str.len().median() < 6:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() > 0.9
    except Exception:
        return False


def _classify_columns(df: pd.DataFrame, cols, n: int):
    out = dict(numeric=[], categorical=[], datetime=[], text=[], id_like=[],
               high_missing=[], constant=[], high_card=[])
    for c in cols:
        try:
            s = df[c]
            miss = s.isna().mean()
            nun = s.nunique(dropna=True)
            if miss >= C.HIGH_MISSING_THRESHOLD:
                out["high_missing"].append(c); continue
            if nun <= 1:
                out["constant"].append(c); continue
            if _looks_datetime(s):
                out["datetime"].append(c); continue
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                is_int = pd.api.types.is_integer_dtype(s)
                if is_int and nun / n > C.ID_UNIQUE_RATIO and s.is_monotonic_increasing:
                    out["id_like"].append(c)
                else:
                    out["numeric"].append(c)
                continue
            if nun / n > C.ID_UNIQUE_RATIO:
                avg_len = s.dropna().astype(str).str.len().mean()
                (out["text"] if avg_len >= C.TEXT_MIN_AVG_LEN else out["id_like"]).append(c)
                continue
            avg_len = s.dropna().astype(str).str.len().mean()
            if avg_len >= C.TEXT_MIN_AVG_LEN:
                out["text"].append(c); continue
            out["categorical"].append(c)
            if nun > C.HIGH_CARDINALITY:
                out["high_card"].append(c)
        except Exception:
            out["numeric"].append(c)
    return out


def _target_stats(y: pd.Series, spec: TaskSpec) -> dict:
    st = {"missing": int(y.isna().sum())}
    y = y.dropna()
    try:
        if spec.task_type == "regression":
            yn = pd.to_numeric(y, errors="coerce").dropna()
            q = yn.quantile([0.01, 0.5, 0.99])
            st.update(mean=round(float(yn.mean()), 3), std=round(float(yn.std()), 3),
                      min=round(float(yn.min()), 3), max=round(float(yn.max()), 3),
                      p01=round(float(q.iloc[0]), 3), median=round(float(q.iloc[1]), 3),
                      p99=round(float(q.iloc[2]), 3), skew=round(float(yn.skew()), 2),
                      all_positive=bool((yn > 0).all()))
        else:
            vc = y.value_counts(normalize=True)
            st["n_classes"] = int(len(vc))
            st["class_share"] = {str(k): round(float(v), 4) for k, v in vc.head(30).items()}
            st["imbalance_ratio"] = round(float(vc.max() / max(vc.min(), 1e-9)), 1)
            st["rare_classes_lt1pct"] = [str(k) for k, v in vc.items() if v < 0.01][:20]
    except Exception:
        pass
    return st


def _detect_group(train, cols, spec, n):
    best = None
    for c in cols:
        try:
            if not any(h in c.lower() for h in _GROUP_HINTS):
                continue
            nun = train[c].nunique()
            if 1 < nun < C.GROUP_MAX_UNIQUE_RATIO * n and nun > 5:
                if best is None or nun > train[best].nunique():
                    best = c
        except Exception:
            continue
    return best


def _detect_time(train, datetime_cols, numeric_cols):
    if datetime_cols:
        return datetime_cols[0]
    for c in numeric_cols:
        if any(h in c.lower() for h in _TIME_HINTS):
            return c
    return None


def _out_of_range(train, test, cols_num):
    res = {}
    if test is None:
        return res
    for c in cols_num:
        try:
            if c not in test.columns:
                continue
            tr, te = train[c].dropna(), test[c].dropna()
            if len(tr) == 0 or len(te) == 0 or te.nunique() > 50 or tr.nunique() > 50:
                continue
            if te.min() > tr.max() or te.max() < tr.min():
                res[c] = {"train_values": sorted(tr.unique().tolist())[:20],
                          "test_values": sorted(te.unique().tolist())[:20]}
        except Exception:
            continue
    return res


def _shift(train, test, cols_cat, cols_num):
    sh = {}
    if test is None:
        return sh
    try:
        new_cat = {}
        for c in cols_cat[:60]:
            seen = set(train[c].dropna().unique())
            share = (~test[c].dropna().isin(seen)).mean() if test[c].notna().any() else 0
            if share > 0.01:
                new_cat[c] = round(float(share), 3)
        if new_cat:
            sh["unseen_category_share_in_test"] = new_cat
        oor = _out_of_range(train, test, cols_num)
        if oor:
            sh["test_outside_train_range"] = oor
        diff = {}
        for c in cols_num + cols_cat:
            if c in test.columns:
                d = abs(train[c].isna().mean() - test[c].isna().mean())
                if d > 0.1:
                    diff[c] = round(float(d), 3)
        if diff:
            sh["missing_share_diff"] = diff
    except Exception:
        pass
    return sh


def _build_summary(p: DataProfile, spec: TaskSpec, train, dictionary, max_tokens: int) -> str:
    L = []
    L.append(f"TASK: {spec.task_type}, target='{spec.target_col}', metric={spec.metric} "
             f"({'higher' if spec.higher_is_better else 'lower'} is better), id='{spec.id_col}'")
    L.append(f"DATA: train {p.n_rows} rows, test {p.n_test_rows} rows, {p.n_cols} feature cols")
    ts = p.target_stats
    if spec.task_type == "regression" and "mean" in ts:
        L.append(f"TARGET: mean={ts['mean']} std={ts['std']} min={ts['min']} max={ts['max']} "
                 f"p01={ts['p01']} median={ts['median']} p99={ts['p99']} skew={ts['skew']} "
                 f"positive_only={ts['all_positive']}")
    elif "class_share" in ts:
        L.append(f"TARGET: {ts['n_classes']} classes, imbalance max/min={ts['imbalance_ratio']}, "
                 f"shares={ts['class_share']}")
        if ts.get("rare_classes_lt1pct"):
            L.append(f"RARE classes (<1%): {ts['rare_classes_lt1pct']}")

    def lst(name, cols):
        if not cols:
            return
        shown = cols[:C.MAX_COLS_LISTED]
        extra = f" (+{len(cols) - len(shown)} more)" if len(cols) > len(shown) else ""
        L.append(f"{name} [{len(cols)}]: {', '.join(shown)}{extra}")

    lst("NUMERIC", p.numeric_cols)
    lst("CATEGORICAL", p.categorical_cols)
    lst("HIGH-CARDINALITY (>%d)" % C.HIGH_CARDINALITY, p.high_cardinality_cols)
    lst("DATETIME", p.datetime_cols)
    lst("TEXT", p.text_cols)
    lst("ID-LIKE (drop)", p.id_like_cols)
    lst("CONSTANT (drop)", p.constant_cols)
    lst("HIGH-MISSING (drop)", p.high_missing_cols)

    try:
        feats = [c for c in p.numeric_cols + p.categorical_cols]
        miss = train[feats].isna().mean().sort_values(ascending=False)
        miss = miss[miss > 0.05].head(10)
        if len(miss):
            L.append("MISSING: " + ", ".join(f"{k}={v:.0%}" for k, v in miss.items()))
    except Exception:
        pass
    if p.group_col:
        L.append(f"GROUP col (use GroupKFold?): {p.group_col}")
    if p.time_col:
        L.append(f"TIME col (consider time split): {p.time_col}")
    oor = p.shift.get("test_outside_train_range")
    if oor:
        L.append("VALIDATION HINT: test is a different period. " + "; ".join(
            f"{c}: train={v['train_values']} test={v['test_values']}" for c, v in oor.items())
            + ". Use time-based validation (hold out the latest train value), NOT random KFold.")
    for k, v in p.shift.items():
        if k == "test_outside_train_range":
            continue
        L.append(f"SHIFT {k}: {dict(list(v.items())[:8])}")

    if dictionary is not None:
        try:
            dd = dictionary.set_index("feature")["description_ru"].dropna()
            used = [c for c in (p.numeric_cols + p.categorical_cols) if c in dd.index][:C.MAX_COLS_LISTED]
            if used:
                L.append("DESCRIPTIONS: " + "; ".join(f"{c}={str(dd[c])[:50]}" for c in used))
        except Exception:
            pass

    text = "\n".join(L)
    while _approx_tokens(text) > max_tokens and len(L) > 3:
        L.pop()
        text = "\n".join(L)
    return text


def profile_data(train: pd.DataFrame, test: pd.DataFrame, spec: TaskSpec,
                 data_dictionary: pd.DataFrame = None,
                 max_tokens: int = C.SUMMARY_MAX_TOKENS) -> DataProfile:
    p = DataProfile()
    try:
        n = max(len(train), 1)
        feat_cols = [c for c in train.columns if c not in (spec.target_col, spec.id_col)]
        p.n_rows, p.n_cols = len(train), len(feat_cols)
        p.n_test_rows = len(test) if test is not None else 0

        cl = _classify_columns(train, feat_cols, n)
        p.numeric_cols, p.categorical_cols = cl["numeric"], cl["categorical"]
        p.datetime_cols, p.text_cols = cl["datetime"], cl["text"]
        p.id_like_cols, p.high_missing_cols = cl["id_like"], cl["high_missing"]
        p.constant_cols, p.high_cardinality_cols = cl["constant"], cl["high_card"]

        if spec.target_col in train.columns:
            p.target_stats = _target_stats(train[spec.target_col], spec)
        p.group_col = _detect_group(train, feat_cols, spec, n)
        p.shift = _shift(train, test, p.categorical_cols, p.numeric_cols)
        oor = p.shift.get("test_outside_train_range", {})
        p.time_col = next(iter(oor), None) or _detect_time(train, p.datetime_cols, p.numeric_cols)
    except Exception:
        pass
    try:
        p.summary_text = _build_summary(p, spec, train, data_dictionary, max_tokens)
    except Exception:
        p.summary_text = f"TASK: {spec.task_type}, target={spec.target_col}, metric={spec.metric}"
    return p