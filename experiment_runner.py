"""Соединение данных, исполнителя, независимой оценки и финального обучения."""
from __future__ import annotations
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import threading

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from contracts import TaskType, MetricName, SandboxOutput, EvaluationReport, direction_for_metric, MetricDirection
from model_code import DRIVER_CODE
from sandbox import run_experiment
from validation import read_predictions, score_predictions


def normalize_spec(spec, train, test, sample):
    if any(frame is None for frame in (train, test, sample)):
        raise ValueError('Required train.csv, test.csv, sample_submission.csv are missing')
    if len(sample.columns) != 2:
        raise ValueError('This implementation requires an ID and a single prediction column')
    if spec.target_col not in train or spec.id_col not in test or spec.id_col not in train:
        raise ValueError('Could not identify target or ID column')
    if list(sample.columns) != [spec.id_col, spec.target_col]:
        raise ValueError('Sample submission columns do not match the task')
    for frame in (train, test, sample):
        if frame[spec.id_col].isna().any() or frame[spec.id_col].duplicated().any():
            raise ValueError('IDs must be non-missing and unique in each input file')
        frame[spec.id_col] = frame[spec.id_col].astype(str)
    if set(test[spec.id_col]) != set(sample[spec.id_col]):
        raise ValueError('Sample and test IDs differ')
    if train[spec.target_col].isna().any():
        raise ValueError('Missing training labels')
    if spec.task_type == 'multiclass_classification':
        spec.task_type = 'multiclass'
    if spec.task_type not in ('regression', 'binary', 'multiclass'):
        raise ValueError('Unsupported task type')
    if spec.task_type != 'regression':
        train[spec.target_col] = train[spec.target_col].astype(str)
        spec.classes = sorted(train[spec.target_col].unique().tolist())
        if len(spec.classes) < 2:
            raise ValueError('Classification requires at least two classes')
    metric = MetricName(spec.metric)
    spec.higher_is_better = direction_for_metric(metric) is MetricDirection.MAXIMIZE
    if metric in (MetricName.ROC_AUC, MetricName.LOGLOSS, MetricName.F1) and spec.task_type != 'binary':
        raise ValueError('Single-column probability/F1 metrics currently require binary classification')
    return spec


def split_rows(train, spec, profile, seed=42):
    if len(train) < 8:
        raise ValueError('At least eight rows are required for research validation')
    indices = np.arange(len(train))
    # Use real datetime columns when present; numeric range shifts alone do not
    # establish temporal structure. Group hints are used only for repeated IDs.
    time_col = profile.time_col
    if time_col and time_col in profile.datetime_cols:
        ordered = pd.to_datetime(train[time_col], errors='raise').sort_values(kind='stable').index.to_numpy()
        cut = max(2, int(len(ordered) * .8))
        tr, va = ordered[:cut], ordered[cut:]
    elif profile.group_col and train[profile.group_col].nunique() < len(train):
        groups = train[profile.group_col].fillna('__missing__').astype(str)
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(indices, groups=groups))
    else:
        if spec.task_type == 'regression':
            tr, va = train_test_split(indices, test_size=.2, random_state=seed)
        else:
            # Singleton classes must remain in training.
            counts = train[spec.target_col].value_counts()
            eligible = indices[train[spec.target_col].map(counts).to_numpy() >= 2]
            singletons = np.setdiff1d(indices, eligible)
            classes = train.iloc[eligible][spec.target_col].nunique()
            n_valid = max(classes, int(len(eligible) * .2), 2)
            if len(eligible) - n_valid < classes:
                raise ValueError('Too few rows for stratified validation')
            tr, va = train_test_split(eligible, test_size=n_valid, random_state=seed,
                                     stratify=train.iloc[eligible][spec.target_col])
            tr = np.concatenate([tr, singletons])
    if spec.task_type != 'regression' and not set(train.iloc[va][spec.target_col]).issubset(set(train.iloc[tr][spec.target_col])):
        raise ValueError('Validation contains a class absent from training; review split')
    return np.asarray(tr), np.asarray(va)


def atomic_submission(values, test_ids, sample, spec, destination):
    candidate = pd.DataFrame({spec.id_col: [str(x) for x in test_ids], spec.target_col: values})
    # Match sample order, preserving its exact column names.
    candidate = candidate.set_index(spec.id_col).loc[sample[spec.id_col]].reset_index()
    temporary = destination.with_suffix('.tmp.csv')
    candidate.to_csv(temporary, index=False)
    probability = spec.metric in ('roc_auc', 'logloss')
    read_predictions(temporary, sample[spec.id_col], spec.id_col, spec.target_col,
                     numeric=spec.task_type == 'regression' or probability,
                     probability=probability, classes=spec.classes)
    os.replace(temporary, destination)


def save_fallback(train, test, sample, spec, destination):
    y = train[spec.target_col]
    if spec.task_type == 'regression':
        value = float(y.median() if spec.metric == 'mae' else y.mean())
    elif spec.metric in ('roc_auc', 'logloss'):
        value = float((y == spec.classes[-1]).mean())
    else:
        value = y.mode().iloc[0]
    atomic_submission(np.full(len(test), value), test[spec.id_col], sample, spec, destination)


async def execute_async(code, directory, timeout, memory, cpus):
    cancel = threading.Event()
    worker = asyncio.create_task(asyncio.to_thread(
        run_experiment, code, directory, timeout, memory, cpus,
        expected_outputs=('test_predictions.csv',), cancel_event=cancel))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancel.set()
        # Await cleanup before the orchestrator proceeds to another experiment.
        await asyncio.shield(worker)
        raise


class CandidateEvaluator:
    def __init__(self, train, test, sample, spec, tr, va, root, budget,
                 experiment_timeout=900, memory_mb=24000, cpus=8):
        self.train, self.test, self.sample, self.spec = train, test, sample, spec
        self.tr, self.va = tr, va
        self.root, self.budget = Path(root), budget
        self.experiment_timeout, self.memory_mb, self.cpus = experiment_timeout, memory_mb, cpus
        self.best_score = None
        self.best_code = None
        self.root.mkdir(parents=True, exist_ok=True)
        self.submission = self.root.parent / 'submission.csv'

    async def _execute(self, code, final, timeout):
        folder = Path(tempfile.mkdtemp(prefix='final-' if final else 'experiment-', dir=self.root))
        fit = self.train if final else self.train.iloc[self.tr]
        validation = self.train.iloc[self.va].drop(columns=[self.spec.target_col])
        predict = self.test if final else pd.concat([validation, self.test], ignore_index=True)
        fit.to_csv(folder / 'train.csv', index=False)
        predict.to_csv(folder / 'predict.csv', index=False)
        config = dict(id_col=self.spec.id_col, target_col=self.spec.target_col,
                      task_type=self.spec.task_type, metric=self.spec.metric, seed=42,
                      cpu_limit=self.cpus, validation_count=0 if final else len(self.va))
        (folder / 'experiment.json').write_text(json.dumps(config), encoding='utf-8')
        (folder / 'candidate.py').write_text(code, encoding='utf-8')
        timeout = min(timeout, self.budget.remaining_seconds - 2)
        if timeout <= 0:
            raise TimeoutError('No time remains for the experiment')
        result = await execute_async(code + '\n' + DRIVER_CODE, folder, timeout, self.memory_mb, self.cpus)
        return folder, result

    def _read(self, folder, name, ids):
        probability = self.spec.metric in ('roc_auc', 'logloss')
        return read_predictions(folder / name, ids, self.spec.id_col, self.spec.target_col,
                                numeric=self.spec.task_type == 'regression' or probability,
                                probability=probability, classes=self.spec.classes)

    async def __call__(self, hypothesis, task_type, timeout_seconds):
        if not hypothesis.code:
            return EvaluationReport(None, error='Hypothesis has no executable code')
        folder, result = await self._execute(hypothesis.code, False, min(timeout_seconds, self.experiment_timeout))
        report = SandboxOutput(**asdict(result))
        if result.status != 'success':
            return EvaluationReport(None, report, result.error or result.status)
        try:
            preds = self._read(folder, 'validation_predictions.csv', self.train.iloc[self.va][self.spec.id_col])
            truth = self.train.iloc[self.va][self.spec.target_col].to_numpy()
            if self.spec.metric in ('roc_auc', 'logloss', 'f1'):
                truth = (truth == self.spec.classes[-1]).astype(int)
                if self.spec.metric == 'f1':
                    preds = (preds == self.spec.classes[-1]).astype(int)
            score = score_predictions(task_type, truth, preds, self.spec.metric)
            test_preds = self._read(folder, 'test_predictions.csv', self.test[self.spec.id_col])
            improved = self.best_score is None or (score > self.best_score if self.spec.higher_is_better else score < self.best_score)
            if improved:
                atomic_submission(test_preds, self.test[self.spec.id_col], self.sample, self.spec, self.submission)
                self.best_score, self.best_code = score, hypothesis.code
            return EvaluationReport(score, report)
        except (ValueError, OSError, KeyError) as error:
            return EvaluationReport(None, report, f'Validation failed: {error}')

    async def finalize(self):
        if not self.best_code or self.budget.remaining_seconds <= 3:
            return False
        folder, result = await self._execute(self.best_code, True, self.budget.remaining_seconds - 2)
        if result.status != 'success':
            return False
        values = self._read(folder, 'test_predictions.csv', self.test[self.spec.id_col])
        atomic_submission(values, self.test[self.spec.id_col], self.sample, self.spec, self.submission)
        return True
