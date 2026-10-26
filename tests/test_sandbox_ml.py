"""Сквозные ML-проверки на искусственных данных, без API и внешних датасетов.

Правильные ответы valid остаются в тестовом процессе. Это разделение данных,
а не гарантия защитной изоляции исполнителя. Рабочие файлы временные.
"""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from sklearn.datasets import make_blobs, make_regression
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split

from agent.sandbox import run_experiment


# Такой код представляет ответ Qwen. Его заранее проверили и зафиксировали,
# чтобы тестировать исполнитель без сетевых запросов и расходов API.
CODE_TEMPLATE = '''
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression

with np.load('inputs.npz', allow_pickle=False) as data:
    model = make_pipeline(StandardScaler(), MODEL_EXPRESSION)
    model.fit(data['X_train'], data['y_train'])
    predictions = model.predict(data['X_valid'])
    np.savez('predictions.npz', row_ids=data['valid_ids'], predictions=predictions)
print('Experiment finished')
# Исполнитель возвращает этот текст, но оценщик не доверяет заявленной метрике.
print('claimed_score=999999')
'''


@unittest.skipUnless(sys.platform == 'linux', 'Исполнитель требует Linux/WSL')
class MLIntegrationTests(unittest.TestCase):
    def execute_model(self, X, y, model_expression, stratify=None):
        train_idx, valid_idx = train_test_split(
            np.arange(len(y)), test_size=.25, random_state=42, stratify=stratify)
        with tempfile.TemporaryDirectory(prefix='automl-ml-test-') as directory:
            root = Path(directory)
            np.savez(root / 'inputs.npz', X_train=X[train_idx], y_train=y[train_idx],
                     X_valid=X[valid_idx], valid_ids=valid_idx)
            result = run_experiment(
                CODE_TEMPLATE.replace('MODEL_EXPRESSION', model_expression), root,
                timeout_sec=30, memory_limit_mb=1024, cpu_limit=1,
                expected_outputs=('predictions.npz',))
            self.assertEqual(result.status, 'success', result)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.missing_outputs, ())
            self.assertEqual(result.output_files, (str(root / 'predictions.npz'),))
            with np.load(result.output_files[0], allow_pickle=False) as output:
                predictions = output['predictions'].copy()
                np.testing.assert_array_equal(output['row_ids'], valid_idx)
            self.assertEqual(predictions.shape, y[valid_idx].shape)
            self.assertTrue(np.isfinite(predictions).all())
            return y[valid_idx], predictions, y[train_idx], result

    def test_regression_training_predictions_and_independent_rmse(self):
        X, y = make_regression(n_samples=400, n_features=6, noise=.1, random_state=7)
        truth, predictions, train_y, result = self.execute_model(X, y, 'Ridge(alpha=.01)')
        rmse = float(np.sqrt(mean_squared_error(truth, predictions)))
        baseline_rmse = float(np.sqrt(mean_squared_error(truth, np.full_like(truth, train_y.mean()))))
        self.assertLess(rmse, 1.0)
        self.assertLess(rmse, baseline_rmse * .1)
        print(f'\nRegression: RMSE={rmse:.4f}; constant baseline={baseline_rmse:.4f}; '
              f'time={result.duration_sec:.2f}s; sampled RSS={result.peak_memory_mb:.1f} MiB')

    def test_multiclass_training_predictions_and_independent_macro_f1(self):
        X, y = make_blobs(n_samples=450, n_features=4, centers=3,
                          cluster_std=.7, random_state=7)
        truth, predictions, train_y, result = self.execute_model(
            X, y, 'LogisticRegression(max_iter=300, random_state=42)', stratify=y)
        score = f1_score(truth, predictions, average='macro')
        baseline = f1_score(truth, np.full_like(truth, np.bincount(train_y).argmax()), average='macro')
        self.assertGreater(score, .95)
        self.assertGreater(score, baseline)
        print(f'\nClassification: Macro F1={score:.4f}; constant baseline={baseline:.4f}; '
              f'time={result.duration_sec:.2f}s; sampled RSS={result.peak_memory_mb:.1f} MiB')


if __name__ == '__main__':
    unittest.main()
