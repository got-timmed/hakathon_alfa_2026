"""Надёжный начальный эксперимент и единый драйвер сгенерированной функции."""
BASELINE_CODE = r'''
def train_predict(X_train, y_train, X_predict, task_type, metric, seed, cpu_limit):
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    numeric = list(X_train.select_dtypes(include='number').columns)
    categorical = [c for c in X_train.columns if c not in numeric]
    X_train, X_predict = X_train.copy(), X_predict.copy()
    for frame in (X_train, X_predict):
        frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)
        for col in categorical:
            frame[col] = frame[col].fillna('__missing__').astype(str)
    transform = ColumnTransformer([
        ('num', SimpleImputer(strategy='median', keep_empty_features=True), numeric),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical),
    ])
    model_class = HistGradientBoostingRegressor if task_type == 'regression' else HistGradientBoostingClassifier
    model = make_pipeline(transform, model_class(max_iter=100, max_leaf_nodes=31, random_state=seed))
    model.fit(X_train, y_train)
    if task_type == 'binary' and metric in ('roc_auc', 'logloss'):
        return model.predict_proba(X_predict)[:, -1]
    return model.predict(X_predict)
'''

DRIVER_CODE = r'''
if __name__ == '__main__':
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path
    c = json.loads(Path('experiment.json').read_text())
    id_col, target = c['id_col'], c['target_col']
    dtypes = {id_col: str}
    if c['task_type'] != 'regression':
        dtypes[target] = str
    train = pd.read_csv('train.csv', dtype=dtypes)
    predict = pd.read_csv('predict.csv', dtype={id_col: str})
    features = [x for x in train.columns if x not in (id_col, target)]
    predictions = np.asarray(train_predict(
        train[features].copy(), train[target].copy(), predict[features].copy(),
        c['task_type'], c['metric'], c['seed'], c['cpu_limit']))
    if predictions.ndim != 1 or len(predictions) != len(predict):
        raise ValueError('train_predict must return one prediction per row')
    n = c['validation_count']
    result = pd.DataFrame({id_col: predict[id_col], target: predictions})
    if n:
        result.iloc[:n].to_csv('validation_predictions.csv', index=False)
    result.iloc[n:].to_csv('test_predictions.csv', index=False)
'''
