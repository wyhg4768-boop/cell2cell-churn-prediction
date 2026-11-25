"""Search local hyperparameter grids on the report split, then score the winners."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split

from experiment import ChurnData, SEED, lightgbm, pipeline, scores, threshold_for_f1


LR, RF, LGB, XGB, STACK = '逻辑回归', '随机森林', 'LightGBM', 'XGBoost', '四模型 Stacking'
GRIDS = {
    LR: [
        {'C': [.03, .1, .3], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']},
        {'C': [.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs']},
    ],
    RF: [
        {'max_depth': [10, 14], 'n_estimators': [100, 200], 'min_samples_leaf': [1, 5]},
        {'max_depth': [8, 10], 'n_estimators': [60], 'min_samples_leaf': [1]},
    ],
    LGB: [
        {'max_depth': [5, 6], 'n_estimators': [100, 220],
         'learning_rate': [.05, .1], 'class_weight': ['balanced']},
        {'max_depth': [4], 'n_estimators': [200], 'learning_rate': [.05], 'class_weight': [None]},
    ],
    XGB: [
        {'max_depth': [5, 6], 'n_estimators': [100, 220], 'learning_rate': [.05, .1]},
        {'max_depth': [4], 'n_estimators': [200], 'learning_rate': [.05]},
        {'max_depth': [5], 'n_estimators': [120], 'learning_rate': [.05]},
    ],
    STACK: [{'C': [.03, .1, .3, 1], 'class_weight': ['balanced', None]}],
}


def make_estimator(name, params):
    if name == LR:
        return LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED, **params)
    if name == STACK:
        return LogisticRegression(max_iter=1000, **params)
    if name == RF:
        return RandomForestClassifier(random_state=SEED, class_weight='balanced', n_jobs=1, **params)
    if name == LGB:
        depth, trees = params['max_depth'], params['n_estimators']
        return lightgbm(depth, trees, True).set_params(**params, n_jobs=1)
    from xgboost import XGBClassifier

    return XGBClassifier(
        **params, min_child_weight=10, subsample=.85, colsample_bytree=.8,
        tree_method='hist', eval_metric='logloss', random_state=SEED, n_jobs=1,
    )


def initialize(csv_path):
    global features, labels, development, test
    raw = ChurnData(csv_path).load()
    labels = raw.Churn.eq('Yes').astype(int).to_numpy()
    development, test = train_test_split(
        np.arange(len(raw)), stratify=labels, test_size=.2, random_state=SEED,
    )
    features = ChurnData.features(raw)


def encoded(fit, evaluate):
    encoder = pipeline(LogisticRegression()).steps[0][1]
    train = encoder.fit_transform(features.iloc[fit])
    return train, encoder.transform(features.iloc[evaluate])


def predict(model, train, target, evaluate):
    return model.fit(train, target).predict_proba(evaluate)[:, 1]


def stacking_base_estimators():
    return {
        LR: make_estimator(LR, dict(C=1, penalty='l2', solver='lbfgs')),
        RF: make_estimator(RF, dict(max_depth=10, n_estimators=60, min_samples_leaf=1)),
        XGB: make_estimator(XGB, dict(max_depth=5, n_estimators=120, learning_rate=.05)),
        LGB: make_estimator(LGB, dict(max_depth=5, n_estimators=100, learning_rate=.1, class_weight='balanced')),
    }


def stacking_inputs(fit, evaluate):
    bases = list(stacking_base_estimators().values())
    meta_train = np.empty((len(fit), len(bases)))
    inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
    for tr, val in inner.split(fit, labels[fit]):
        train, validation = encoded(fit[tr], fit[val])
        for j, model in enumerate(bases):
            meta_train[val, j] = predict(clone(model), train, labels[fit[tr]], validation)
    train, validation = encoded(fit, evaluate)
    meta_validation = np.column_stack([
        predict(model, train, labels[fit], validation) for model in bases
    ])
    return meta_train, meta_validation


def run_batch(name, candidates, fit, evaluate):
    train, validation = stacking_inputs(fit, evaluate) if name == STACK else encoded(fit, evaluate)
    return np.column_stack([
        predict(make_estimator(name, params), train, labels[fit], validation)
        for params in candidates
    ])


def run(csv_path, output, workers):
    initialize(csv_path)
    output.mkdir(parents=True, exist_ok=True)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(development, labels[development]))
    candidates = {name: list(ParameterGrid(grid)) for name, grid in GRIDS.items()}
    all_scores = {name: np.full((len(development), len(grid)), np.nan) for name, grid in candidates.items()}
    print('候选配置:', {name: len(grid) for name, grid in candidates.items()}, flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        batches = {}
        for name in (STACK, XGB, RF, LGB, LR):
            for fold, (tr, val) in enumerate(folds, 1):
                job = pool.submit(run_batch, name, candidates[name], development[tr], development[val])
                batches[job] = (name, fold, val)
        for job in as_completed(batches):
            name, fold, val = batches[job]
            all_scores[name][val] = job.result()
            print(f'{name} 第 {fold}/5 折完成', flush=True)

        rows, best = [], {}
        for name, matrix in all_scores.items():
            if not np.isfinite(matrix).all():
                raise ValueError(f'{name} 的折外预测不完整')
            model_rows = []
            for params, prob in zip(candidates[name], matrix.T):
                cutoff = threshold_for_f1(labels[development], prob)
                result = scores(labels[development], prob, cutoff)
                model_rows.append(dict(model=name, parameters=json.dumps(params, sort_keys=True), **result))
            chosen = max(model_rows, key=lambda row: row['f1'])
            best[name] = chosen
            rows.extend(dict(row, selected=row is chosen) for row in model_rows)
            print(f'{name} 入选 {chosen["parameters"]} 折外 F1={chosen["f1"]:.6f}', flush=True)
        pd.DataFrame(rows).to_csv(output / 'tuning_candidates.csv', index=False)

        finals = {
            name: pool.submit(run_batch, name, [json.loads(row['parameters'])], development, test)
            for name, row in best.items()
        }
        test_rows = []
        for name, job in finals.items():
            prob = job.result()[:, 0]
            chosen = best[name]
            test_rows.append(dict(
                model=name, parameters=chosen['parameters'], oof_f1=chosen['f1'],
                test_f1_default=f1_score(labels[test], prob >= .5),
                **scores(labels[test], prob, chosen['threshold'])
            ))
        results = pd.DataFrame(test_rows)
        results.to_csv(output / 'test_comparison.csv', index=False)
        print(results.round(6).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output', type=Path, default=Path('revision_results/local_grid'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    run(args.csv, args.output, args.workers)
