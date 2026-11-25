"""Re-evaluate Cell2Cell churn models using labeled customers only."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 20261124
CORE = ['MonthlyRevenue', 'MonthlyMinutes', 'TotalRecurringCharge', 'DirectorAssistedCalls',
        'OverageMinutes', 'RoamingCalls', 'PercChangeMinutes', 'PercChangeRevenues']
ENGINEERED_INPUTS = ['ServiceArea', 'HandsetPrice', 'DroppedCalls', 'BlockedCalls',
                     'MonthsInService', 'CurrentEquipmentDays', 'CustomerCareCalls']


class ChurnData:
    def __init__(self, path):
        self.path = path

    def load(self):
        frame = pd.read_csv(self.path)
        required = {'CustomerID', 'Churn', 'AgeHH1', 'AgeHH2', *CORE, *ENGINEERED_INPUTS}
        if missing := required.difference(frame.columns):
            raise ValueError(f'缺少字段 {sorted(missing)}')
        if frame.CustomerID.isna().any() or frame.CustomerID.duplicated().any():
            raise ValueError('客户 ID 缺失或重复')
        if set(frame.Churn) != {'Yes', 'No'}:
            raise ValueError('需要带 Yes/No 真实流失标签的数据')
        return frame.set_index('CustomerID')

    @staticmethod
    def features(frame, extended=False):
        features = frame.drop(columns=['Churn', 'ServiceArea']).copy()
        missing = features[CORE].isna()
        features['CoreInfo_Status'] = np.select(
            [missing.all(axis=1), missing.any(axis=1)], [-1, 1], default=0)
        ages = features.pop('AgeHH1').to_frame()
        ages['AgeHH2'] = features.pop('AgeHH2')
        features['Age_Combined'] = ages.where((ages >= 18) & (ages <= 99)).mean(axis=1).fillna(0)
        if extended:
            features['area_prefix'] = frame['ServiceArea'].str.slice(0, 3)
            features['handset_price_missing'] = frame['HandsetPrice'].eq('Unknown').astype(int)
            features['HandsetPrice'] = pd.to_numeric(frame['HandsetPrice'], errors='coerce')
            minutes = frame['MonthlyMinutes'].clip(lower=0) + 1
            features['dropped_blocked_per_100min'] = (
                100 * (frame['DroppedCalls'] + frame['BlockedCalls']) / minutes)
            features['care_calls_per_100min'] = 100 * frame['CustomerCareCalls'] / minutes
            features['device_days_per_month_in_service'] = (
                frame['CurrentEquipmentDays'] / frame['MonthsInService'].clip(lower=1))
        features[CORE] = features[CORE].fillna(0)
        return features


def pipeline(model):
    encoder = ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()),
         make_column_selector(dtype_include=np.number)),
        ('cat', make_pipeline(SimpleImputer(strategy='constant', fill_value='Missing'),
                              OneHotEncoder(handle_unknown='ignore')),
         make_column_selector(dtype_include=object)),
    ])
    return make_pipeline(encoder, model)


def lightgbm(depth, trees, weighted):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        max_depth=depth, num_leaves=2 ** depth - 1, n_estimators=trees,
        learning_rate=.05 if trees > 100 else .1, colsample_bytree=.8,
        reg_alpha=.1, reg_lambda=1, min_child_samples=50,
        class_weight='balanced' if weighted else None,
        random_state=SEED, n_jobs=2, verbosity=-1)


def xgboost(depth, trees):
    from xgboost import XGBClassifier

    return XGBClassifier(
        max_depth=depth, n_estimators=trees, learning_rate=.05, min_child_weight=10,
        subsample=.85, colsample_bytree=.8, tree_method='hist', eval_metric='logloss',
        random_state=SEED, n_jobs=2)


def threshold_for_f1(y, probability):
    precisions, recalls, thresholds = precision_recall_curve(y, probability)
    numerator = 2 * precisions[:-1] * recalls[:-1]
    denominator = precisions[:-1] + recalls[:-1]
    f1 = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    return thresholds[np.argmax(f1)]


def scores(y, probability, threshold):
    prediction = probability >= threshold
    return dict(threshold=threshold, f1=f1_score(y, prediction, zero_division=0),
                precision=precision_score(y, prediction, zero_division=0),
                recall=recall_score(y, prediction, zero_division=0),
                auc=roc_auc_score(y, probability), ap=average_precision_score(y, probability))


def run(path, output):
    raw = ChurnData(path).load()
    target = raw.Churn.eq('Yes').astype(int)
    development, test = train_test_split(raw.index, stratify=target, test_size=.2, random_state=SEED)
    y_dev, y_test = target.loc[development], target.loc[test]
    feature_sets = {flag: ChurnData.features(raw, flag) for flag in (False, True)}
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(development, y_dev))
    inner_folds = StratifiedKFold(3, shuffle=True, random_state=SEED)
    print(f'全部 {len(raw)} 开发 {len(development)} 测试 {len(test)} 流失比例 {target.mean():.4f}', flush=True)
    logistic = LogisticRegression(C=1, class_weight='balanced', max_iter=1000)
    stacking = StackingClassifier(
        estimators=[
            ('lr', pipeline(clone(logistic))),
            ('rf', pipeline(RandomForestClassifier(n_estimators=60, max_depth=10,
                                                   class_weight='balanced', random_state=SEED, n_jobs=2))),
            ('xgb', pipeline(xgboost(5, 120))),
            ('lgbm', pipeline(lightgbm(5, 100, True))),
        ],
        final_estimator=LogisticRegression(class_weight='balanced'),
        cv=inner_folds, stack_method='predict_proba', n_jobs=1)
    candidates = {
        '逻辑回归（原特征）': [(pipeline(logistic), False)],
        'LightGBM（原特征）': [(pipeline(lightgbm(5, 100, True)), False)],
        '四模型Stacking（原特征）': [(stacking, False)],
        'LightGBM（扩展特征）': [(pipeline(lightgbm(depth, trees, weighted)), True)
                              for depth, trees, weighted in [(5, 100, True), (4, 200, False),
                                                             (6, 220, False), (6, 220, True)]],
        'XGBoost（扩展特征）': [(pipeline(xgboost(depth, trees)), True)
                              for depth, trees in [(4, 200), (6, 220), (7, 300)]],
    }
    best = {}
    for name, configurations in candidates.items():
        for number, (model, extended) in enumerate(configurations, start=1):
            x_dev = feature_sets[extended].loc[development]
            oof = cross_val_predict(model, x_dev, y_dev, cv=folds, method='predict_proba')[:, 1]
            threshold = threshold_for_f1(y_dev, oof)
            result = scores(y_dev, oof, threshold)
            print(name, number, 'OOF', {k: round(v, 4) for k, v in result.items()}, flush=True)
            if name not in best or result['f1'] > best[name][0]['f1']:
                best[name] = (result, [(model, extended)], oof, number)
    base_names = ['LightGBM（扩展特征）', 'XGBoost（扩展特征）']
    blend = np.mean([best[name][2] for name in base_names], axis=0)
    best['LightGBM＋XGBoost等权融合'] = (
        scores(y_dev, blend, threshold_for_f1(y_dev, blend)),
        [spec for name in base_names for spec in best[name][1]], blend, 1)
    winner = max(best, key=lambda name: best[name][0]['f1'])
    print('测试前按开发集折外F1选定', winner, flush=True)

    rows = []
    for name, (validation, specs, _, number) in best.items():
        test_probabilities = []
        for model, extended in specs:
            features = feature_sets[extended]
            fitted = clone(model).fit(features.loc[development], y_dev)
            test_probabilities.append(fitted.predict_proba(features.loc[test])[:, 1])
        predicted = np.mean(test_probabilities, axis=0)
        rows.append(dict(model=name, candidate=number, oof_f1=validation['f1'],
                         test_f1_default=f1_score(y_test, predicted >= .5),
                         **scores(y_test, predicted, validation['threshold'])))
    results = pd.DataFrame(rows)
    results.to_csv(output, index=False)
    print(results.round(4).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output', type=Path, default=Path('results.csv'))
    args = parser.parse_args()
    try:
        run(args.csv, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f'实验失败：{error}\n')
