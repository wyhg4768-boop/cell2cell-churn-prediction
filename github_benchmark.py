"""Retrain published Cell2Cell model settings on our common evaluation split."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from experiment import ChurnData, SEED, pipeline, scores, threshold_for_f1


def tree_models(y_dev):
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    weight = (y_dev == 0).sum() / (y_dev == 1).sum()
    skn_lightgbm = {
        'learning_rate': .04842147117612157, 'num_leaves': 57, 'max_depth': 6,
        'min_child_samples': 156, 'subsample': .8635788546358015,
        'colsample_bytree': .9580706780343285, 'reg_alpha': .015983273605254422,
        'reg_lambda': 8.540546845775278, 'min_split_gain': .11647421019585992,
    }
    skn_xgboost = {
        'n_estimators': 194, 'learning_rate': .0742499952424855, 'max_depth': 4,
        'min_child_weight': 6, 'reg_alpha': .022332084450689918,
        'reg_lambda': .006328735787217118, 'subsample': .9461312762050127,
        'colsample_bytree': .9795374127310457,
    }
    models = {
        'Leyla XGBoost': [XGBClassifier(max_depth=3, min_child_weight=1, scale_pos_weight=weight,
                                      learning_rate=.1, n_estimators=400, subsample=.9,
                                      colsample_bytree=.9, n_jobs=2, random_state=42)],
        'SKN24 XGBoost': [XGBClassifier(**skn_xgboost, eval_metric='auc', n_jobs=2,
                                       random_state=42)],
        'SKN24 LightGBM': [LGBMClassifier(**skn_lightgbm, n_estimators=trees,
                                         class_weight='balanced', n_jobs=2,
                                         random_state=42, verbosity=-1)
                          for trees in (200, 400, 700)],
        'yoel3imari 软投票': [VotingClassifier([
            ('lr', LogisticRegression(class_weight='balanced', max_iter=2500, random_state=42)),
            ('xgb', XGBClassifier(scale_pos_weight=weight, eval_metric='aucpr', n_jobs=2,
                                  random_state=42)),
            ('lgbm', LGBMClassifier(class_weight='balanced', verbosity=-1, n_jobs=2,
                                    random_state=42)),
        ], voting='soft')],
    }
    return {name: [pipeline(model) for model in options] for name, options in models.items()}


def catboost_models(features):
    from catboost import CatBoostClassifier

    categorical = list(features.select_dtypes(exclude=np.number).columns)
    for column in categorical:
        features[column] = features[column].fillna('Missing').astype(str)
    return {'SKN24 CatBoost': [CatBoostClassifier(
        iterations=988, depth=6, learning_rate=.028124781194669244,
        l2_leaf_reg=5.691073892881603, random_strength=.20531112463660664,
        bagging_temperature=.81735113850536, rsm=.6236002964928216,
        min_data_in_leaf=68, auto_class_weights='Balanced', loss_function='Logloss',
        eval_metric='AUC', cat_features=categorical, thread_count=4, random_seed=42,
        verbose=False, allow_writing_files=False,
    )]}


def run(path, group, output):
    data = ChurnData(path).load()
    target = data.Churn.eq('Yes').astype(int)
    development, test = train_test_split(data.index, stratify=target, test_size=.2, random_state=SEED)
    for split in (development, test):
        print(len(split), hashlib.sha256(np.asarray(split, dtype='<i8').tobytes()).hexdigest())
    y_dev, y_test = target.loc[development], target.loc[test]
    features = ChurnData.features(data, extended=False)
    candidates = tree_models(y_dev) if group == 'trees' else catboost_models(features)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(development, y_dev))
    rows = []
    selected = []
    for name, options in candidates.items():
        best_f1 = -1
        for number, model in enumerate(options, start=1):
            probability = cross_val_predict(model, features.loc[development], y_dev,
                                            cv=folds, method='predict_proba')[:, 1]
            threshold = threshold_for_f1(y_dev, probability)
            oof_f1 = scores(y_dev, probability, threshold)['f1']
            print(name, number, 'OOF F1', round(oof_f1, 4),
                  '阈值', round(threshold, 4), flush=True)
            if oof_f1 > best_f1:
                best_f1, best = oof_f1, (number, model, threshold)
        selected.append((name, best_f1, *best))
    for name, best_f1, number, model, threshold in selected:
        model.fit(features.loc[development], y_dev)
        probability = model.predict_proba(features.loc[test])[:, 1]
        rows.append(dict(model=name, candidate=number, oof_f1=best_f1,
                         test_f1_default=f1_score(y_test, probability >= .5),
                         **scores(y_test, probability, threshold)))
    results = pd.DataFrame(rows)
    results.to_csv(output, index=False)
    print(results.round(4).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('group', choices=['trees', 'catboost'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.csv, args.group, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f'实验失败：{error}\n')
