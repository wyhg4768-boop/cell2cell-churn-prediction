"""Reproduce the full development-only ablation and the alternate-fold check."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from experiment import CORE, SEED, ChurnData, lightgbm, pipeline, scores, threshold_for_f1

GROUPS = {
    '账户阶段与设备使用': ['MonthsInService', 'CurrentEquipmentDays'],
    '设备信息': ['CurrentEquipmentDays', 'Handsets', 'HandsetModels', 'HandsetRefurbished',
                'HandsetWebCapable', 'HandsetPrice', 'NewCellphoneUser', 'NotNewCellphoneUser'],
    '月度费用与费用变化': ['MonthlyRevenue', 'TotalRecurringCharge', 'PercChangeRevenues'],
    '通话量及变化': ['MonthlyMinutes', 'PercChangeMinutes', 'OverageMinutes'],
    '两项近期变化': ['PercChangeMinutes', 'PercChangeRevenues'],
    '通话质量': ['DroppedCalls', 'BlockedCalls', 'DroppedBlockedCalls', 'UnansweredCalls'],
    '三项挽留记录': ['RetentionCalls', 'RetentionOffersAccepted', 'MadeCallToRetentionTeam'],
    '账单与用量八项及其状态': [*CORE, 'CoreInfo_Status'],
}


def evaluate(case, features, label, folds, baseline_threshold=None):
    kind, name, columns = case
    model = pipeline(lightgbm(5, 100, True).set_params(n_jobs=1))
    predict = cross_val_predict(model, features.drop(columns=columns), label, cv=folds,
                                method='predict_proba')[:, 1]
    threshold = threshold_for_f1(label, predict)
    fixed = threshold if baseline_threshold is None else baseline_threshold
    return dict(kind=kind, name=name, columns=' | '.join(columns), **scores(label, predict, threshold),
                f1_fixed=scores(label, predict, fixed)['f1'])


def run(csv_path, output, workers, alternate):
    raw = ChurnData(csv_path).load()
    target = raw.Churn.eq('Yes').astype(int)
    ids, _ = train_test_split(raw.index, stratify=target, test_size=.2, random_state=SEED)
    features, label = ChurnData.features(raw.loc[ids]), target.loc[ids]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=20261125 if alternate else SEED)
                 .split(features, label))
    base = evaluate(('baseline', '完整', []), features, label, folds)
    if alternate:
        chosen = ('MonthsInService', 'CurrentEquipmentDays')
        cases = [('single', col, [col]) for col in chosen]
        cases.append(('group', '账户阶段与设备使用', list(chosen)))
    else:
        cases = [('single', col, [col]) for col in features.columns]
        cases.extend(('group', name, cols) for name, cols in GROUPS.items())
    check = partial(evaluate, features=features, label=label, folds=folds,
                    baseline_threshold=base['threshold'])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [base, *pool.map(check, cases)]
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output', type=Path, default=Path('runs/eda/full_ablation.csv'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--alternate-folds', action='store_true')
    args = parser.parse_args()
    run(args.csv, args.output, args.workers, args.alternate_folds)
