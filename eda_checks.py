"""Reproduce the development-set checks used in the revised report."""
import argparse
import json
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from experiment import ChurnData, SEED, lightgbm, pipeline, scores, threshold_for_f1
from revision_experiments import groupings


def run(csv_path, output):
    raw = ChurnData(csv_path).load()
    target = raw.Churn.eq('Yes').astype(int)
    ids, _ = train_test_split(raw.index, stratify=target, test_size=.2, random_state=SEED)
    data = raw.loc[ids]
    y = target.loc[ids]
    groups = groupings(data)
    cross = (y.groupby([groups['tenure'].rename('tenure'), groups['equipment'].rename('equipment')],
                       observed=True).agg(n='size', rate='mean').reset_index())
    features = ChurnData.features(data)
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    ablations = {}
    for name, columns in [
        ('完整', []),
        ('移除设备使用天数', ['CurrentEquipmentDays']),
        ('移除在网时长', ['MonthsInService']),
        ('移除通话量变化', ['PercChangeMinutes']),
        ('移除在网时长和设备使用天数', ['MonthsInService', 'CurrentEquipmentDays']),
    ]:
        probability = cross_val_predict(pipeline(lightgbm(5, 100, True)),
                                        features.drop(columns=columns), y, cv=folds,
                                        method='predict_proba')[:, 1]
        ablations[name] = scores(y, probability, threshold_for_f1(y, probability))
    output.mkdir(parents=True, exist_ok=True)
    values = dict(cross=cross.to_dict('records'), ablation=ablations)
    (output / 'eda_checks.json').write_text(json.dumps(values, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output', type=Path, default=Path('revision_results'))
    args = parser.parse_args()
    try:
        run(args.csv, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f'实验失败：{error}\n')
