"""Export deterministic, row-local features and development-fold assignments."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from experiment import ChurnData, SEED


def export(source, output):
    dataset = ChurnData(source)
    raw = dataset.load()
    label = raw.Churn.eq('Yes').astype('int8')
    development, test = train_test_split(
        raw.index, test_size=.2, random_state=SEED, stratify=label,
    )
    features = dataset.features(raw)
    features.insert(0, 'Churn', label)
    features.insert(0, 'validation_fold', np.zeros(len(raw), dtype='int8'))
    features.insert(0, 'split', 'development')
    features.loc[test, 'split'] = 'test'
    folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fold, (_, indices) in enumerate(folds.split(development, label.loc[development]), 1):
        features.loc[development[indices], 'validation_fold'] = fold
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False, compression={'method': 'gzip', 'mtime': 0})
    print(features.groupby('split')['Churn'].agg(['size', 'sum']).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output', type=Path, default=Path('data/processed/cell2cell_features.csv.gz'))
    args = parser.parse_args()
    export(args.csv, args.output)
