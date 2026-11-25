"""Recreate the exploratory charts on the development sample."""
import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from experiment import ChurnData, SEED
from revision_experiments import exploration


def run(csv_path, output):
    output.mkdir(parents=True, exist_ok=True)
    raw = ChurnData(csv_path).load()
    target = raw.Churn.eq('Yes').astype(int)
    development, _ = train_test_split(
        raw.index, stratify=target, test_size=.2, random_state=SEED,
    )
    eda, quantiles = exploration(raw.loc[development], target.loc[development], output)
    (output / 'development_eda.json').write_text(
        json.dumps(dict(eda=eda, quantiles=quantiles), ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output', type=Path, default=Path('runs/eda'))
    args = parser.parse_args()
    run(args.csv, args.output)
