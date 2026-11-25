"""Repeat the 56-variable development-set screen."""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

from experiment import SEED

missing, rare = '（缺失）', '（低频）'

def regions(train, val):
    if pd.api.types.is_numeric_dtype(train) and train.nunique() > 5:
        nonzero = train.dropna()
        split_zero = nonzero.eq(0).mean() >= .15 and nonzero.ne(0).sum() >= 50
        if split_zero:
            nonzero = nonzero[nonzero.ne(0)]
        interior = np.unique(nonzero.quantile([.2,.4,.6,.8]).to_numpy())
        interior = interior[(interior > nonzero.min()) & (interior < nonzero.max())]
        edges = np.r_[-np.inf, interior, np.inf]
        def transform(series):
            labels = pd.cut(series, bins=edges).astype(object).fillna(missing)
            return labels.mask(series.eq(0), '0') if split_zero else labels
    else:
        def transform(series):
            return series.astype(object).fillna(missing)
    train_levels, val_levels = transform(train), transform(val)
    freq = train_levels.value_counts()
    keep = freq[freq >= .02 * train.size].index
    def pool(levels):
        return levels.where(levels.isin(keep) | levels.eq(missing), rare)
    return pool(train_levels), pool(val_levels)

def observed(levels, labels):
    return labels.groupby(levels, sort=False).agg(['size', 'mean'])

def run(source, output):
    raw = pd.read_csv(source).set_index('CustomerID')
    target = raw.pop('Churn').eq('Yes').astype(int)
    dev_ids, _ = train_test_split(raw.index, stratify=target, test_size=.2, random_state=SEED)
    x = raw.loc[dev_ids].drop(columns='ServiceArea')
    y = target.loc[dev_ids]
    x['DroppedPer100Minutes'] = 100 * x.DroppedCalls / x.MonthlyMinutes.where(x.MonthlyMinutes.gt(0))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(x, y))
    rows = []
    for name, column in x.items():
        oof = np.full(y.size, np.nan)
        aucs = []
        for fit, val in folds:
            g_fit, g_val = regions(column.iloc[fit], column.iloc[val])
            sy, vy = y.iloc[fit], y.iloc[val]
            counts = sy.groupby(g_fit, sort=False).agg(['size','sum'])
            mean = sy.mean()
            estimates = (counts['sum'] + 25 * mean) / (counts['size'] + 25)
            prob = g_val.map(estimates).astype(float).fillna(mean)
            oof[val] = prob
            aucs.append(roc_auc_score(vy, prob))
        levels, _ = regions(column,column)
        obs = observed(levels, y)
        eligible = obs[obs['size'].ge(.02*y.size) & ~obs.index.isin([missing, rare])]
        span = eligible['mean'].max() - eligible['mean'].min() if len(eligible) > 1 else np.nan
        rows.append((name,np.mean(aucs),np.min(aucs),np.max(aucs),roc_auc_score(y,oof),span))
    tbl = pd.DataFrame(rows,columns=['field','cv_auc_mean','fold_min','fold_max','oof_auc','regular_gap']).sort_values('cv_auc_mean',ascending=False)
    tbl['rank'] = np.arange(1,len(tbl)+1)
    output.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(output, index=False)
    print(tbl.head(10).round(4).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output', type=Path, default=Path('runs/eda/feature_screen.csv'))
    args = parser.parse_args()
    run(args.csv, args.output)
