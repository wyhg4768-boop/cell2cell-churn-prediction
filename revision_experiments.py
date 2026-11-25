"""Tune the core churn models and export aggregate report statistics."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
import json

from experiment import CORE, SEED, ChurnData, lightgbm, pipeline, scores, threshold_for_f1, xgboost


def groupings(data):
    return {
        'tenure': pd.cut(data.MonthsInService, [0, 12, 24, np.inf], labels=['1–12', '13–24', '25+']),
        'minutes': pd.cut(data.PercChangeMinutes, [-np.inf, -20, 20, np.inf],
                          labels=['≤ −20', '(−20, 20]', '> 20']).cat.add_categories('Missing').fillna('Missing'),
        'revenue': pd.cut(data.MonthlyRevenue, [-np.inf, 30, 60, 100, np.inf],
                          labels=['≤30', '30–60', '60–100', '>100']).cat.add_categories('Missing').fillna('Missing'),
        'charge': pd.cut(data.TotalRecurringCharge, [-np.inf, 30, 40, 49, 60, np.inf],
                         labels=['≤30', '(30, 40]', '(40, 49]', '(49, 60]', '>60']
                         ).cat.add_categories('Missing').fillna('Missing'),
        'retention': data.MadeCallToRetentionTeam,
        'equipment': pd.cut(data.CurrentEquipmentDays, [-np.inf, 365, 730, np.inf],
                            labels=['≤365', '366–730', '>730']).cat.add_categories('Missing').fillna('Missing'),
    }


def group_rates(values, grouping):
    return (values.groupby(grouping.rename('group'), observed=True)
            .agg(n='size', events='sum', rate='mean').reset_index().to_dict('records'))


def exploration(raw, y, output):
    groups = groupings(raw)
    eda = {key: group_rates(y, group) for key, group in groups.items()}
    font_path = Path('/Applications/Microsoft Word.app/Contents/Resources/DFonts/SimHei.ttf')
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'PingFang SC', 'Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.1), constrained_layout=True)
    panels = [('tenure', '在网时长（月）'), ('equipment', '设备使用天数'),
              ('charge', '月均经常性收费'), ('minutes', '通话量变化（原始值）')]
    for ax, (key, title) in zip(axes.flat, panels):
        rows = eda[key]
        translations = {'Missing': '缺失', 'Yes': '是', 'No': '否'}
        bars = ax.bar([translations.get(r['group'], r['group']).replace('−', '-') for r in rows],
                      [r['rate'] for r in rows], color='#355D78', width=.60)
        ax.bar_label(bars, labels=[f'{r["rate"]:.1%}\nn={r["n"]:,}' for r in rows], fontsize=8, padding=3)
        ax.axhline(y.mean(), ls='--', color='#a55c33', linewidth=1)
        ax.set(title=title, ylabel='流失率', ylim=(0,.68))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(output / 'eda_groups.png', dpi=160)
    plt.close(fig)

    fields = {'CurrentEquipmentDays': '设备使用天数', 'TotalRecurringCharge': '月均经常性收费',
              'PercChangeMinutes': '通话量变化'}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.25), constrained_layout=True)
    quantiles = {}
    for ax, (field, label) in zip(axes, fields.items()):
        column = raw[field]
        quantiles[field] = {str(i): column[y.eq(i)].dropna().quantile([.25,.5,.75]).tolist()
                            for i in (0, 1)}
        low, high = column.quantile([.01,.99])
        for value, name, color in [(0,'留存组','#355D78'), (1,'流失组','#a55c33')]:
            ax.hist(column[y.eq(value)].dropna().clip(low,high), bins=np.linspace(low,high,26),
                    density=True, histtype='step', label=name, color=color, linewidth=1.6)
        ax.set(title=label, ylabel='密度')
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.savefig(output / 'eda_distributions.png', dpi=160)
    plt.close(fig)
    return eda, quantiles


def run(csv_path, output):
    output.mkdir(parents=True, exist_ok=True)
    raw = ChurnData(csv_path).load()
    target = raw.Churn.eq('Yes').astype(int)
    dev, test = train_test_split(raw.index, stratify=target, test_size=.2, random_state=SEED)
    y_dev, y_test = target.loc[dev], target.loc[test]
    eda, quantiles = exploration(raw.loc[dev], y_dev, output)
    missing = raw[CORE].isna()
    patterns = missing.apply(lambda row: ' / '.join(row.index[row]) or '完整', axis=1).value_counts()
    stats = {'eda':eda,'quantiles':quantiles,'missing_patterns':patterns.to_dict()}
    (output / 'report_stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))

    def lr(c):
        return LogisticRegression(C=c,class_weight='balanced',max_iter=1000)

    def rf(depth, trees):
        return RandomForestClassifier(n_estimators=trees,max_depth=depth,
                                      class_weight='balanced',random_state=SEED,n_jobs=2)

    def stack(c):
        return StackingClassifier(
            estimators=[('lr',pipeline(lr(1))), ('rf',pipeline(rf(10,60))),
                        ('xgb',pipeline(xgboost(5,120))), ('lgbm',pipeline(lightgbm(5,100,True)))],
            final_estimator=lr(c), cv=StratifiedKFold(3,shuffle=True,random_state=SEED),
            stack_method='predict_proba', n_jobs=1)

    configs = {
        '逻辑回归': [(f'C={c}',pipeline(lr(c))) for c in (.1,1,10)],
        '随机森林': [(f'深度={depth}，树数={trees}',pipeline(rf(depth,trees)))
                   for depth,trees in [(8,60),(10,60),(14,100)]],
        'LightGBM': [(f'深度={d}，树数={n}，平衡权重={w}',pipeline(lightgbm(d,n,w)))
                     for d,n,w in [(5,100,True),(4,200,False),(6,220,True)]],
        'XGBoost': [(f'深度={d}，树数={n}',pipeline(xgboost(d,n)))
                    for d,n in [(4,200),(5,120),(6,220)]],
        '四模型 Stacking': [(f'第二层 C={c}',stack(c)) for c in (.1,1)],
    }
    x=ChurnData.features(raw)
    folds=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(x.loc[dev],y_dev))
    grid=[]
    best={}
    for name, choices in configs.items():
        for parameters, estimator in choices:
            prob=cross_val_predict(estimator,x.loc[dev],y_dev,cv=folds,method='predict_proba')[:,1]
            cutoff=threshold_for_f1(y_dev,prob)
            summary=scores(y_dev,prob,cutoff)
            grid.append(dict(model=name,parameters=parameters,**summary))
            print(name,parameters,round(summary['f1'],4),round(cutoff,4),flush=True)
            if name not in best or summary['f1'] > best[name][2]['f1']:
                best[name]=(parameters,estimator,summary)
    pd.DataFrame(grid).to_csv(output/'tuning_candidates.csv',index=False)
    rows=[]
    for name,(parameters,estimator,summary) in best.items():
        model=clone(estimator).fit(x.loc[dev],y_dev)
        prob=model.predict_proba(x.loc[test])[:,1]
        cutoff=summary['threshold']
        result=scores(y_test,prob,cutoff)
        rows.append(dict(model=name,parameters=parameters,oof_f1=summary['f1'],
                         test_f1_default=f1_score(y_test,prob>=.5),**result))
        if name == '四模型 Stacking':
            pred=pd.Series(prob>=cutoff,index=test)
            tn,fp,fn,tp=confusion_matrix(y_test,pred).ravel()
            set_positive=y_test.eq(1)
            stats['confusion']=dict(tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))
            test_groups = groupings(raw.loc[test])
            test_groups['usage'] = pd.cut(
                raw.loc[test, 'MonthlyMinutes'], [-np.inf, 122, 276, 478, 842, np.inf],
                labels=['≤122', '(122, 276]', '(276, 478]', '(478, 842]', '>842'],
            ).cat.add_categories('Missing').fillna('Missing')
            stats['errors'] = {
                key: group_rates((~pred)[set_positive], group[set_positive])
                for key, group in test_groups.items()
            }
            fpr,tpr,_=roc_curve(y_test,prob)
            pr,rec,_=precision_recall_curve(y_test,prob)
            fig,axes=plt.subplots(1,2,figsize=(9,3.2),constrained_layout=True)
            axes[0].plot(fpr,tpr,c='#355D78',label=f'Stacking (AUC {result["auc"]:.3f})')
            axes[0].plot([0,1],[0,1],ls='--',c='#a55c33',label='Random ranking')
            axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='ROC')
            axes[1].plot(rec,pr,c='#355D78',label=f'Stacking (AP {result["ap"]:.3f})')
            axes[1].axhline(y_test.mean(),ls='--',c='#a55c33',label='Random ranking')
            axes[1].scatter(result['recall'],result['precision'],color='#355D78')
            axes[1].set(xlabel='Recall',ylabel='Precision',title='Precision–recall')
            for ax in axes:
                ax.set(xlim=(0,1),ylim=(0,1.02))
                ax.legend(fontsize=8)
                ax.spines[['top','right']].set_visible(False)
            fig.savefig(output/'report_curves.png',dpi=160)
            plt.close(fig)
    (output/'report_stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
    result=pd.DataFrame(rows)
    result.to_csv(output/'test_comparison.csv',index=False)
    print(result.round(4).to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv',type=Path)
    parser.add_argument('--output',type=Path,default=Path('revision_results'))
    args=parser.parse_args()
    try:
        run(args.csv,args.output)
    except (OSError,ValueError) as error:
        parser.exit(1,f'实验失败：{error}\n')
