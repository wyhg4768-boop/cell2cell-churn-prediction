"""Reproduce the selected grid model’s test curve and error groups."""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import grid_search_experiments as grid
from revision_experiments import groupings, group_rates



def run(source, out):
    grid.initialize(source)
    results=pd.read_csv(out/'test_comparison.csv')
    best=results.loc[results.oof_f1.idxmax()]
    name=best['model']
    params=json.loads(best['parameters'])
    prob=grid.run_batch(name,[params],grid.development,grid.test)[:,0]
    y=grid.labels[grid.test]
    threshold=float(best.threshold)
    metrics=grid.scores(y,prob,threshold)
    if not all(abs(metrics[key] - best[key]) < 1e-10 for key in metrics):
        raise ValueError('复验指标与保存的测试结果不一致')
    decision=prob>=threshold
    tn,fp,fn,tp=map(int,confusion_matrix(y,decision).ravel())
    raw=grid.ChurnData(source).load().iloc[grid.test]
    actual=y==1
    missed=pd.Series(~decision,index=raw.index)
    groups=groupings(raw)
    groups['usage']=pd.cut(raw.MonthlyMinutes,[-np.inf,122,842,np.inf],labels=['≤122','(122, 842]','>842']).cat.add_categories('Missing').fillna('Missing')
    groups['charge']=pd.cut(raw.TotalRecurringCharge,[-np.inf,30,60,np.inf],labels=['≤30','(30, 60]','>60']).cat.add_categories('Missing').fillna('Missing')
    errors={key:group_rates(missed[actual],g[actual]) for key,g in groups.items()}
    assert all(sum(r['n'] for r in rows)==tp+fn and sum(r['events'] for r in rows)==fn for rows in errors.values())
    lr=results[results.model==grid.LR].iloc[0]
    encoder=grid.pipeline(grid.LogisticRegression()).steps[0][1]
    train=encoder.fit_transform(grid.features.iloc[grid.development])
    model=grid.make_estimator(grid.LR,json.loads(lr.parameters)).fit(train,grid.labels[grid.development])
    numeric_names = encoder.transformers_[0][2]
    numeric_coefs = model.coef_[0][encoder.output_indices_['num']]
    coefficient_by_name = dict(zip(numeric_names, numeric_coefs))
    coefficients = {}
    for field in ('CurrentEquipmentDays', 'MonthsInService', 'PercChangeMinutes', 'TotalRecurringCharge'):
        coefficient = float(coefficient_by_name[field])
        coefficients[field] = dict(coef=coefficient, odds_ratio=float(np.exp(coefficient)))
    encoded_test = encoder.transform(grid.features.iloc[grid.test])
    base_scores = {
        base_name: grid.predict(base, train, grid.labels[grid.development], encoded_test)
        for base_name, base in grid.stacking_base_estimators().items()
    }
    correlation = pd.DataFrame(base_scores).corr(method='spearman').to_dict()
    stats = dict(model=name, metrics=metrics, confusion=dict(tn=tn, fp=fp, fn=fn, tp=tp), errors=errors,
                 device_coefficient=coefficients['CurrentEquipmentDays']['coef'],
                 device_odds_ratio=coefficients['CurrentEquipmentDays']['odds_ratio'],
                 coefficients=coefficients, correlation=correlation)
    (out/'diagnostics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))

    font_path=Path('/Applications/Microsoft Word.app/Contents/Resources/DFonts/SimHei.ttf')
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
    plt.rcParams['font.sans-serif']=['SimHei','PingFang SC','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    fpr,tpr,_=roc_curve(y,prob)
    pr,rec,_=precision_recall_curve(y,prob)
    fig,axes=plt.subplots(1,2,figsize=(9,3.2),constrained_layout=True)
    axes[0].plot(fpr,tpr,c='#355D78',label=f'{name}  AUC {metrics["auc"]:.3f}')
    axes[0].plot([0,1],[0,1],ls='--',c='#a55c33',label='随机排序')
    axes[0].set(xlabel='假阳性率',ylabel='真正率',title='ROC 曲线')
    axes[1].plot(rec,pr,c='#355D78',label=f'{name}  AP {metrics["ap"]:.3f}')
    axes[1].axhline(y.mean(),ls='--',c='#a55c33',label='随机排序')
    axes[1].scatter(metrics['recall'],metrics['precision'],color='#355D78')
    axes[1].set(xlabel='召回率',ylabel='精确率',title='精确率与召回率曲线')
    for ax in axes:
        ax.set(xlim=(0,1),ylim=(0,1.02))
        ax.legend(fontsize=8)
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'report_curves.png',dpi=160)
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output', type=Path, default=Path('revision_results/local_grid'))
    args = parser.parse_args()
    run(args.csv, args.output)
