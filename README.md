# 基于客户信息的电信客户流失预测

本项目用 Cell2Cell 电信客户的账单、通话、在网时长、设备和服务记录预测是否流失。采用同一套分层划分和开发集折外预测，比较逻辑回归、随机森林、LightGBM、XGBoost 和四模型 Stacking。46 组配置在开发集内选参和确定阈值，留出测试集用于最终评分。

开发集折外 F1 选出的主模型是 XGBoost。留出测试集 F1 为 **0.5014**，精确率为 **37.54%**，召回率为 **75.46%**，ROC-AUC 为 **0.6760**。[模型结果](results/grid/test_comparison.csv)

## 数据来源和文件

- [Kaggle 原始数据页](https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom)和 [Cell2Cell 字段字典](https://github.com/leylatulu/Telecom-Churn-Analysis/blob/main/Cell2Cell-Data-Documentation.pdf)。
- `data/raw/cell2celltrain.csv` 是报告使用的原始带标签文件，包含 51,047 行和 58 列。SHA-256 为 `3f6b8ff0b8e8f5975aa0de9f1f54967e6c8cb867a9bec6cbb64f4c8990f9a645`。Kaggle 官方元数据将许可标为 `Unknown`。原始和派生数据存放在课程私有仓库，进一步分发前请核对数据权利。
- `data/processed/cell2cell_features.csv.gz` 为完成逐行特征处理的压缩 CSV，包含 55 个输入特征、二元标签、数据集归属和验证折编号。开发样本的验证折编号为 1–5，测试样本为 0。客户编号不作为模型输入。模型使用的中位数填补、标准化和独热编码仍在每个训练折单独拟合。
- `report/` 包含最终 Word、PDF 和公式图。`results/grid/` 和 `results/eda/` 保存报告使用的输出。`results/baseline/` 和 `results/external/` 保存早期探索记录。

## 运行环境

报告分析在 macOS x86_64、Python 3.9.13 下执行。主要软件包为 pandas 1.5.3、scikit-learn 1.0.2、LightGBM 4.6.0、XGBoost 2.1.4、Matplotlib 3.5.2 和 SciPy 1.13.1。历史 Anaconda 环境实际载入 NumPy 1.21.5，并产生 SciPy 兼容性提示。新环境的 `requirements.txt` 固定为满足 SciPy 约束的 NumPy 1.24.4。保存的报告指标来自原运行环境。更换环境后，中文字体或末位浮点数可能有变化。

```sh
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 复现步骤

在仓库根目录运行。网格搜索和全量移除实验需要数分钟，时间取决于 CPU。新输出写入 `runs/`，可与 `results/` 中保存的结果比较。

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/.mpl"
data=data/raw/cell2celltrain.csv

python export_data.py "$data" --output runs/data/cell2cell_features.csv.gz
python feature_screen.py "$data" --output runs/eda/feature_screen.csv
python report_eda.py "$data" --output runs/eda
python eda_checks.py "$data" --output runs/eda
python full_ablation.py "$data" --output runs/eda/full_ablation.csv
python full_ablation.py "$data" --alternate-folds --output runs/eda/alternate_folds.csv
python grid_search_experiments.py "$data" --workers 4 --output runs/grid
python grid_diagnostics.py "$data" --output runs/grid
python render_formula.py
python build_report.py --eda runs/eda --grid runs/grid --formula runs/report/scoring-equation.png
```

- 分层留出 20% 测试集，随机种子为 `20261124`。开发集做五折分层验证。每组用开发集折外 F1 确定阈值，按折外 F1 选配置，再用固定阈值评价留出集。
- `feature_screen.py` 在开发集筛查 56 个原始或派生指标。`report_eda.py` 重画图 1 和图 2。`eda_checks.py` 重建分层和主要移除对照；`full_ablation.py` 复核 55 项单独移除、成组移除及更换开发集分折时的结果。
- `grid_search_experiments.py` 输出候选参数和留出指标。`grid_diagnostics.py` 重建主模型曲线、混淆矩阵、分组漏判率、逻辑回归系数和 Stacking 基模型评分相关性。
- `build_report.py` 以最终编辑版为模板，更新仓库链接和分析图。正文、表格和版式保存在模板中。可用 Word 或 LibreOffice 将 DOCX 导出为 PDF。重画中文图时需有 SimHei、苹方或 Noto Sans CJK SC 等中文字体。
- `experiment.py`、`revision_experiments.py` 和 `github_benchmark.py` 同时保留了早期分析入口。外部 CatBoost 复验使用 Python 3.12、CatBoost 1.2.8、scikit-learn 1.6.1 和 pandas 3.1.1。

## 结果解读

报告中的测试指标是同一批历史数据上的内部留出表现。随机基线、算法结果和分组漏判用于解释筛查效率。分组统计和逻辑回归系数描述数据或模型中的关联。
