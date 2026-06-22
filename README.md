# Supply Chain Inventory Analysis

基于 **Tableau Superstore 真实零售数据集** 的库存管理分析项目，涵盖 ABC 分类、经济订货批量（EOQ）、安全库存和再订货点策略的完整分析链路。

## 📊 项目概览

本项目使用零售行业真实交易数据（9,994条记录，200个SKU，3大品类），系统性地应用供应链库存管理方法论：

| 分析模块 | 方法 | 核心产出 |
|---------|------|---------|
| **ABC 分类** | 帕累托分析（Pareto） | 识别A类产品（前80%消耗金额），制定分级管理策略 |
| **EOQ 分析** | 经济订货批量模型 | 计算最优订货批量，平衡订货成本与持有成本 |
| **安全库存** | 服务水平法（Z-score） | 99%/95%/90%服务水平下的安全库存水平 |
| **再订货点** | ROP = d×LT + SS | 综合策略：90天库存模拟验证 |

## 🛠 技术栈

- **Python 3**：Pandas, NumPy, Matplotlib, Seaborn, SciPy
- **Jupyter Notebook**：分步式分析，含可视化与中文注释
- **数据源**：Tableau Superstore Dataset（Kaggle）

## 📁 项目结构

```
supply-chain-inventory-analysis/
├── README.md
├── data/
│   └── superstore_products.csv    # 聚合后的产品级数据（200 SKU）
├── notebooks/
│   ├── 01_abc_analysis.ipynb      # ABC分类 + 帕累托图
│   ├── 02_eoq_analysis.ipynb      # EOQ计算 + 敏感性分析
│   ├── 03_safety_stock.ipynb      # 安全库存计算
│   └── 04_reorder_point.ipynb     # 再订货点 + 库存模拟
├── images/
│   ├── abc_analysis.png
│   ├── eoq_sensitivity.png
│   ├── safety_stock_analysis.png
│   └── reorder_point_analysis.png
└── requirements.txt
```

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install pandas numpy matplotlib seaborn scipy jupyter

# 2. 运行分析
cd notebooks
jupyter notebook
# 按顺序打开 01 → 02 → 03 → 04
```

## 📈 业务成果（Business Impact）

基于200个SKU的完整分析，得出以下可量化的业务结论：

### 分级管理策略的业务效果

| 指标 | 优化前（统一管理） | 优化后（ABC分级管理） | 改善 |
|------|:---:|:---:|:--:|
| A类产品盘点频率 | 月度 | **周度** | 4x ↑ |
| C类产品管理成本 | 与A类相同 | **双箱系统，月盘** | 管理成本 -67% |
| 预估年度库存持有成本 | $346,394 (25%持有率) | $277,115 | **-$69,279（-20%）** |
| 高价值产品缺货风险 | 无差异化管控 | A类 99% 服务水平 | 缺货概率从 ~5% 降至 ~1% |

### 关键发现

1. **帕累托分布显著**：A类产品（62.5% SKU）贡献 **79.7%** 的库存价值（$110.5万 / $138.6万），验证了ABC分类的必要性
2. **安全库存成本量化**：99%服务水平比95%需要 **约40%更多** 的安全库存——这为企业提供了「成本 vs 服务水平」的决策依据
3. **EOQ敏感性分析**：EOQ对持有成本率的变化比订货成本更敏感，说明**降低仓储成本比优化采购流程更有效**
4. **90天库存模拟验证**：再订货点策略在模拟中**零缺货**运行，验证了ROP = d×LT + SS 策略的有效性
5. **分级管理节省20%持有成本**：通过ABC差异化策略（A类高频率+高服务、C类低频率+低服务），预估可节省约 **$69,279/年** 的库存持有成本

## 📝 数据说明

- 数据来源：[Tableau Superstore Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)
- 原始数据：9,994条交易记录
- 聚合后：200个SKU，3大品类（Furniture / Technology / Office Supplies）
- 时间范围：2016年全年

## 📄 License

MIT
