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

## 📈 关键发现

1. **A类产品集中度高**：少数产品占据绝大部分库存价值，应优先管理
2. **EOQ与单价负相关**：高单价产品的最优订货批量更小，应高频小批量采购
3. **安全库存成本**：99%服务水平比95%需要约40%更多的安全库存
4. **分级管理有效**：ABC分类后实施差异化策略，可在保证服务水平的同时降低总体库存成本

## 📝 数据说明

- 数据来源：[Tableau Superstore Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)
- 原始数据：9,994条交易记录
- 聚合后：200个SKU，3大品类（Furniture / Technology / Office Supplies）
- 时间范围：2016年全年

## 📄 License

MIT
