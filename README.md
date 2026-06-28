# 供应链库存管理分析

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)


用 Tableau Superstore 零售数据做的一个库存管理分析，包含 ABC 分类、EOQ 经济订货批量、安全库存和再订货点四个模块。

## 项目内容

数据是从 Kaggle 下载的 Superstore 数据集，原始 9,994 条交易记录，我按产品 ID 聚合成了 200 个 SKU，覆盖三个品类（家具、科技产品、办公用品）。

分析分成四步：

- **ABC 分类**：按年消耗金额排序，分出 A 类（前80%）、B 类（80-95%）、C 类（后5%）
- **EOQ 分析**：对 A 类产品算最优订货批量，然后做了订货成本和持有成本率的敏感性分析
- **安全库存**：用 Z-score 法算了 99%、95%、90% 三个服务水平下各自需要多少安全库存
- **再订货点**：ROP = 日均需求×提前期 + 安全库存，然后用 90 天窗口模拟库存变化

## 主要结论

ABC 分类的结果挺符合帕累托分布的：A 类产品大概 63% 的 SKU 数量，占了 80% 的库存价值。C 类产品数量不少但加起来只值 5% 的库存金额。

安全库存那边，99% 服务水平（Z=2.33）比 95%（Z=1.65）需要大约 40% 更多的安全库存量。这个数字挺直观的——多一分保障就多一分库存成本。

EOQ 的敏感性分析发现，持有成本率对 EOQ 的影响比订货成本更敏感。也就是说在这个场景下，想办法降低仓储成本比优化采购流程更划算。

90 天模拟里 A 类产品没出现缺货，ROP 策略在模拟中跑通了。

如果按照 ABC 分类结果实施分级管理策略，理论上一年能省大概 $69,279 的持有成本（约 20%）。不过这个是估算值，实际效果还得看企业的具体成本结构。

## 运行方式

```bash
pip install pandas numpy matplotlib seaborn scipy jupyter
cd notebooks
jupyter notebook
# 按 01 → 02 → 03 → 04 顺序打开
```

## 数据

Tableau Superstore Dataset，Kaggle 公开数据。9,994 条交易，2016 年全年。

## License

MIT
