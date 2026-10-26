# 两级库存网络优化 — 结果

20 个需求区 · 8 个候选区域仓 · 5 个品类 · 服务水平 95% · 固定成本系数 1

## 三种做法的年化成本

| policy | open_dcs | fixed | inbound | outbound | holding_cycle | holding_safety | total | vs_status_quo |
|---|---|---|---|---|---|---|---|---|
| single central (status quo) | 0 | $0 | $0 | $4,784,198 | $727,597 | $1,044,131 | $6,555,926 | 0.0% |
| all 8 DCs, nearest assignment | 8 | $660,000 | $1,150,778 | $1,955,984 | $727,597 | $904,244 | $5,398,602 | 17.7% |
| MILP optimum | 5 | $425,000 | $936,269 | $1,397,496 | $727,597 | $967,706 | $4,454,068 | 32.1% |

- 相对现状（单中心直发）节省 **32.1%**
- 相对「全开 8 仓」节省 **17.5%**
- LP 下界 $4,454,068，**最优性 gap 0.00%**（本例 LP 松弛直接取到整数解，所以 gap 为 0——说明这个规模和成本结构下连续松弛已经足够紧，规模再大才需要 branch-and-bound）

## 选址与分配

开仓：Lagos、Rotterdam、Shanghai、Singapore、São Paulo

| 需求区 | 服务的仓 |
|---|---|
| Australia | Singapore |
| Brazil | São Paulo |
| China | Shanghai |
| Cuba | central direct |
| Dominican Republic | central direct |
| El Salvador | central direct |
| France | Rotterdam |
| Germany | Rotterdam |
| Guatemala | central direct |
| Honduras | central direct |
| India | Singapore |
| Indonesia | Singapore |
| Italy | Rotterdam |
| Mexico | central direct |
| Nicaragua | central direct |
| Nigeria | Lagos |
| Spain | Rotterdam |
| Turkey | Rotterdam |
| United Kingdom | Rotterdam |
| United States | central direct |

## 灵敏度

 fixed_cost_multiplier  open_dcs    total_cost  vs_status_quo  service_level
                 0.500         6 4,229,239.625          0.355            NaN
                 1.000         5 4,454,068.046          0.321            NaN
                 1.500         3 4,615,267.535          0.296            NaN
                 2.000         3 4,752,767.535          0.275            NaN
                 1.000         5 4,240,329.011          0.353          0.900
                 1.000         5 4,454,068.046          0.321          0.950
                 1.000         5 4,855,006.915          0.259          0.990
