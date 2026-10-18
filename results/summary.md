# Inventory simulation — refreshed numbers

SKUs = 200 · horizon = 365 days · replications = 100 · common random numbers across policies

- ordering cost $50/order · holding rate 25% · shortage cost $25/unit
- lead time lognormal, mean 7 d, CV 30%; daily demand Poisson (unit demand)

## Policy comparison (365 days × 100 replications, all 200 SKUs)

               policy  fill_rate  stockout_day_share  orders_per_year  avg_inventory_units  holding_cost  ordering_cost  stockout_cost  total_cost  saving_vs_no_ss
      no safety stock       0.92                0.01             3.21                 3.93     46,042.91      32,128.50       9,385.62   87,557.04             0.00
       uniform 95% SL       0.98                0.00             3.64                 4.77     57,777.78      36,399.00       1,943.59   96,120.38            -0.10
ABC tiered (99/95/90)       0.99                0.00             3.72                 5.01     62,457.47      37,222.50       1,253.85  100,933.82            -0.15
 cost-optimal per SKU       0.92                0.01             3.23                 3.98     46,354.90      32,287.00       8,754.95   87,396.85             0.00

## Cost–service frontier

 service_level    z  safety_stock_units  holding_cost  ordering_cost  expected_shortage_cost  total_cost
         50.00 0.00                0.00     39,979.55      39,608.42                5,656.43   85,244.39
         60.00 0.25               36.01     42,622.47      39,608.42                4,040.94   86,271.82
         70.00 0.52               74.54     45,450.09      39,608.42                2,699.21   87,757.72
         80.00 0.84              119.63     48,759.34      39,608.42                1,582.86   89,950.62
         85.00 1.04              147.32     50,791.62      39,608.42                1,101.59   91,501.62
         90.00 1.28              182.16     53,348.69      39,608.42                  671.26   93,628.37
         92.00 1.41              199.72     54,637.25      39,608.42                  514.12   94,759.79
         95.00 1.64              233.81     57,138.66      39,608.42                  296.23   97,043.31
         97.00 1.88              267.34     59,599.99      39,608.42                  164.73   99,373.13
         98.00 2.05              291.93     61,404.25      39,608.42                  104.12  101,116.79
         99.00 2.33              330.67     64,248.01      39,608.42                   48.05  103,904.47
         99.50 2.58              366.14     66,850.60      39,608.42                   22.41  106,481.42

## Optimal service level vs. shortage cost

 shortage_cost  optimal_service_level  min_total_cost
            10                   50.0        81,850.5
            25                   50.0        85,244.4
            50                   60.0        90,312.8
           100                   80.0        94,699.2
           250                   92.0        99,386.9
           500                   97.0       102,503.0

Cost-optimal service level: **50.0%** (total $85,244/yr).

ABC tiering vs uniform 95%: $100,934 vs $96,120 (-5.0%).
