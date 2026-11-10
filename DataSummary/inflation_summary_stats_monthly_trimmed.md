```{raw:typst}
#set page(margin: auto)
```

```{raw:typst}
#set text(fill: rgb("#808080"))
#show text: set text(fill: rgb("#808080"))
```

# Inflation Summary Statistics (Monthly, Trimmed)

Data vintage 2026-10-20 (`Aggregate_CPI_inflation_20261020.xlsx`).

```{important} Data decision: interpolated CPI
BLS did not publish CPIAUCSL for 2025M10. **The missing level is filled by geometric interpolation between the neighbouring published months** (2025M10 = 324.6537), so the published price change across the gap is split evenly. It is an estimate, not BLS data; see the data decision in `DataSummary/README.md`.
```

## Full workbook sample

|               |   Inflation |   Forecasted inflation |
|:--------------|------------:|-----------------------:|
| Date Start    |     1947-02 |                1969-01 |
| Date End      |     2026-09 |                2026-09 |
| #Observations |         956 |                    693 |

## Variables: Inflation, SPF, SPF_shock

|                 |   Inflation |     SPF |   SPF_shock |
|:----------------|------------:|--------:|------------:|
| Date Start      |     1969-02 | 1969-02 |     1969-02 |
| Date End        |     2026-09 | 2026-09 |     2026-09 |
| #Observations   |         692 |     692 |         692 |
| Mean            |      0.3252 |  0.2727 |      0.0524 |
| Median          |      0.2751 |  0.2103 |      0.0415 |
| Std             |      0.3248 |  0.1593 |      0.2744 |
| Min             |     -1.7705 |  0.0349 |     -1.9684 |
| Max             |      1.8100 |  0.8123 |      1.4582 |
| P5              |     -0.0792 |  0.1160 |     -0.3415 |
| P25             |      0.1616 |  0.1600 |     -0.0783 |
| P75             |      0.4776 |  0.3271 |      0.1893 |
| P95             |      0.9516 |  0.6584 |      0.4775 |
| Skewness        |      0.1660 |  1.4381 |     -0.4188 |
| Excess kurtosis |      4.1414 |  1.4823 |      6.6091 |
| AC(1)           |      0.6223 |  0.9901 |      0.4590 |
| AC(2)           |      0.4343 |  0.9803 |      0.1861 |
| AC(4)           |      0.3995 |  0.9622 |      0.1120 |
| AC(12)          |      0.2927 |  0.8895 |     -0.0076 |
