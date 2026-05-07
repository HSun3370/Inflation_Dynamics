

```{raw:typst}
#set page(margin: auto)
```


# DataSummary

`garch1.ipynb` is a reusable data-summary notebook for inflation mean/volatility modeling.


**In the 2023 version, I did a mistake removing two more observations in garch1 when cleaning the data.**



It is crucial to unify how to treat the lag variable, which could lead to different sample size and thus lead the likelihood function un-comparable.

In this exercise, 


## What changed
- Removed hard-coded Windows path logic.
- Added automatic data-file discovery.
- Standardized column parsing from the Excel sheet.
- Added a single `CONFIG` block to control frequency, sample start, HAC lags, and distribution.
- Refactored mean-model and ARCH-family estimation into reusable functions.
- Returns tidy AIC/BIC comparison tables and best model per mean specification.

## How to use
1. Open `garch1.ipynb`.
2. Edit `CONFIG` if needed.
3. Run all cells.

## Required file
- `Aggregate_CPI_inflation_20230513.xls` (same folder as notebook, or discoverable from project root).
