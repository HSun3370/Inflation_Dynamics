```{raw:typst}
#set page(margin: auto)
```

# Inflation Summary Statistics
This section provides a statistical summary of quarterly and monthly inflation and the professional forecast over time.

## Data Sources and Construction

The workbook is built from two public sources:

1. **CPIAUCSL** (FRED, St. Louis Fed): Consumer Price Index for All Urban Consumers: All Items in U.S. City Average, Index 1982--1984=100, Monthly, Seasonally Adjusted. <https://fred.stlouisfed.org/series/CPIAUCSL>
2. **Survey of Professional Forecasters** (Philadelphia Fed): median *level* forecasts of the GNP/GDP price deflator (`Median_PGDP_Level.xlsx`). <https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters>

Construction rules:

- **Quarterly price index**: the CPIAUCSL level in the *last month* of each quarter; quarterly inflation $\pi_t$ is its percent change.
- **Monthly inflation**: percent change of monthly CPIAUCSL.
- **SPF expected inflation** for quarter $t$: $(PGDP3/PGDP2 - 1)\times 100$ from the survey taken in quarter $t-1$, i.e. the median one-quarter-ahead deflator forecast over the median nowcast. The first target quarter is 1969Q1 (from the 1968Q4 survey).
- **Monthly SPF expected inflation**: the quarterly value de-compounded geometrically, $\left((1+q/100)^{1/3}-1\right)\times 100$, held constant within the quarter.
- **Inflation shock**: realized inflation minus SPF expected inflation, at both frequencies.
- **Unpublished CPI months** inside the sample are filled by geometric interpolation between the neighbouring published months. This is a data decision; see the box below.

These rules were reverse-engineered from the reference workbook `Aggregate_CPI_inflation_20230513.xls` (the advisor's 2023 snapshot, kept unchanged in this folder) and replicate it exactly; every update re-checks this before writing any file. The gap-filling rule is an addition of mine: the reference period has no unpublished months, so it does not affect that check.

```{raw:typst}
#[
#set text(fill: rgb("#b22222"))
#show text: set text(fill: rgb("#b22222"))
#let importantBlock(body, heading: [Important]) = admonition(body, heading: heading, color: rgb("#b22222"))
```

```{important} Data decision: the October 2025 CPI is interpolated
BLS did not publish a CPI for October 2025. Prices were not collected during the October--November 2025 federal government shutdown, so FRED's CPIAUCSL has no value for 2025M10. **I fill this month with the geometric mean of the two neighbouring published months:**

$$
P_{2025M10} = \sqrt{P_{2025M9}\,P_{2025M11}}.
$$

The published September-to-November price change is therefore split evenly: **monthly inflation in 2025M10 and 2025M11 is constructed, not observed**. The two months have identical inflation rates that compound exactly to the published two-month change. The interpolated level is an estimate, not a BLS figure; the value used in the current vintage is listed under [Current Vintage](#current-vintage).

*Why:* without this fill, monthly inflation would be missing in 2025M10--2025M11 and its lags through 2026M1, so mean specifications with different lag lengths would drop different months and their likelihoods would not be comparable. With it, every monthly specification uses the same months.

*Scope:* only the monthly series changes. The quarterly series is unaffected because the quarterly price index uses the last month of each quarter (December for 2025Q4). The downloaded FRED file in `raw/` keeps the month missing. The interpolated value is written to the workbook and the monthly pickle and recorded in `data_vintage.json` (`interpolated_cpi_months`). The same rule (log-linear interpolation between the nearest published months) is applied automatically to any future unpublished month inside the sample, and the update script reports every month it fills.
```

```{raw:typst}
]
```

## Routine Data Update

All empirical work reads the trimmed estimation datasets `Aggregate_CPI_inflation_Quarterly.pkl` and `Aggregate_CPI_inflation_Monthly.pkl`, which always hold the newest vintage. To refresh them, run from the repository root with the pinned project environment:

```bash
~/.conda/envs/inflation_dynamics/bin/python DataSummary/update_inflation_data.py
~/.conda/envs/inflation_dynamics/bin/python DataSummary/data_summary.py
```

`update_inflation_data.py` downloads both sources, rebuilds the quarterly and monthly series (filling any unpublished CPI month by geometric interpolation and reporting it as a `DATA DECISION` line), and compares them with the 2023 reference workbook on every overlapping date. It stops without writing anything unless SPF expected inflation (quarterly and monthly) matches exactly and CPIAUCSL levels and inflation match exactly before 2019M1. BLS re-estimates the last five years of seasonal factors every February, so seasonally adjusted CPI from 2019M1 onward has been revised since the 2023 snapshot, and those differences are reported rather than rejected. It then writes:

- `raw/CPIAUCSL_YYYYMMDD.csv` and `raw/median_pgdp_level_YYYYMMDD.xlsx`: the source files as downloaded (pass `--cpi-csv` and `--spf-xlsx` to rebuild a vintage offline);
- `Aggregate_CPI_inflation_YYYYMMDD.xlsx`: the workbook, in the reference layout;
- both estimation pickles;
- `data_vintage.json`: vintage tag, sample windows and sizes, unpublished and interpolated CPI months, and the reference-check report.

The script refuses to run outside the pinned `pandas`/`numpy` versions, because pickles written with newer versions cannot be read on the Grid. `data_summary.py` then regenerates the summary-statistics files, both figures, and every table and vintage fact in this README (the blocks between `% BEGIN auto:` and `% END auto:` markers). Earlier workbooks are kept in this folder as a vintage history.

### Current Vintage

% BEGIN auto:vintage

- **Data vintage**: 2026-10-20 (`Aggregate_CPI_inflation_20261020.xlsx`; source files as downloaded in `raw/CPIAUCSL_20261020.csv` and `raw/median_pgdp_level_20261020.xlsx`).
- **Source coverage**: CPIAUCSL 1947M1--2026M9; SPF target quarters 1969Q1--2026Q4.
- **Quarterly effective sample**: **1969Q2--2026Q3**, **230 observations**; no missing values.
- **Monthly effective sample**: **1969M2--2026M9**, **692 months**; no missing values.
- **Interpolated CPIAUCSL months** (not published by BLS; filled by geometric interpolation, see the data decision above): **2025M10 = 324.6537**.
- **Reference check** against `Aggregate_CPI_inflation_20230513.xls`: SPF expected inflation is identical at every overlapping date at both frequencies; CPIAUCSL is identical before 2019M1 and differs from 2019M1 onward by at most 0.29 index points (0.17 percentage points of monthly inflation).

% END auto:vintage

### Comparability of Committed Results

Likelihoods are comparable only within one data vintage. The GARCH-family results (quarterly and monthly) were re-estimated on the current vintage (2026-10-20). The other quarterly estimation results (regime-switching, BEGE, OLS mean processes) were committed before the 2026-07-14 update and were estimated on the 2023-vintage sample **1969Q2--2022Q4** (**215 observations**). The other monthly results (OLS, regime-switching) were estimated on the 2026-07-14 vintage, **1969M2--2026M5** (**688 months**). Those results are not likelihood-comparable with the current vintage until the models are re-estimated.

```{figure} cpi_inflation_quarterly.png
:name: fig:cpi_inflation_quarterly
Quarterly inflation $\pi_t$ and professional forecast $SPF_{t-1}$.
```
Inflation data are available from 1947, but the professional forecast series begins in 1969.

% BEGIN auto:quarterly-coverage

```{table} Quarterly Inflation Statistics
:align: center

|               |   Inflation |   Forecasted inflation |
|:--------------|------------:|-----------------------:|
| Date Start    |      1947Q2 |                 1969Q1 |
| Date End      |      2026Q3 |                 2026Q3 |
| #Observations |         318 |                    231 |
```

% END auto:quarterly-coverage

## Effective Sample

To make log-likelihood values comparable across specifications, I use the same effective sample size for all mean models and all volatility models.[^effective-sample-correction]
 
In ARX models, lagged regressors (for example, $SPF_{t-1}$) are not observed at the very beginning of the raw sample. Therefore, I start the estimation sample at the first date where both current forecast inflation ($SPF_t$), and one-lag forecast inflation ($SPF_{t-1}$)  are available.
For the quarterly data, forecast inflation ($SPF_t$) is available from **1969Q1**, so requiring one lag ($SPF_{t-1}$) shifts the usable start to **1969Q2**. The trimmed estimation window runs from 1969Q2 to the last quarter with complete CPI data; its end date and size for the current vintage are listed under [Current Vintage](#current-vintage).
This same trimmed sample is then used for all mean specifications, so the residual series has the same length across models.
 
[^effective-sample-correction]: Earlier versions of the estimation code contained a sample-trimming mistake and therefore used different numbers of observations across model settings. For example, some GARCH runs used 208 observations, while some BEGE runs used 210 observations. This coding error has been corrected: all estimates reported on the 2023 vintage use the common **1969Q2--2022Q4** effective sample with **215 observations**. As a result, some of those log-likelihood, AIC, and BIC values differ from earlier archived numbers.


For the reported GARCH-family estimates, conditional-variance recursion starts
use the default initialization implemented by the `arch` package. The estimator
therefore no longer iterates the pre-sample recursion to force the initial
variance to equal a parameter-implied unconditional variance.


## Effective Sample Summary Statistics
Summary statistics for the trimmed effective sample are reported below (generated by `data_summary.py`). Skewness and excess kurtosis are bias-corrected for sample size.

% BEGIN auto:quarterly-summary

```{table} Inflation statistics
:align: center

|                 |   Inflation ($\pi$) |   $SPF$ |   $\pi - SPF$ |
|:----------------|--------------------:|--------:|--------------:|
| Date Start      |              1969Q2 |  1969Q2 |        1969Q2 |
| Date End        |              2026Q3 |  2026Q3 |        2026Q3 |
| #Observations   |                 230 |     230 |           230 |
| Mean            |              0.9785 |  0.8212 |        0.1573 |
| Median          |              0.7914 |  0.6310 |        0.1052 |
| Std             |              0.8355 |  0.4830 |        0.6473 |
| Min             |             -3.4170 |  0.1048 |       -4.0117 |
| Max             |              4.1612 |  2.4566 |        2.1729 |
| P5              |             -0.0368 |  0.3559 |       -0.7282 |
| P25             |              0.5259 |  0.4814 |       -0.1620 |
| P75             |              1.2693 |  0.9840 |        0.4579 |
| P95             |              2.5589 |  1.9777 |        1.2361 |
| Skewness        |              0.4069 |  1.4489 |       -0.6934 |
| Excess kurtosis |              4.2058 |  1.5316 |        7.7018 |
| AC(1)           |              0.5893 |  0.9704 |        0.2669 |
| AC(2)           |              0.5330 |  0.9458 |        0.1309 |
| AC(4)           |              0.4598 |  0.8892 |        0.0795 |
| AC(12)          |              0.2689 |  0.6899 |       -0.0010 |
```

% END auto:quarterly-summary

```{raw:typst}
#set text(fill: rgb("#808080"))
#show text: set text(fill: rgb("#808080"))
```

## Monthly Data

The monthly counterpart to the quarterly estimation dataset is `Aggregate_CPI_inflation_Monthly.pkl`, built by `update_inflation_data.py` with the same trimming rule: the sample starts at the first month where both $SPF_t$ and $SPF_{t-1}$ are available. Monthly SPF is available from **1969M1**, so the trimmed monthly estimation window starts in **1969M2** and runs to the last month of CPI data (see [Current Vintage](#current-vintage)). The dataset carries the same columns as the quarterly pickle (`Inflation`, `Inflation_lag_1`, `Inflation_lag_2`, `SPF`, `SPF_lag_1`, `SPF_shock`); note that `SPF` and `SPF_lag_1` are the geometrically de-compounded quarterly forecasts and are therefore constant within each quarter.

The monthly effective sample has no missing values because the unpublished October 2025 CPI is interpolated (see the data decision under [Data Sources and Construction](#data-sources-and-construction)). **Monthly inflation in 2025M10 and 2025M11 is therefore constructed rather than observed**, and these two months enter the lagged-inflation regressors through 2026M1.

```{figure} cpi_inflation_monthly.png
:name: fig:cpi_inflation_monthly
Monthly inflation $\pi_t$ and professional forecast $SPF_{t-1}$.
```

% BEGIN auto:monthly-coverage

```{table} Monthly Inflation Statistics
:align: center

|               |   Inflation |   Forecasted inflation |
|:--------------|------------:|-----------------------:|
| Date Start    |     1947-02 |                1969-01 |
| Date End      |     2026-09 |                2026-09 |
| #Observations |         956 |                    693 |
```

% END auto:monthly-coverage

Summary statistics for the trimmed monthly effective sample (generated by `data_summary.py`):

% BEGIN auto:monthly-summary

```{table} Monthly inflation statistics
:align: center

|                 |   Inflation ($\pi$) |   $SPF$ |   $\pi - SPF$ |
|:----------------|--------------------:|--------:|--------------:|
| Date Start      |             1969-02 | 1969-02 |       1969-02 |
| Date End        |             2026-09 | 2026-09 |       2026-09 |
| #Observations   |                 692 |     692 |           692 |
| Mean            |              0.3252 |  0.2727 |        0.0524 |
| Median          |              0.2751 |  0.2103 |        0.0415 |
| Std             |              0.3248 |  0.1593 |        0.2744 |
| Min             |             -1.7705 |  0.0349 |       -1.9684 |
| Max             |              1.8100 |  0.8123 |        1.4582 |
| P5              |             -0.0792 |  0.1160 |       -0.3415 |
| P25             |              0.1616 |  0.1600 |       -0.0783 |
| P75             |              0.4776 |  0.3271 |        0.1893 |
| P95             |              0.9516 |  0.6584 |        0.4775 |
| Skewness        |              0.1660 |  1.4381 |       -0.4188 |
| Excess kurtosis |              4.1414 |  1.4823 |        6.6091 |
| AC(1)           |              0.6223 |  0.9901 |        0.4590 |
| AC(2)           |              0.4343 |  0.9803 |        0.1861 |
| AC(4)           |              0.3995 |  0.9622 |        0.1120 |
| AC(12)          |              0.2927 |  0.8895 |       -0.0076 |
```

% END auto:monthly-summary
