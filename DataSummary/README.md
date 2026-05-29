

```{raw:typst}
#set page(margin: auto)
```

# Inflation Summary Statistics
I start with simple statistical summary by plotting inflation and professional forecast over time. 

```{figure} cpi_inflation_quarterly.png
:name: fig:cpi_inflation_quarterly
Quarterly inflation $\pi_t$ and profesional forcast $SPF_{t-1}$. 
```
Notice that inflation was recorded from 1947 but profesional forecast started from 1969.
```{table} Quaterly Inflation Statistics
:align: center

|               |   Inflation |   Forecasted inflation | 
|:--------------|------------:|-----------------------:|
| Date Start    |      1947Q2 |                 1969Q1 |
| Date End      |      2022Q4 |                 2022Q4 |
| #Observations |         303 |                    216 |
```

## Effective Sample

To make log-likelihood values comparable across specifications, I use an effective sample for all mean models and all volatility models.
 
In ARX models, lagged regressors (for example, $SPF_{t-1}$) are not observed at the very beginning of the raw sample. Therefore, I start the estimation sample at the first date where both current forecast inflation ($SPF_t$), and one-lag forecast inflation ($SPF_{t-1}$)  are available.
For the quarterly data, forecast inflation ($SPF_t$) is available from **1969Q1** to **2022Q4** (216 observations), requiring one lag ($SPF_{t-1}$) shifts the usable start to **1969Q2**. So the trimmed estimation window is **1969Q2--2022Q4**, with **215 observations**.
This same trimmed sample is then used for all mean specifications, so the residual series has the same length across models.
 

For GARCH-type recursion, the first few conditional variances are unobserved (for example, $\sigma_0^2$, $\sigma_{-1}^2$, $u_0^2$, $u_{-1}^2$ ).  I initialize these pre-sample variances using the model-implied unconditional variance, for example in GARCH model:

$$
\bar{\sigma}^2 = \frac{\omega}{1-\sum_i \alpha_i - \sum_j \beta_j},
$$

and set

$$
u_0^2 = u_{-1}^2 = \cdots = \sigma_0^2 = \sigma_{-1}^2 = \cdots = \bar{\sigma}^2.
$$


## Effective Sample Summary Statistics
I report the summary statistics for trimmed effect sample. Skewness and Kurtosis are adjusted by sample size.

```{table} Inflation statistics
:align: center
|               |   Inflation ($\pi$) |    $SPF$ |  $\pi - SPF$ |
|:--------------|------------:|-------:|------------:|
| Date Start    |      1969Q2 | 1969Q2 |      1969Q2 |
| Date End      |      2022Q4 | 2022Q4 |      2022Q4 |
| #Observations |         215 |    215 |         215 |
| Mean          |      0.9918 | 0.8320 |      0.1598 |
| Median        |      0.7937 | 0.6296 |      0.1101 |
| Std           |      0.8580 | 0.4969 |      0.6642 |
| Min           |     -3.4170 | 0.1048 |     -4.0117 |
| Max           |      4.1612 | 2.4566 |      2.1729 |
| P5            |     -0.0681 | 0.3482 |     -0.7745 |
| P25           |      0.5328 | 0.4748 |     -0.1669 |
| P75           |      1.2913 | 1.0150 |      0.4617 |
| P95           |      2.5883 | 2.0030 |      1.2581 |
| Skewness      |      0.3550 | 1.3633 |     -0.7131 |
| Kurtosis      |      3.8995 | 1.2053 |      7.3006 |
| AC(1)         |      0.5981 | 0.9717 |      0.2764 |
| AC(2)         |      0.5353 | 0.9473 |      0.1322 |
| AC(4)         |      0.4708 | 0.8929 |      0.0881 |
| AC(12)        |      0.2918 | 0.6949 |      0.0011 |
```

