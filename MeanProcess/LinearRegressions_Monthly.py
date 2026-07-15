"""Monthly mean-process estimation: run the four mean specifications on the
monthly dataset and export results to OLS_Results_Monthly.md.

This mirrors the quarterly workflow in LinearRegressions.ipynb (the reporting
cell that writes OLS_Results.md), applied to
DataSummary/Aggregate_CPI_inflation_Monthly.pkl.

Models (pi is monthly CPI inflation, SPF the monthly de-compounded forecast):
  Constant : pi_{t+1} = SPF_t + mu_{t+1}                       (no estimated params)
  ARX(1,1) : pi_{t+1} = c + rho1*pi_t + phi1*SPF_t + mu
  ARX(2,1) : pi_{t+1} = c + rho1*pi_t + rho2*pi_{t-1} + phi1*SPF_t + mu
  ARX(2,2) : pi_{t+1} = c + rho1*pi_t + rho2*pi_{t-1} + phi1*SPF_t + phi2*SPF_{t-1} + mu

For each model we report:
  - parameter estimates with HAC(12) standard errors below in parentheses
  - log-likelihood (under Gaussian residuals)
  - AIC, BIC
  - skewness and excess kurtosis of residuals
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
df = pd.read_pickle(HERE.parent / "DataSummary" / "Aggregate_CPI_inflation_Monthly.pkl")
L = 12  # HAC lags (12 months, matching the 12-lag HAC choice used quarterly)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def diagnostics_constant(resid):
    """Diagnostics for the Constant model (no estimated coefficients).

    Log-likelihood under residuals ~ N(0, sigma^2) with sigma^2 by MLE.
    Only one parameter (sigma^2) is estimated, so k = 1 for AIC/BIC.
    """
    resid = np.asarray(resid)
    n = len(resid)
    sigma2 = np.mean(resid ** 2)
    llf = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2) + 1)
    k = 1
    aic = 2 * k - 2 * llf
    bic = k * np.log(n) - 2 * llf
    return dict(n=n, llf=llf, aic=aic, bic=bic,
                skew=stats.skew(resid, bias=False), kurt=stats.kurtosis(resid, bias=False))


def diagnostics_ols(model):
    """Diagnostics for an OLS model: use statsmodels' built-in llf/aic/bic."""
    resid = np.asarray(model.resid)
    return dict(n=int(model.nobs), llf=model.llf, aic=model.aic, bic=model.bic,
                skew=stats.skew(resid, bias=False), kurt=stats.kurtosis(resid, bias=False))


def render_equation(lhs, params, ses, names, decimals=4):
    r"""Render an equation with standard errors below each estimate."""
    terms = []
    for i, (b, name) in enumerate(zip(params, names)):
        val = f"{abs(b):.{decimals}f}"
        if i == 0:
            sign = "-" if b < 0 else ""
        else:
            sign = "-" if b < 0 else "+"
        if name == "1":  # intercept
            body = val
        else:
            body = f"{val}\\,{name}"
        terms.append(f"{sign}\\,{body}" if sign else body)

    ncol = len(terms)
    col_spec = "c" * ncol
    coef_row = " & ".join(terms)
    se_row = " & ".join(f"({se:.3f})" for se in ses)

    return (
        "$$\n"
        "\\begin{array}{rl}\n"
        f"{lhs} = & \\begin{{array}}{{{col_spec}}}\n"
        f"{coef_row} \\\\\n"
        f"{se_row}\n"
        "\\end{array}\n"
        "\\end{array}\n"
        "$$\n"
    )


def model_to_markdown(label, equation_lhs, params, ses, names, diag):
    eq = render_equation(equation_lhs, params, ses, names)
    diag_lines = (
        f"- Log-likelihood: **{diag['llf']:.3f}**\n"
        f"- AIC: **{diag['aic']:.3f}**, BIC: **{diag['bic']:.3f}**\n"
        f"- Residual skewness: **{diag['skew']:.3f}**, "
        f"excess kurtosis: **{diag['kurt']:.3f}**\n"
    )
    return f"**{label}**:\n\n{eq}\n{diag_lines}\n"


# ---------------------------------------------------------------------------
# Model 0: Constant (no estimated parameters)
# ---------------------------------------------------------------------------
resid0 = (df['Inflation'] - df['SPF']).dropna()
diag0 = diagnostics_constant(resid0)

# ---------------------------------------------------------------------------
# Model 1: ARX(1,1)
# ---------------------------------------------------------------------------
X1 = sm.add_constant(df[['Inflation_lag_1', 'SPF']])
model1 = sm.OLS(df['Inflation'], X1, missing='drop').fit(
    cov_type='HAC', cov_kwds={'maxlags': L}
)
diag1 = diagnostics_ols(model1)

# ---------------------------------------------------------------------------
# Model 2: ARX(2,1)
# ---------------------------------------------------------------------------
X2 = sm.add_constant(df[['Inflation_lag_1', 'Inflation_lag_2', 'SPF']])
model2 = sm.OLS(df['Inflation'], X2, missing='drop').fit(
    cov_type='HAC', cov_kwds={'maxlags': L}
)
diag2 = diagnostics_ols(model2)

# ---------------------------------------------------------------------------
# Model 3: ARX(2,2)
# ---------------------------------------------------------------------------
X3 = sm.add_constant(df[['Inflation_lag_1', 'Inflation_lag_2',
                          'SPF', 'SPF_lag_1']])
model3 = sm.OLS(df['Inflation'], X3, missing='drop').fit(
    cov_type='HAC', cov_kwds={'maxlags': L}
)
diag3 = diagnostics_ols(model3)


# ---------------------------------------------------------------------------
# Build markdown
# ---------------------------------------------------------------------------
header = (
    "```{raw:typst}\n#set page(margin: auto)\n```\n\n"
    "# OLS Results (Monthly)\n\n"
    "Estimation results of four mean models on the monthly effective sample "
    f"({df.index[0]}--{df.index[-1]}, {len(df)} observations) with HAC(12) "
    "standard errors in parentheses below each estimate. (The log-likelihood "
    "is computed under the Gaussian assumption.)\n\n"
)

constant_block = (
    "**Constant**:\n\n"
    "$$\n"
    "\\hat{\\pi}_{t+1} = SPF_t\n"
    "$$\n\n"
    "No estimated coefficients. Residuals $\\mu_{t+1} = \\pi_{t+1} - SPF_t$.\n\n"
    f"- Log-likelihood: **{diag0['llf']:.3f}**\n"
    f"- AIC: **{diag0['aic']:.3f}**, BIC: **{diag0['bic']:.3f}**\n"
    f"- Residual skewness: **{diag0['skew']:.3f}**, "
    f"excess kurtosis: **{diag0['kurt']:.3f}**\n\n"
)

arx11_block = model_to_markdown(
    label="ARX(1,1)",
    equation_lhs=r"\hat{\pi}_{t+1}",
    params=model1.params.values,
    ses=model1.bse.values,
    names=["1", r"\pi_t", r"SPF_t"],
    diag=diag1,
)

arx21_block = model_to_markdown(
    label="ARX(2,1)",
    equation_lhs=r"\hat{\pi}_{t+1}",
    params=model2.params.values,
    ses=model2.bse.values,
    names=["1", r"\pi_t", r"\pi_{t-1}", r"SPF_t"],
    diag=diag2,
)

arx22_block = model_to_markdown(
    label="ARX(2,2)",
    equation_lhs=r"\hat{\pi}_{t+1}",
    params=model3.params.values,
    ses=model3.bse.values,
    names=["1", r"\pi_t", r"\pi_{t-1}", r"SPF_t", r"SPF_{t-1}"],
    diag=diag3,
)

markdown = header + constant_block + arx11_block + arx21_block + arx22_block

out_path = HERE / 'OLS_Results_Monthly.md'
with open(out_path, 'w') as f:
    f.write(markdown)

print(f"Wrote {out_path}")
