"""OLS estimates of the six mean processes at both frequencies.

Estimates every mean process by OLS on the current effective samples
(DataSummary/Aggregate_CPI_inflation_{Quarterly,Monthly}.pkl) with HAC(12)
standard errors, writes OLS_Results_Quarterly.md and OLS_Results_Monthly.md,
and refreshes the blocks of README.md between the `% BEGIN auto:` and
`% END auto:` markers.

Models (pi is CPI inflation, SPF the professional forecast, u the residual):
  Constant : pi_{t+1} = SPF_t + u_{t+1}                                  (no estimated params)
  ARX(1,0) : pi_{t+1} = c + rho1*pi_t + u
  ARX(1,1) : pi_{t+1} = c + rho1*pi_t + phi1*SPF_t + u
  ARX(2,0) : pi_{t+1} = c + rho1*pi_t + rho2*pi_{t-1} + u
  ARX(2,1) : pi_{t+1} = c + rho1*pi_t + rho2*pi_{t-1} + phi1*SPF_t + u
  ARX(2,2) : pi_{t+1} = c + rho1*pi_t + rho2*pi_{t-1} + phi1*SPF_t + phi2*SPF_{t-1} + u

For each model the report gives the estimates with HAC(12) standard errors in
parentheses, the Gaussian log likelihood, AIC, BIC, and the residual skewness
and excess kurtosis.

Run: python MeanProcess/mean_process_ols.py
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "DataSummary"
HAC_LAGS = 12  # same HAC truncation at both frequencies

PREAMBLE = "```{raw:typst}\n#set page(margin: auto)\n```\n\n"
GREY = '```{raw:typst}\n#set text(fill: rgb("#808080"))\n#show text: set text(fill: rgb("#808080"))\n```\n\n'

# label -> (regressors, rendered names); the Constant anchor has no estimated coefficients.
MODELS = {
    "ARX(1,0)": (["Inflation_lag_1"], [r"\pi_t"]),
    "ARX(1,1)": (["Inflation_lag_1", "SPF"], [r"\pi_t", r"SPF_t"]),
    "ARX(2,0)": (["Inflation_lag_1", "Inflation_lag_2"], [r"\pi_t", r"\pi_{t-1}"]),
    "ARX(2,1)": (["Inflation_lag_1", "Inflation_lag_2", "SPF"], [r"\pi_t", r"\pi_{t-1}", r"SPF_t"]),
    "ARX(2,2)": (
        ["Inflation_lag_1", "Inflation_lag_2", "SPF", "SPF_lag_1"],
        [r"\pi_t", r"\pi_{t-1}", r"SPF_t", r"SPF_{t-1}"],
    ),
}


def period_label(p: pd.Period) -> str:
    return f"{p.year}Q{p.quarter}" if p.freqstr.startswith("Q") else f"{p.year}M{p.month}"


def residual_moments(resid) -> dict:
    resid = np.asarray(resid, dtype=float)
    return dict(skew=stats.skew(resid, bias=False), kurt=stats.kurtosis(resid, bias=False))


def constant_diagnostics(resid) -> dict:
    """Gaussian likelihood with sigma^2 by MLE; sigma^2 is the only parameter (k = 1)."""
    resid = np.asarray(resid, dtype=float)
    n = len(resid)
    llf = -0.5 * n * (np.log(2 * np.pi) + np.log(np.mean(resid**2)) + 1)
    return dict(llf=llf, aic=2 - 2 * llf, bic=np.log(n) - 2 * llf, **residual_moments(resid))


def render_equation(params, ses, names, decimals=4) -> str:
    """Equation with standard errors below each estimate."""
    terms = []
    for i, (b, name) in enumerate(zip(params, ["1", *names])):
        sign = ("-" if b < 0 else "") if i == 0 else ("-" if b < 0 else "+")
        body = f"{abs(b):.{decimals}f}" if name == "1" else f"{abs(b):.{decimals}f}\\,{name}"
        terms.append(f"{sign}\\,{body}" if sign else body)
    return (
        "$$\n"
        "\\begin{array}{rl}\n"
        f"\\hat{{\\pi}}_{{t+1}} = & \\begin{{array}}{{{'c' * len(terms)}}}\n"
        f"{' & '.join(terms)} \\\\\n"
        f"{' & '.join(f'({se:.3f})' for se in ses)}\n"
        "\\end{array}\n"
        "\\end{array}\n"
        "$$\n"
    )


def diagnostics_lines(d: dict) -> str:
    return (
        f"- Log-likelihood: **{d['llf']:.3f}**\n"
        f"- AIC: **{d['aic']:.3f}**, BIC: **{d['bic']:.3f}**\n"
        f"- Residual skewness: **{d['skew']:.3f}**, excess kurtosis: **{d['kurt']:.3f}**\n"
    )


def estimate(frequency: str) -> tuple[pd.DataFrame, dict, str]:
    """Return the sample, per-model diagnostics, and the markdown model blocks."""
    df = pd.read_pickle(DATA_DIR / f"Aggregate_CPI_inflation_{frequency}.pkl")
    diags = {}
    resid0 = (df["Inflation"] - df["SPF"]).to_numpy()
    diags["Constant"] = constant_diagnostics(resid0)
    blocks = [
        "**Constant**:\n\n$$\n\\hat{\\pi}_{t+1} = SPF_t\n$$\n\n"
        "No estimated coefficients. Residuals $u_{t+1} = \\pi_{t+1} - SPF_t$.\n\n"
        + diagnostics_lines(diags["Constant"])
    ]
    for label, (cols, names) in MODELS.items():
        fit = sm.OLS(df["Inflation"], sm.add_constant(df[cols])).fit(
            cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}
        )
        diags[label] = dict(llf=fit.llf, aic=fit.aic, bic=fit.bic, **residual_moments(fit.resid))
        blocks.append(
            f"**{label}**:\n\n" + render_equation(fit.params.values, fit.bse.values, names)
            + "\n" + diagnostics_lines(diags[label])
        )
    return df, diags, "\n".join(blocks)


def preference_sentence(diags: dict) -> str:
    best_aic = min(diags, key=lambda m: diags[m]["aic"])
    best_bic = min(diags, key=lambda m: diags[m]["bic"])
    if best_aic == best_bic:
        return f"{best_aic} has the lowest AIC and BIC."
    return f"{best_aic} has the lowest AIC and {best_bic} the lowest BIC."


def replace_auto_block(text: str, name: str, body: str) -> str:
    pattern = re.compile(rf"(% BEGIN auto:{name}\n).*?(% END auto:{name})", re.S)
    if not pattern.search(text):
        raise ValueError(f"README.md has no auto:{name} block")
    return pattern.sub(lambda m: m.group(1) + body.rstrip("\n") + "\n" + m.group(2), text)


def main() -> None:
    results = {}
    for frequency in ("Quarterly", "Monthly"):
        df, diags, blocks = estimate(frequency)
        window = f"{period_label(df.index[0])}--{period_label(df.index[-1])}, {len(df)} observations"
        intro = (
            f"Estimation results of the six mean models on the {frequency.lower()} effective sample "
            f"({window}) with HAC(12) standard errors in parentheses below each estimate. "
            "(The log-likelihood is computed under the Gaussian assumption.) "
            + preference_sentence(diags)
        )
        header = PREAMBLE + (GREY if frequency == "Monthly" else "") + f"# OLS Results ({frequency})\n\n"
        out_path = HERE / f"OLS_Results_{frequency}.md"
        out_path.write_text(header + intro + "\n\n" + blocks, encoding="utf-8")
        print(f"Wrote {out_path}")
        results[frequency] = (window, diags, intro, blocks)

    readme = HERE / "README.md"
    text = readme.read_text(encoding="utf-8")
    _, _, q_intro, q_blocks = results["Quarterly"]
    text = replace_auto_block(text, "ols-quarterly", q_intro + "\n\n" + q_blocks)
    m_window, m_diags, _, _ = results["Monthly"]
    text = replace_auto_block(
        text,
        "ols-monthly",
        f"The same six mean models are also estimated on the monthly effective sample ({m_window}) "
        f"with HAC(12) standard errors; the estimates follow in the monthly OLS results. "
        + preference_sentence(m_diags) + "\n",
    )
    readme.write_text(text, encoding="utf-8")
    print(f"Refreshed {readme}")


if __name__ == "__main__":
    main()
