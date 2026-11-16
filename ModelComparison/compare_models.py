"""Rank the admissible estimates of all model classes by AIC and BIC at each frequency.

Collects the reported estimates of every model class on the current effective samples:

* OLS mean processes (Gaussian, constant variance), refitted here so that the innovation
  variance is counted as a parameter, as in the other model classes;
* GARCH family: estimates whose `selection_status` is `passed_checks` (fallback estimates
  that failed the admissibility checks are excluded);
* regime switching: estimates that passed all checks and final-solution verification;
* BEGE: for each available specification and each mean process, the best estimate that
  converged and passed the collection checks (`selection_eligible`).

Writes `model_comparison.md` (ranking summary with the top models by AIC and by BIC and the
best estimate of each class; not part of the PDF report) and `model_ranking_<frequency>.csv`
(all admissible estimates) next to this script. The abstract in `README.md` states the quarterly
AIC and BIC winners; recheck it with this script whenever collected results change.
Run: python ModelComparison/compare_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "MeanProcess"))
from mean_process_ols import MODELS as OLS_MODELS  # noqa: E402

TYPST_PREAMBLE = "```{raw:typst}\n#set page(margin: auto)\n```"
GREY = '```{raw:typst}\n#set text(fill: rgb("#808080"))\n#show text: set text(fill: rgb("#808080"))\n```'
TOP_N = 5
LLF_HIT_TOL = 1e-3  # a start reaches the reported optimum if its log likelihood is within this tolerance
FREQUENCIES = {"quarterly": ("Quarterly", ""), "monthly": ("Monthly", "_monthly")}
BEGE_SPECS = {
    "Constant_BEGE": "Constant BEGE",
    "Symmetric_BEGE": "Symmetric BEGE",
    "BadGood_BEGE": "BadGood BEGE",
    "InflationDeflation_BEGE": "Inflation/Deflation BEGE-GJR",
    "ConstantP_BEGE": "Constant-p Full BEGE",
    "ConstantN_BEGE": "Constant-n Full BEGE",
    "Full_BEGE": "Full BEGE",
    "SymmetricP_BEGE": "Symmetric-p/full-n BEGE",
    "PositiveOnlyP_BEGE": "Positive-only-p/full-n BEGE",
    "BEGE_EGARCH": "BEGE-EGARCH",
}
CLASS_ORDER = ["OLS", "GARCH family", "Regime switching", "BEGE"]


def _true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "y", "1"})


def _period_label(p: pd.Period) -> str:
    return f"{p.year}Q{p.quarter}" if p.freqstr.startswith("Q") else f"{p.year}M{p.month}"


def _criteria(llf: float, k: int, nobs: int) -> dict:
    return {"llf": llf, "k": k, "aic": 2 * k - 2 * llf, "bic": k * np.log(nobs) - 2 * llf}


def ols_rows(df: pd.DataFrame) -> list[dict]:
    """Gaussian OLS likelihoods; k counts the regression coefficients and the innovation variance."""
    nobs = len(df)
    resid = (df["Inflation"] - df["SPF"]).to_numpy()
    llf = -0.5 * nobs * (np.log(2 * np.pi) + np.log(np.mean(resid**2)) + 1)
    rows = [dict(cls="OLS", model="Constant, Gaussian", **_criteria(llf, 1, nobs))]
    for label, (cols, _) in OLS_MODELS.items():
        fit = sm.OLS(df["Inflation"], sm.add_constant(df[cols])).fit()
        rows.append(dict(cls="OLS", model=f"{label}, Gaussian", **_criteria(fit.llf, len(cols) + 2, nobs)))
    return rows


def garch_rows(suffix: str, nobs: int) -> list[dict]:
    g = pd.read_csv(ROOT / f"GARCH/results_garch_distributions{suffix}/garch_family_estimation_summary.csv")
    if (g["nobs"] != nobs).any():
        raise ValueError("GARCH estimates use a different effective sample.")
    g = g[g["selection_status"] == "passed_checks"]
    return [
        dict(cls="GARCH family", model=f"{r.vol_spec}, {r.mean_label}, {r.distribution_label}",
             llf=r.loglikelihood, k=int(r.num_parameters), aic=r.aic, bic=r.bic,
             hits=int(r.best_loglik_hits), starts=int(r.n_starts_requested))
        for r in g.itertuples()
    ]


def _switching_text(r) -> str:
    if r.switching_ar == "Y" and r.switching_spf == "Y":
        return "switching AR, SPF, and variance"
    if r.switching_ar == "Y":
        return "switching AR and variance"
    return "switching variance"


def regime_switching_rows(suffix: str, nobs: int) -> list[dict]:
    out_dir = ROOT / f"RegimeSwitching/results{suffix}"
    res = pd.read_csv(out_dir / "regime_switching_results.csv")
    if (res["nobs"] != nobs).any():
        raise ValueError("Regime-switching estimates use a different effective sample.")
    res = res[_true(res["checks_passed"]) & (res["error_distribution"] == "Normal")]
    attempts = pd.read_csv(out_dir / "regime_switching_attempts.csv.gz")
    attempts = attempts[_true(attempts["checks_passed"])]
    rows = []
    for r in res.itertuples():
        valid = attempts.loc[attempts["model_label"] == r.model_label, "llf"]
        hits = int((valid >= r.llf - LLF_HIT_TOL).sum())
        k = int(round((r.aic + 2 * r.llf) / 2))
        rows.append(dict(cls="Regime switching", model=f"{r.mean_process}, K={r.k_regimes}, {_switching_text(r)}",
                         llf=r.llf, k=k, aic=r.aic, bic=r.bic, hits=hits, starts=len(valid)))
    return rows


def bege_rows(suffix: str) -> list[dict]:
    rows = []
    for folder, title in BEGE_SPECS.items():
        path = ROOT / f"BEGE_GARCH/{folder}/results{suffix}/selection_diagnostics.csv"
        if folder in {"SymmetricP_BEGE", "PositiveOnlyP_BEGE"} and not path.exists():
            print(f"Pending restricted search; not ranked yet: {path}")
            continue
        d = pd.read_csv(path)
        ok = d[_true(d["success"]) & _true(d["selection_eligible"])]
        for mean, grp in ok.groupby("mean_type"):
            r = grp.sort_values("loglik", ascending=False).iloc[0]
            k = int(round((r.AIC + 2 * r.loglik) / 2))
            mean = "Constant" if str(mean).lower() == "constant" else mean
            rows.append(dict(cls="BEGE", model=f"{title}, {mean}", llf=r.loglik, k=k, aic=r.AIC, bic=r.BIC))
    return rows


def ranking(frequency: str) -> tuple[pd.DataFrame, int, str]:
    name, suffix = FREQUENCIES[frequency]
    df = pd.read_pickle(ROOT / f"DataSummary/Aggregate_CPI_inflation_{name}.pkl")
    nobs = len(df)
    sample = f"{_period_label(df.index[0])}--{_period_label(df.index[-1])}"
    table = pd.DataFrame(ols_rows(df) + garch_rows(suffix, nobs) + regime_switching_rows(suffix, nobs) + bege_rows(suffix))
    implied_k = (table["aic"] + 2 * table["llf"]) / 2
    if not np.allclose(implied_k, table["k"], atol=1e-6):
        raise ValueError("AIC is inconsistent with the log likelihood and parameter count.")
    table["aic_rank"] = table["aic"].rank(method="min").astype(int)
    table["bic_rank"] = table["bic"].rank(method="min").astype(int)
    return table.sort_values(["aic", "bic"]).reset_index(drop=True), nobs, sample


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def _top_table(table: pd.DataFrame, criterion: str) -> str:
    other = "bic" if criterion == "aic" else "aic"
    top = table.sort_values([criterion, other]).head(TOP_N)
    lines = [f"| Rank | Model | Class | $k$ | LogLik | AIC | BIC | {other.upper()} rank |",
             "|---:|---|---|---:|---:|---:|---:|---:|"]
    for r in top.itertuples():
        lines.append(f"| {getattr(r, criterion + '_rank')} | {r.model} | {r.cls} | {r.k} | {_fmt(r.llf)} | "
                     f"{_fmt(r.aic)} | {_fmt(r.bic)} | {getattr(r, other + '_rank')} |")
    return "\n".join(lines)


def _class_table(table: pd.DataFrame) -> str:
    lines = ["| Class | Criterion | Best estimate in class | Value | Overall rank |",
             "|---|---|---|---:|---:|"]
    for cls in CLASS_ORDER:
        sub = table[table["cls"] == cls]
        for criterion in ("aic", "bic"):
            best = sub.sort_values(criterion).iloc[0]
            label = cls if criterion == "aic" else ""
            lines.append(f"| {label} | {criterion.upper()} | {best.model} | {_fmt(best[criterion])} | "
                         f"{best[criterion + '_rank']} of {len(table)} |")
    return "\n".join(lines)


def _search_note(table: pd.DataFrame) -> str:
    """How often the search reached the listed regime-switching optima, with GARCH for reference."""
    listed = pd.concat([table.sort_values("aic").head(TOP_N), table.sort_values("bic").head(TOP_N)])
    listed = listed.drop_duplicates("model")
    rs = listed[listed["cls"] == "Regime switching"]
    garch = listed[listed["cls"] == "GARCH family"]
    if rs.empty:
        return ""
    items = "; ".join(f"{r.model}: {int(r.hits):,} of {int(r.starts):,} valid starts" for r in rs.itertuples())
    note = ("Starts that reached the listed regime-switching optimum, with a log likelihood within "
            f"$10^{{-3}}$ of it: {items}.")
    if not garch.empty:
        note += (f" For reference, the listed GARCH-family optima were reached by {int(garch['hits'].min()):,} "
                 f"to {int(garch['hits'].max()):,} of {int(garch['starts'].max()):,} starts.")
    return note


def section(frequency: str, table: pd.DataFrame, nobs: int, sample: str) -> str:
    counts = ", ".join(f"{(table['cls'] == c).sum()} {c}" for c in CLASS_ORDER)
    title = FREQUENCIES[frequency][0]
    parts = [
        f"## {title} Comparison",
        f"The {frequency} comparison uses the effective sample {sample} ($N={nobs}$) and covers "
        f"{len(table)} admissible estimates ({counts}).",
        f"### Top {TOP_N} Models by AIC", _top_table(table, "aic"),
        f"### Top {TOP_N} Models by BIC", _top_table(table, "bic"),
        "### Best Estimate in Each Class", _class_table(table),
    ]
    note = _search_note(table)
    if note:
        parts.append(note)
    return "\n\n".join(parts)


def main() -> None:
    sections = []
    for frequency in FREQUENCIES:
        table, nobs, sample = ranking(frequency)
        table.to_csv(HERE / f"model_ranking_{frequency}.csv", index=False)
        if frequency == "monthly":
            sections.append(GREY)
        sections.append(section(frequency, table, nobs, sample))
    intro = (
        "# Model Comparison\n\n"
        "This section ranks the admissible estimates of all model classes at each frequency. All "
        "models at a given frequency use the same effective sample and the residual definition "
        "$u_t=\\pi_t-\\hat{\\pi}_t$, so their log likelihoods are comparable. Each information "
        "criterion counts every estimated parameter $k$, including variance, distribution, and "
        "transition parameters: $\\mathrm{AIC}=2k-2\\,\\mathrm{LogLik}$ and "
        "$\\mathrm{BIC}=k\\ln N-2\\,\\mathrm{LogLik}$; lower values are better. OLS denotes the "
        "mean processes with constant Gaussian innovations. Only estimates that passed the "
        "admissibility checks of their class enter the ranking: GARCH-family fallback estimates, "
        "unverified regime-switching estimates, and BEGE estimates that failed the collection "
        "checks are excluded. For BEGE, the best eligible estimate of each specification and mean "
        "process enters.\n\n"
        "The Gaussian regime-switching likelihood is unbounded because no variance floor is "
        "imposed, so the regime-switching estimates are verified local maxima rather than global "
        "maxima. The number of starts that reached each listed regime-switching optimum indicates "
        "how reliably the search finds it."
    )
    text = TYPST_PREAMBLE + "\n\n" + intro + "\n\n" + "\n\n".join(sections) + "\n"
    (HERE / "model_comparison.md").write_text(text, encoding="utf-8")
    print(f"Wrote {HERE / 'model_comparison.md'} and model_ranking_<frequency>.csv")


if __name__ == "__main__":
    main()
