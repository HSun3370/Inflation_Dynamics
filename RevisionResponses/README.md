```{raw:typst}
#set page(margin: auto)
```

# Revision response audit

Reviewed on 2026-11-14. Q1–Q11 refer to Questions/Remarks; B1–B5 refer to Bigger Picture/Finalization in the supplied letter. The PDF places each response in a footnote beside its relevant discussion, except Q1, Q7, Q8, Q10, Q11, and B2, which are recorded only here. This audit distinguishes completed work, explanations added in this revision, changes from the proposed method, and remaining work. The initial audit used saved estimates. Subsequent restricted Full BEGE estimation is tracked in Q11 and its supporting run records.

| Item | Concern or suggestion | Status | Remaining work |
|---|---|---|---|
| Q1 | BEGE density evaluation and technical paper | Evaluation documented; publication deferred | Separate technical article with Justin and coordinated website release remain future work. |
| Q2 | GARCH search size and parameter ranges | Completed | None for documenting the current search size and ranges. |
| Q3 | EGARCH local optima and standard errors | Search checked; inference qualification remains | No global-optimum proof or independent bootstrap/profile validation of the standard errors. |
| Q4 | GJR coefficients and their own standard errors | Completed | None; boundary estimates retain the stated inference qualification. |
| Q5 | Regime counts, parameter spaces, and preliminary search | Larger searches completed; screening not implemented | The proposed large likelihood-only screen followed by optimization of the top starts is absent. |
| Q6 | Identification of regimes | Completed and scope clarified | Mean-only switching remains an algebraic illustration, not an estimated specification. |
| Q7 | Report volatility rather than variance | Completed | None. |
| Q8 | AR(2) equivalence and the scope of stationarity | Derivation and scope clarified in this revision | No joint inflation–SPF VAR or full switched-system stationarity analysis has been estimated. |
| Q9 | EWMA initialization and explosive variance paths | Investigated; envelope removed by prior decision | The suggested tighter envelope and full joint-process stationarity proof are not supplied. |
| Q10 | BEGE jobs, starts, and computational screening | Start hierarchy clarified; screening not implemented | No likelihood-screening benchmark or screened-versus-random search comparison. |
| Q11 | Full BEGE fit and proposed shape restrictions | Restricted-model results withdrawn from PDF at user request | Correct and validate numerical standard errors before reconsidering these results; archived Full BEGE search reconciliation remains outstanding. |
| B1 | BIC excluding the SPF-only mean | Comparison added in this revision | Whether to exclude the anchored mean from the final model menu remains a project decision. |
| B2 | Add BEGE-EGARCH | Completed | None for adding and estimating this extension. |
| B3 | Extend the data beyond 2022 | Completed beyond the requested endpoint | None for the data extension; the density accuracy benchmark intentionally retains its older sample. |
| B4 | Monthly estimation and SPF timing | Monthly estimates completed; convention differs | Literal division by three and a preceding-quarter SPF regressor for every month have not been estimated as alternatives. |
| B5 | PCE and core-inflation extensions | Deferred; not estimated | PCE, core PCE, and core CPI estimates with SPF predictors remain future work. |


## Responses and supporting evidence


### Q1: BEGE density evaluation and technical paper

The online appendix compares accuracy and speed against numerical integration and documents the stabilized evaluator. Those benchmarks use the older 215-observation sample and are separate from the updated estimation results. The proposed joint technical article and coordinated website release remain future work; the appendix is not that completed publication.

**Evidence:** [BEGE_GARCH/BEGE_Density/BEGE_Density.md](../BEGE_GARCH/BEGE_Density/BEGE_Density.md); [OnlineAppendix/BEGE_Density_Evaluation.md](../OnlineAppendix/BEGE_Density_Evaluation.md).


### Q2: GARCH search size and parameter ranges

Each GARCH-family combination now uses 5,000 optimizer starts, rather than 50: one default start and 4,999 random starts. The parameter-bounds and random-draw table in the quarterly GARCH results gives the ranges, and applies at both frequencies. Starting-value ranges are distinguished from optimization bounds and final selection restrictions.

**Evidence:** [GARCH/README.md](../GARCH/README.md); [GARCH/results_garch_distributions/garch_family_best_models.md](../GARCH/results_garch_distributions/garch_family_best_models.md); [GARCH/estimate_garch_results.py](../GARCH/estimate_garch_results.py).


### Q3: EGARCH local optima and standard errors

The current quarterly AIC-selected EGARCH(2,1), ARX(1,1), Gaussian-mixture fit was reached within 0.01 log-likelihood units by 1,518 of 5,000 starts. It passes convergence, first-order, AR, volatility, and mixture checks; its KKT residual is 0.0020. Its robust standard errors are finite and reported below. This is evidence against an isolated numerical solution, not a proof of global optimality or an independent validation of the standard errors. Profile-likelihood or bootstrap inference has not been supplied.

**Evidence:** [GARCH/results_garch_distributions/garch_family_estimation_summary.csv](../GARCH/results_garch_distributions/garch_family_estimation_summary.csv); [GARCH/results_garch_distributions/garch_family_best_models.md](../GARCH/results_garch_distributions/garch_family_best_models.md).


### Q4: GJR coefficients and their own standard errors

Both positive-shock loadings $\alpha_i$ and negative-shock loadings $\gamma_i$ now have their own standard errors, at both frequencies. The difference $\gamma_i-\alpha_i$ is also retained. Its covariance with $\alpha_i$ is included when calculating $SE(\hat\gamma_i)$: $\operatorname{Var}(\hat\gamma_i)=\operatorname{Var}(\hat\alpha_i)+\operatorname{Var}(\widehat{\gamma_i-\alpha_i})+2\operatorname{Cov}(\hat\alpha_i,\widehat{\gamma_i-\alpha_i})$. Normal inference remains qualified at active bounds.

**Evidence:** [GARCH/results_garch_distributions/garch_family_best_models.md](../GARCH/results_garch_distributions/garch_family_best_models.md); [GARCH/results_garch_distributions_monthly/garch_family_best_models.md](../GARCH/results_garch_distributions_monthly/garch_family_best_models.md); [GARCH/estimate_garch_results.py](../GARCH/estimate_garch_results.py).


### Q5: Regime counts, parameter spaces, and preliminary search

Both $K=2$ and $K=3$ are estimated: 17 specifications at each frequency. All switch innovation variance; two-regime ARX models allow all mean coefficients, the intercept/AR block alone, or no mean coefficients to switch; three-regime ARX models use the latter two cases. Constant contributes a variance-only model at each $K$. The table gives the bounds and OLS-based draws. Each of 5,000 starts receives EM initialization and joint MLE, followed by verification of the selected solution. The suggested likelihood-only screening of, for example, 10,000 draws to retain the best 50 has not been implemented in either regime switching or GARCH.

**Evidence:** [RegimeSwitching/README.md](../RegimeSwitching/README.md); [RegimeSwitching/estimate_regime_switching_models.py](../RegimeSwitching/estimate_regime_switching_models.py); [RegimeSwitching/ols_initialization.py](../RegimeSwitching/ols_initialization.py); [RegimeSwitching/results/regime_switching_run_audit.json](../RegimeSwitching/results/regime_switching_run_audit.json).


### Q6: Identification of regimes

Every specification in the estimated menu switches innovation variance, so increasing $\sigma_i$ orders all reported regimes. The same permutation is applied to coefficients, both transition-matrix axes, and probability paths. Mean-only switching with a shared variance is discussed only to illustrate the algebra; variance ordering would not identify those regimes. Exact variance ties likewise leave the tied labels unidentified.

**Evidence:** [RegimeSwitching/README.md](../RegimeSwitching/README.md); [RegimeSwitching/estimate_regime_switching_models.py](../RegimeSwitching/estimate_regime_switching_models.py).


### Q7: Report volatility rather than variance

The quarterly and monthly regime parameter tables now report $\sigma_i=\sqrt{\sigma_i^2}$, in percentage points per period, rather than the innovation variance. The likelihood still uses $\sigma_i^2$. This is volatility conditional on a regime; when means switch, the predictive variance also includes uncertainty about the regime mean, so $\sigma_i$ is not the unconditional standard deviation of inflation.

**Evidence:** [RegimeSwitching/results/regime_switching_best_model.md](../RegimeSwitching/results/regime_switching_best_model.md); [RegimeSwitching/results_monthly/regime_switching_best_model.md](../RegimeSwitching/results_monthly/regime_switching_best_model.md).


### Q8: AR(2) equivalence and the scope of stationarity

No sign restriction is needed. Let $r_1,r_2$ solve $p(r)=r^2-\rho_1r-\rho_2=0$; these are companion-matrix eigenvalues, reciprocal to roots of the lag polynomial. If $|r_j|<1$, then $p(1)>0$, $p(-1)>0$, and $r_1r_2=-\rho_2<1$, giving the three stated inequalities. Conversely, for conjugate roots, $|r_j|^2=-\rho_2<1$. For real roots, $p(1)>0$ places both on the same side of 1, and $p(-1)>0$ places both on the same side of $-1$; $r_1r_2<1$ rules out both being above 1 or below $-1$. Hence both are inside the unit circle. The report collects mean and volatility restrictions beside the GARCH equations and in the regime-switching and BEGE bounds tables. These checks restrict the inflation AR block conditional on observed SPF; they do not establish joint inflation–SPF stationarity. No SPF law of motion is estimated. Regime-by-regime stability alone also does not prove stability of an arbitrary switching AR system.

**Evidence:** [GARCH/report_model_equations.md](../GARCH/report_model_equations.md); [RegimeSwitching/README.md](../RegimeSwitching/README.md); [BEGE_GARCH/README.md](../BEGE_GARCH/README.md).


### Q9: EWMA initialization and explosive variance paths

I no longer impose the EWMA bound on the implied variance. Estimations with and without the bound gave essentially the same results, and the bound was never active at a converged estimate. With the current BEGE density evaluator, the likelihood does not overflow without it.

**Evidence:** [BEGE_GARCH/BEGE_GARCH.py](../BEGE_GARCH/BEGE_GARCH.py); [BEGE_GARCH/README.md](../BEGE_GARCH/README.md); [summary.md](/scratch/hs3370/Inflation_Dynamics/BEGE_GARCH/ewma_bound_test/analysis/summary.md).


### Q10: BEGE jobs, starts, and computational screening

A seed job is an independently seeded batch. The current protocol is 10 jobs, each with 40 batches of 25 optimizer starts: 10,000 starts per mean/specification/frequency, reduced from the earlier 50-job protocol. Each of the 25 starts draws both mean and volatility parameters anew, then optimizes them jointly; the mean is not held at one draw for all 25 starts. Only the best eligible result of each batch is saved. Parallel jobs distribute the cost, but all starts are optimized. The proposed likelihood screen, retained pure-random fraction, and timing comparison with that alternative remain unimplemented.

**Evidence:** [BEGE_GARCH/bege_batch.py](../BEGE_GARCH/bege_batch.py); [BEGE_GARCH/BEGE_GARCH.py](../BEGE_GARCH/BEGE_GARCH.py); [BEGE_GARCH/README.md](../BEGE_GARCH/README.md).


### Q11: Full BEGE fit and proposed shape restrictions

The symmetric-p/full-n and positive-only-p/full-n searches finished at both frequencies. Their results and comparison were withdrawn from the PDF at the user’s request after the quarterly positive-only-p standard errors were found to differentiate penalty values across the stability boundary. Saved estimates are retained for investigation; the PDF retains the earlier result set.

**Evidence:** [BEGE_GARCH/Full_BEGE/README.md](../BEGE_GARCH/Full_BEGE/README.md); [BEGE_GARCH/ConstantP_BEGE/README.md](../BEGE_GARCH/ConstantP_BEGE/README.md); [BEGE_GARCH/ConstantN_BEGE/README.md](../BEGE_GARCH/ConstantN_BEGE/README.md); [RevisionResponses/archived_full_candidate_check.csv](../RevisionResponses/archived_full_candidate_check.csv); [BEGE_GARCH/restricted_full_comparison.md](../BEGE_GARCH/restricted_full_comparison.md); [BEGE_GARCH/SymmetricP_BEGE/README.md](../BEGE_GARCH/SymmetricP_BEGE/README.md); [BEGE_GARCH/SymmetricP_BEGE/README_monthly.md](../BEGE_GARCH/SymmetricP_BEGE/README_monthly.md); [BEGE_GARCH/PositiveOnlyP_BEGE/README.md](../BEGE_GARCH/PositiveOnlyP_BEGE/README.md); [BEGE_GARCH/PositiveOnlyP_BEGE/README_monthly.md](../BEGE_GARCH/PositiveOnlyP_BEGE/README_monthly.md).


### B1: BIC excluding the SPF-only mean

The anchored mean remains a benchmark. Excluding it, while retaining estimated SPF loadings, the three lowest quarterly BICs are: EGARCH(1,1), ARX(1,1), Student's $t$ (BIC 397.5303); GJR-GARCH(1,1), ARX(1,1), Student's $t$ (BIC 397.8925); EGARCH(1,1), ARX(2,1), Student's $t$ (BIC 399.9321). The monthly winner under this exclusion is GARCH(1,1), ARX(1,1), Student's $t$ (BIC -149.7452). The PDF retains the earlier result set and the four report means. Mean coefficients remain jointly estimated with volatility; this is a sensitivity comparison, not a change to the estimation menu.

**Evidence:** [GARCH/results_garch_distributions/garch_family_best_models.md](../GARCH/results_garch_distributions/garch_family_best_models.md); [ModelComparison/model_ranking_quarterly.csv](../ModelComparison/model_ranking_quarterly.csv); [ModelComparison/model_ranking_monthly.csv](../ModelComparison/model_ranking_monthly.csv); [DataSummary/README.md](../DataSummary/README.md).


### B2: Add BEGE-EGARCH

BEGE-EGARCH is specified and estimated at both frequencies, using coupled log-shape recursions. The report shows their equivalence to EGARCH recursions for the component volatilities. Its quarterly ARX(1,1) AIC is 360.6832. The PDF retains the result set preceding the two restricted Full BEGE searches.

**Evidence:** [BEGE_GARCH/BEGE_EGARCH/SPECIFICATION.md](../BEGE_GARCH/BEGE_EGARCH/SPECIFICATION.md); [BEGE_GARCH/BEGE_EGARCH/README.md](../BEGE_GARCH/BEGE_EGARCH/README.md); [BEGE_GARCH/BEGE_EGARCH/README_monthly.md](../BEGE_GARCH/BEGE_EGARCH/README_monthly.md); [ModelComparison/model_ranking_quarterly.csv](../ModelComparison/model_ranking_quarterly.csv).


### B3: Extend the data beyond 2022

All production estimates now use the October 20, 2026 vintage, extending beyond the requested end of 2025: quarterly 1969Q2–2026Q3 (230 observations) and monthly 1969M2–2026M9 (692 observations). The construction is checked against the original workbook over the overlap. The unpublished October 2025 monthly CPI is interpolated as documented; the quarterly series is unaffected. The older BEGE density accuracy benchmark is separately labeled and is not used for model ranking.

**Evidence:** [DataSummary/data_vintage.json](../DataSummary/data_vintage.json); [DataSummary/README.md](../DataSummary/README.md).


### B4: Monthly estimation and SPF timing

Monthly OLS, GARCH-family, regime-switching, and BEGE results are included. The forecast is repeated within each target quarter, but uses exact geometric de-compounding, $100[(1+q/100)^{1/3}-1]$, rather than $q/3$, matching the supplied workbook. The second SPF regressor is lagged one month: April uses the March and December survey forecasts; May and June use the March forecast in both SPF columns. If the intended second regressor is instead the previous quarterly survey throughout April–June, that alternative has not been estimated. The current convention is preserved, and the proposed monthly comparison is now available for review.

**Evidence:** [DataSummary/README.md](../DataSummary/README.md); [DataSummary/update_inflation_data.py](../DataSummary/update_inflation_data.py); [MeanProcess/OLS_Results_Monthly.md](../MeanProcess/OLS_Results_Monthly.md); [RegimeSwitching/results_monthly/regime_switching_best_model.md](../RegimeSwitching/results_monthly/regime_switching_best_model.md).


### B5: PCE and core-inflation extensions

The current report estimates headline CPI inflation only. PCE, core PCE, and core CPI have not been added to the data construction or estimation results. These remain proposed extensions after evaluating the monthly CPI evidence. The present SPF predictor is a GDP/GNP deflator forecast; using it for another inflation measure would retain that distinction.

**Evidence:** [DataSummary/README.md](../DataSummary/README.md); [DataSummary/data_vintage.json](../DataSummary/data_vintage.json).


## Maintaining these notes

Edit `responses.json`, then run `python scripts/build_report.py`. The builder checks each insertion point and regenerates this checklist. Re-audit numerical statements when results change. The saved archived Full BEGE candidate was evaluated with the current collector on the current quarterly data; its parameters and diagnostics are in `archived_full_candidate_check.csv`. It is evidence for the outstanding search-reconciliation task, not a replacement reported estimate.
