"""Write the QuantMFR comparison after the complete corrected Grid search."""
from pathlib import Path
import pandas as pd
import numpy as np
import estimate_regime_switching_models as rs


def table(rows):
    return rs._markdown_table(pd.DataFrame(rows))


def write_report():
    out = rs.RS_DIR / 'audit_20261112'
    audit = pd.read_csv(out / 'likelihood_audit.csv')
    updated = pd.read_csv(out / 'updated' / 'likelihood_audit.csv')
    if len(updated) != 34 or not updated.full_search_coverage.all():
        raise ValueError('The updated audit must cover all 34 complete searches')
    if updated.saved_minus_recomputed.abs().max() > 1e-4:
        raise ValueError('Updated saved likelihoods are not reproducible')
    sections = [rs.TYPST_PREAMBLE, '', '# Regime-switching estimation audit and re-estimation', '',
        'Audit started 2026-11-12; completed 2026-11-13. Both frequencies use the 2026-10-20 data vintage.', '',
        '## What the comparison establishes', '',
        'The original inflation likelihood and filtering implementation are correct at the '
        'saved parameter vectors. The independent log-domain Hamilton filter and the original '
        'QuantMFR filter reproduce them to numerical precision. The correction made here concerns '
        'the EM initialization of models with shared mean coefficients. Final estimation still '
        'maximizes the same observed-data likelihood. Neither multiple starts nor local verification '
        'proves that a finite global maximum exists.', '',
        'The comparison uses the authenticated original repository, branch `QuantMFR`, commit '
        '`f207800b7927f175839ebb74bc6c9187741afc09`. The checkout is at '
        '`/user/hs3370/TenuousBeliefs`. The relevant sources are '
        '[MLE.py](https://github.com/lphansen/TenuousBeliefs/blob/f207800b7927f175839ebb74bc6c9187741afc09/MLE.py#L1428), '
        '[EM_MSVAR.ipynb](https://github.com/lphansen/TenuousBeliefs/blob/f207800b7927f175839ebb74bc6c9187741afc09/EM_MSVAR.ipynb), '
        'and [MS_VAR.ipynb](https://github.com/lphansen/TenuousBeliefs/blob/f207800b7927f175839ebb74bc6c9187741afc09/MS_VAR.ipynb). '
        'Their access permissions also apply to these links.', '',
        '## Method comparison', '',
        table([
            {'Component': 'Model', 'QuantMFR': 'Multivariate VAR; all regression coefficients and covariance matrices switch',
             'Inflation implementation': 'Univariate Gaussian ARX; preserves shared AR/SPF blocks and anchored Constant mean'},
            {'Component': 'Initial state probabilities', 'QuantMFR': 'Stationary distribution of P (power iteration)',
             'Inflation implementation': 'Stationary distribution of P at each likelihood evaluation'},
            {'Component': 'Likelihood', 'QuantMFR': 'Hamilton prediction/update with Gaussian densities',
             'Inflation implementation': 'Same recursion; numerically stable log-domain evaluation'},
            {'Component': 'EM regression step', 'QuantMFR': 'Separate probability-weighted regressions; ridge=1e-10',
             'Inflation implementation': 'Joint probability/variance-weighted regression with exact sharing restrictions; no ridge'},
            {'Component': 'Optimization', 'QuantMFR': 'EM-only random search; EM notebook uses 200 starts, 500 iterations, tolerance 1e-6',
             'Inflation implementation': '5,000 starts; up to 1,000 EM iterations then joint BFGS/L-BFGS, up to 1,500 iterations'},
            {'Component': 'Selection', 'QuantMFR': 'Largest finite EM likelihood; no returned convergence flag or final stationarity rejection',
             'Inflation implementation': 'Optimizer success and existing parameter/AR checks, followed by 20 perturbation refits'},
            {'Component': 'Identification', 'QuantMFR': 'EM labels arbitrary; Gibbs plots optionally align by stationary probabilities',
             'Inflation implementation': 'All selected parameters, transitions, and probability paths ordered by innovation variance'},
            {'Component': 'Gibbs sampler', 'QuantMFR': 'Posterior state/parameter draws; a different inferential procedure',
             'Inflation implementation': 'Not substituted for the requested Bekaert-style likelihood estimation'},
        ]), '',
        'QuantMFR is a comparison implementation, not a correctness certificate. Its EM '
        'transition-count update omits the dependence of the initial stationary distribution '
        'on P, and its search can retain an iteration-limit result without checking convergence. '
        'The inflation estimator uses this type of transition update only for initialization; '
        'the final numerical optimization evaluates the full stationary-initialized likelihood. '
        'The source also exponentiates Gaussian log densities directly, which can underflow '
        'for extreme candidate parameters. No such arithmetic is copied into production.', '',
        '## Correction to the previous initializer', '',
        'In statsmodels 0.14.2, `_em_exog` first estimates non-switching coefficients by '
        'pooled OLS, subtracts their fitted contribution, then estimates switching coefficients '
        'separately. When coefficients are shared and innovation variances differ, this generally '
        'does not maximize the Gaussian mean-block EM objective. The corrected update solves', '',
        '$$\n\\min_{\\{\\beta_k\\}}\\sum_{t=1}^T\\sum_{k=1}^K '
        '\\frac{\\gamma_{k,t}}{\\sigma_{k,\\mathrm{old}}^2}'
        '(y_t-x_t^\\prime\\beta_k)^2,\\qquad '
        '\\gamma_{k,t}=\\Pr(s_t=k\\mid Y_T;\\theta_{\\mathrm{old}}),\n$$', '',
        'subject to the model\'s equality restrictions on shared coefficients. With all '
        'coefficients switching, the constant variance weight cancels inside each separate '
        'regression, recovering the usual regime-specific weighted least-squares update. '
        'The subsequent variance update uses weighted squared residuals. The anchored Constant '
        'model has no mean coefficients to update. Tests check the weighted first-order '
        'conditions, coefficient equality restrictions, and integration with the actual smoother.', '',
        'The correction changes the starting point supplied to the final optimizer, not the '
        'likelihood or economic model. Thus a different selected result reflects the search '
        'path or verification outcome. It is not evidence that the previous likelihood formula '
        'was wrong. The optional Student-t initializer is unchanged.', '',
        '## Audit of the previous search', '',
        'All 34 specifications have exactly 5,000 distinct initial attempts, covering attempt '
        'numbers 1–5,000. Each frequency uses the full effective sample without additional '
        'lag trimming: quarterly 1969Q2–2026Q3 (230 observations), monthly 1969M2–2026M9 '
        '(692 observations). The monthly interpolation convention remains unchanged. '
        'The stored anchored residual is exactly inflation minus SPF.', '',
        'For the source comparison, only QuantMFR\'s data-preparation function is replaced '
        'with the already constructed inflation response and design matrix. Its Gaussian '
        'density, forward filter, backward smoother, and likelihood functions execute unchanged. '
        'Transition subtraction roundoff below zero is removed at machine precision before '
        'passing the matrix to the reference code; no economic cutoff is imposed.', '',
        table([{
            'Frequency': frequency, 'Models': len(group),
            'Max independent vs statsmodels LogLik gap': f"{group.independent_minus_statsmodels.abs().max():.2e}",
            'Max QuantMFR vs statsmodels LogLik gap': f"{group.quantmfr_minus_statsmodels.abs().max():.2e}",
            'Max filtered probability gap': f"{group.max_filtered_probability_difference.max():.2e}",
            'Max smoothed probability gap': f"{group.max_smoothed_probability_difference.max():.2e}",
        } for frequency, group in audit.groupby('frequency')]), '',
        'Two rejected quarterly fits have saved likelihoods that are not reproduced by their '
        'saved parameters. Their extremely small variances make the likelihood sensitive to '
        'floating-point changes in fitted means. The independently recomputed likelihoods '
        'agree with statsmodels at those saved parameter vectors as well. These are numerical '
        'degeneracies, not evidence for a different Hamilton likelihood.', '',
        '## Fresh estimation results', '',
        'The corrected search uses methodology `'+rs.METHODOLOGY_VERSION+'`: '
        '170 Grid blocks × 1,000 starts = 170,000 starts across 34 specifications. Each '
        'attempt keeps the previous deterministic seed and starting-draw rule. Only the '
        'selected provisional optimum of each specification is verified after collection; '
        'every refit in its final batch must converge, pass the model checks, and return '
        'within 0.0001 log-likelihood units. Verification refits are additional to the '
        '5,000 initial starts. A failed selected optimum is not replaced by a lower optimum.', '']
    winners, comparison, counts = [], [], []
    for frequency, folder in [('Quarterly', 'results'), ('Monthly', 'results_monthly')]:
        old = pd.read_csv(out / 'previous' / folder / 'regime_switching_results.csv')
        new = pd.read_csv(rs.RS_DIR / folder / 'regime_switching_results.csv')
        if len(new) != 17 or not (new.fit_attempts == 5000).all() or not (new.search_methodology == rs.METHODOLOGY_VERSION).all():
            raise ValueError(f'{frequency}: complete corrected search has not yet been collected')
        valid = new[new.checks_passed & new.stability_verified]
        refits = pd.read_csv(rs.RS_DIR / folder / 'regime_switching_stability_attempts.csv')
        counts.append({'Frequency': frequency, 'Previous verified': int(old.checks_passed.sum()),
                       'New verified': len(valid), 'New unverified': len(new)-len(valid),
                       'Additional verification refits': len(refits)})
        for criterion in ['aic', 'bic']:
            if valid.empty:
                continue
            row = valid.loc[valid[criterion].idxmin()]
            winners.append({'Frequency': frequency, 'Criterion': criterion.upper(), 'Mean': row.mean_process,
                'K': int(row.k_regimes), 'Sw.AR': row.switching_ar, 'Sw.SPF': row.switching_spf,
                'N': int(row.nobs), 'Sample': f'{row.sample_start}–{row.sample_end}',
                'LogLik': f'{row.llf:.4f}', 'AIC': f'{row.aic:.4f}', 'BIC': f'{row.bic:.4f}'})
        joined = old.merge(new, on='model_label', suffixes=('_old', '_new'))
        entries = []
        for _, r in joined.iterrows():
            change = r.llf_new-r.llf_old
            if abs(change) < 0.00005:
                change = 0.0
            entries.append({'Model': r.model_label, 'Previous LogLik': f'{r.llf_old:.4f}',
                'New LogLik': f'{r.llf_new:.4f}', 'Change': f'{change:.4f}',
                'Previous status': r.stability_status_old, 'New status': r.stability_status_new,
                'New min variance': f'{r.min_sigma2_new:.2e}'})
        comparison.extend(['', f'### {frequency}: every specification', '', table(entries)])
        joined.to_csv(out / f'comparison_{frequency.lower()}.csv', index=False)
    sections += [table(counts), '', table(winners), '',
        'AIC and BIC select among verified fits only and can favor different specifications. '
        'Likelihoods are compared within frequency; quarterly and monthly likelihoods are '
        'not ranked against each other. The full parameter tables, transition matrices, '
        'conditional-variance quantiles, and regime plots are in the '
        '[quarterly best-model report](../results/regime_switching_best_model.md) and '
        '[monthly best-model report](../results_monthly/regime_switching_best_model.md).', '',
        'The former quarterly AIC winner, three-regime ARX(1,1) with switching AR '
        'coefficients, had a verified local log likelihood of −167.3355. The new search '
        'selected a higher candidate at −158.3755, with minimum variance 1.47e-7, '
        'which failed verification. Similarly, the former monthly AIC winner, '
        'three-regime ARX(2,1) with switching AR coefficients, moved from a verified '
        'local log likelihood of 116.7792 to an unverified candidate at 134.3093 '
        '(minimum variance 3.47e-7). Under the requested rule against falling back '
        'to a lower optimum, these specifications are excluded from the new ranking. '
        'This explains the change in AIC winners; it does not invalidate the older '
        'fits as local optima.', '',
        'All 34 newly saved likelihoods reproduce exactly from their saved parameter '
        'values. The updated independent and QuantMFR comparisons are:', '',
        table([{'Frequency': frequency,
                'Max independent LogLik gap': f'{group.independent_minus_statsmodels.abs().max():.2e}',
                'Max QuantMFR LogLik gap': f'{group.quantmfr_minus_statsmodels.abs().max():.2e}',
                'Max saved vs recomputed LogLik gap': f'{group.saved_minus_recomputed.abs().max():.2e}'}
               for frequency, group in updated.groupby('frequency')])]
    sections += comparison
    sections += ['', '## Interpretation and reproducibility', '',
        'In flexible switching-mean Gaussian specifications, a regime can fit a few '
        'observations almost exactly while its variance tends to zero. Consequently the '
        'unrestricted likelihood is unbounded. Keeping the user-approved removal of the '
        'variance floor means these models need not possess a finite global MLE. A '
        'successful local perturbation check establishes only numerical reproducibility '
        'around the selected fit; it does not make the likelihood bounded. Conversely, '
        'a failed check need not mean variance collapse: optimizer failures or convergence '
        'to another local solution can also cause rejection. The saved verification CSVs '
        'distinguish these cases.', '',
        'Regime-specific AR checks are the existing inflation application\'s restrictions. '
        'They should not be interpreted as a separate proof of stationarity for arbitrary '
        'Markov-switching AR dynamics.', '',
        'The previous summaries, parameters, verification attempts, and notes are archived '
        'under `previous/`. `likelihood_audit.csv` records the previous-fit cross-checks; '
        '`provenance.json` records their input hashes and the reference commit. The '
        '`updated/` directory contains the same audit of the fresh estimates. Fresh '
        'search blocks and scheduler logs are at '
        '`/scratch/hs3370/Inflation_Dynamics/RegimeSwitching/20261112_quantmfr_joint_em_01/`. '
        'Collected production output includes every initial attempt in '
        '`regime_switching_attempts.csv.gz` and each final refit separately in '
        '`regime_switching_stability_attempts.csv`.', '',
        '```bash',
        '~/.conda/envs/inflation_dynamics/bin/python RegimeSwitching/audit_estimation.py \\\n  --reference-repo /user/hs3370/TenuousBeliefs \\\n  --output-dir RegimeSwitching/audit_20261112/updated',
        '~/.conda/envs/inflation_dynamics/bin/python RegimeSwitching/write_audit_report.py',
        '```', '']
    (out / 'report.md').write_text('\n'.join(sections))
    print(out / 'report.md')


if __name__ == '__main__':
    write_report()
