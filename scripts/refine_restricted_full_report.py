"""Validate completed restricted searches, compare them with Full BEGE, rebuild PDF.

Run after all four dependent collectors. Never publish a partial search. The
comparison uses the four report means, while research CSVs retain all six.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from BEGE_GARCH import bege_batch as batch

PREAMBLE = '```{raw:typst}\n#set page(margin: auto)\n```\n\n'
MEANS = ['constant', 'ARX(1,1)', 'ARX(2,1)', 'ARX(2,2)']
SPECS = {'SymmetricP_BEGE': ('symmetric_p', 'Symmetric-p'),
         'PositiveOnlyP_BEGE': ('positive_p', 'Positive-only-p')}
FREQUENCIES = {'quarterly': ('', 230, '1969Q2--2026Q3'),
               'monthly': ('_monthly', 692, '1969M2--2026M9')}
RUN_DIR = ROOT / 'RevisionResponses/restricted_full_run_20261114'
COMPARISON = ROOT / 'BEGE_GARCH/restricted_full_comparison.md'


def read_results(folder, suffix):
    directory = ROOT / 'BEGE_GARCH' / folder / ('results' + suffix)
    raw = pd.read_csv(directory / 'all_estimations.csv')
    checks = pd.read_csv(directory / 'selection_diagnostics.csv')
    keys = ['seed', 'draw', 'mean_type']
    # Take eligibility only from the collector; join parameters by unique draw key.
    valid = checks.loc[batch.selection_eligible_mask(checks), keys]
    eligible = raw.merge(valid, on=keys, validate='one_to_one')
    return raw, checks, eligible


def validate_completed_searches():
    audit = []
    manifest = json.loads((RUN_DIR/'manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        if name.startswith('DataSummary/'):
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'{name}: data changed since submission.')
    expected_keys = {(seed, draw, mean) for seed in range(1, 11)
                     for draw in range(1, 41) for mean in batch.MEAN_TYPES}
    for frequency, (suffix, nobs, period) in FREQUENCIES.items():
        os.environ['BEGE_FREQUENCY'] = frequency
        effective = batch.load_effective_sample(ROOT)
        if len(effective) != nobs:
            raise ValueError(f'{frequency}: data vintage changed; do not mix samples.')
        for folder, (family, _) in SPECS.items():
            raw, checks, eligible = read_results(folder, suffix)
            keys = list(zip(raw.seed, raw.draw, raw.mean_type))
            if len(keys) != len(expected_keys) or set(keys) != expected_keys:
                raise ValueError(f'{folder}/{frequency}: incomplete search.')
            if set(eligible.mean_type) != set(batch.MEAN_TYPES):
                raise ValueError(f'{folder}/{frequency}: a mean has no admissible estimate.')
            expected_k = raw.mean_type.map(lambda mean: len(batch.MEAN_PARAM_NAMES[mean]) + 9)
            finite = np.isfinite(raw.loglik)
            if not np.allclose(raw.loc[finite, 'AIC'], 2*expected_k[finite]-2*raw.loc[finite, 'loglik']):
                raise ValueError(f'{folder}/{frequency}: wrong AIC parameter count.')
            if not np.allclose(raw.loc[finite, 'BIC'], np.log(nobs)*expected_k[finite]-2*raw.loc[finite, 'loglik']):
                raise ValueError(f'{folder}/{frequency}: wrong BIC parameter count.')
            directory = ROOT/'BEGE_GARCH'/folder/('results'+suffix)
            reported = pd.read_csv(directory/'best_loglik_with_se.csv')
            if len(reported) != 1 or reported.iloc[0].mean_type not in MEANS:
                raise ValueError(f'{folder}/{frequency}: missing reported best estimate.')
            row = reported.iloc[0]
            best_ll = eligible.loc[eligible.mean_type.isin(MEANS), 'loglik'].max()
            if abs(best_ll-row.loglik) > 1e-6:
                raise ValueError(f'{folder}/{frequency}: reported row is not likelihood-best.')
            names = batch.MEAN_PARAM_NAMES[row.mean_type] + batch.model_param_names_for_family(family)
            if not all(np.isfinite(row['se_'+name]) and row['se_'+name] >= 0 for name in names):
                raise ValueError(f'{folder}/{frequency}: incomplete reported standard errors.')
            specs = {s['mean_type']: s for s in batch.build_model_specs(effective, True)}
            evaluated = batch._selection_metrics_for_row(row, model_family=family, specs_by_mean=specs)
            if abs(evaluated['corrected_loglik']-row.loglik) > 1e-6:
                raise ValueError(f'{folder}/{frequency}: saved likelihood is not reproducible.')
            audit.append(dict(frequency=frequency, specification=folder, saved_results=len(raw),
                              admissible_results=len(eligible), random_starts_per_mean=10000,
                              N=nobs, sample=period, reported_mean=row.mean_type,
                              loglik=row.loglik, AIC=row.AIC, BIC=row.BIC))
    return audit


def comparison_rows():
    rows = []
    for frequency, (suffix, nobs, period) in FREQUENCIES.items():
        _, _, full = read_results('Full_BEGE', suffix)
        for folder, (_, label) in SPECS.items():
            _, _, restricted = read_results(folder, suffix)
            for mean in MEANS:
                f = full.loc[full.mean_type.eq(mean)].sort_values('loglik').iloc[-1]
                r = restricted.loc[restricted.mean_type.eq(mean)].sort_values('loglik').iloc[-1]
                rows.append(dict(frequency=frequency, mean=mean, specification=folder,
                                 label=label, N=nobs, sample=period, full_loglik=f.loglik,
                                 loglik=r.loglik, AIC=r.AIC, BIC=r.BIC,
                                 delta_loglik=r.loglik-f.loglik,
                                 delta_AIC=r.AIC-f.AIC, delta_BIC=r.BIC-f.BIC,
                                 seed=int(r.seed), draw=int(r.draw)))
    return pd.DataFrame(rows)


def comparison_text(table):
    parts = [PREAMBLE + '# Restricted Full BEGE: comparison',
        'Both restrictions retain the full $n_t$ recursion and estimate nine volatility '
        'parameters, one fewer than Full BEGE. Every comparison below holds the mean process, '
        'sample, density evaluator, and shape initialization convention fixed. Each restricted '
        'search uses 10,000 random starts per mean and frequency.',
        r'$\Delta\ell$, $\Delta\mathrm{AIC}$ and $\Delta\mathrm{BIC}$ denote the restricted-model '
        r'value minus the currently reported Full BEGE value for the same mean. Negative '
        r'criterion differences favor the restriction. Specifically, '
        r'$\Delta\mathrm{AIC}=-2\Delta\ell-2$ and '
        r'$\Delta\mathrm{BIC}=-2\Delta\ell-\log N$. '
        '“Symmetric-p” and “Positive-only-p” in the table both retain full n dynamics.']
    for frequency, (_, nobs, period) in FREQUENCIES.items():
        sub = table.loc[table.frequency.eq(frequency)]
        if frequency == 'monthly':
            parts.append('```{raw:typst}\n#set text(fill: rgb("#808080"))\n#show text: set text(fill: rgb("#808080"))\n```')
        parts.extend([f'## {frequency.title()}', f'Sample: {period}; $N={nobs}$.',
            r'| Mean | Restriction | LogLik | $\Delta\ell$ | $\Delta$AIC | $\Delta$BIC |',
            '|---|---|---:|---:|---:|---:|'])
        for r in sub.itertuples():
            mean = 'Constant' if r.mean == 'constant' else r.mean
            parts.append(f'| {mean} | {r.label} | {r.loglik:.4f} | {r.delta_loglik:.4f} | {r.delta_AIC:.4f} | {r.delta_BIC:.4f} |')
        parts.append('')
        for folder, (_, label) in SPECS.items():
            selected = sub.loc[sub.specification.eq(folder)]
            a = int((selected.delta_AIC < 0).sum()); b = int((selected.delta_BIC < 0).sum())
            winner = selected.loc[selected.AIC.idxmin()]
            parts.append(f'**{label}/full-n:** lower AIC than Full BEGE in {a} of the four means '
                         f'and lower BIC in {b}. Its lowest AIC is {winner.AIC:.4f} '
                         f'with {"Constant" if winner["mean"] == "constant" else winner["mean"]}.')
        if (sub.delta_loglik > 1e-4).any():
            parts.append('Some restricted fits exceed the saved Full BEGE likelihood. Because '
                'the restricted parameter vectors are feasible for Full BEGE, this identifies '
                'an incomplete unrestricted numerical search. These differences compare saved '
                'fits and must not be interpreted as likelihood-ratio tests at global maxima.')
        if frequency == 'quarterly':
            parts.append('An archived, admissible Full BEGE ARX(2,2) fit has log likelihood '
                '−168.8770, compared with −168.9699 in the current reported search. Using '
                'that archived benchmark increases the ARX(2,2) criterion differences above '
                'by 0.1858. Reconciliation of the unrestricted archived searches remains separate '
                'from these newly completed restricted searches.')
    parts.append('The model-specific pages select the likelihood-best estimate across the four '
        'reported means and provide parameter standard errors and path quantiles. The comparison '
        'above also shows criterion differences within each mean; likelihood-based selection '
        'across means need not coincide with AIC- or BIC-based selection. Boundary coefficients '
        'and rough standard errors do not support ordinary Wald significance claims.')
    # Keep table rows contiguous, while separating prose paragraphs.
    text = '\n\n'.join(parts)
    text = re.sub(r'(?<=\|)\n\n(?=\|)', '\n', text)
    return text + '\n'


def update_responses(table):
    path = ROOT/'RevisionResponses/responses.json'
    notes = json.loads(path.read_text())
    counts = []
    for frequency in FREQUENCIES:
        sub = table.loc[table.frequency.eq(frequency)]
        for folder, (_, label) in SPECS.items():
            selected = sub.loc[sub.specification.eq(folder)]
            counts.append(f'{frequency} {label}/full-n: lower AIC than the current Full BEGE fit '
                          f'in {int((selected.delta_AIC < 0).sum())} of four means, lower BIC in '
                          f'{int((selected.delta_BIC < 0).sum())}')
    for note in notes:
        if note['id'] == 'Q11':
            note['status'] = 'Both proposed restrictions estimated at both frequencies'
            note['remaining'] = 'Reconciliation of unrestricted Full BEGE archived searches remains outstanding.'
            note['response'] = ('Both proposed restrictions have now been estimated at both frequencies, '
                'with 10,000 random starts per mean/specification/frequency. Both retain the full n recursion '
                'and use nine free volatility parameters. Comparing the four reported means: '
                + '; '.join(counts) + '. Standard errors are computed only for the selected best estimates. '
                'The restricted-model comparison reports likelihood and criterion differences using the '
                'same mean and sample. Separately, the admissible archived quarterly Full BEGE ARX(2,2) '
                'candidate has log likelihood −168.8770 versus −168.9699 in the current Full report; '
                'the unrestricted archived searches have not been merged. A restricted likelihood above '
                'a saved Full likelihood indicates incomplete numerical search, not a rejection of nesting.')
            note['evidence'] += [str(COMPARISON.relative_to(ROOT)),
                'BEGE_GARCH/SymmetricP_BEGE/README.md', 'BEGE_GARCH/SymmetricP_BEGE/README_monthly.md',
                'BEGE_GARCH/PositiveOnlyP_BEGE/README.md', 'BEGE_GARCH/PositiveOnlyP_BEGE/README_monthly.md']
            note['evidence'] = list(dict.fromkeys(note['evidence']))
        if note['id'] == 'B2':
            note['response'] = ('BEGE-EGARCH is specified and estimated at both frequencies, using '
                'coupled log-shape recursions. The report shows their equivalence to EGARCH recursions '
                'for the component volatilities. Its quarterly ARX(1,1) AIC is 360.6832. Updated '
                'rankings include the newly estimated symmetric-p/full-n and positive-only-p/full-n '
                'models; the report abstract identifies the quarterly criterion winners.')
    path.write_text(json.dumps(notes, indent=2, ensure_ascii=False)+'\n')


def include_articles():
    path = ROOT/'myst.yml'; text = path.read_text()
    for suffix in ('', '_monthly'):
        anchor = f'        - file: BEGE_GARCH/Full_BEGE/README{suffix}.md\n'
        additions = [f'        - file: BEGE_GARCH/{folder}/README{suffix}.md\n' for folder in SPECS]
        if additions[0] not in text:
            text = text.replace(anchor, anchor+''.join(additions), 1)
    article = f'        - file: {COMPARISON.relative_to(ROOT)}\n'
    if article not in text:
        # Put the joint comparison after all quarterly and monthly BEGE result pages.
        anchor = '        - file: BEGE_GARCH/BEGE_EGARCH/README_monthly.md\n'
        text = text.replace(anchor, anchor+article, 1)
    path.write_text(text)


def update_abstract_and_rankings():
    from ModelComparison import compare_models
    compare_models.main()
    ranked = pd.read_csv(ROOT/'ModelComparison/model_ranking_quarterly.csv')
    ranked = ranked.loc[~ranked.model.str.contains(r'ARX\([12],0\)', regex=True)]
    a = ranked.sort_values('aic').iloc[0]; b = ranked.sort_values('bic').iloc[0]
    path = ROOT/'README.md'; text = path.read_text()
    opening = ('This computational notebook documents the data construction, model specifications, '
        'estimation workflows, and best-model reporting used to study U.S. quarterly inflation dynamics. '
        'On the quarterly sample, the lowest '
        f'AIC among the reported models is {a.aic:.4f} for {a.model}; the lowest BIC is '
        f'{b.bic:.4f} for {b.model}. The specification labeled Constant uses the SPF-anchored mean.')
    text, count = re.subn(r'(\+\+\+ \{"part": "abstract"\}\n).*?(\n\+\+\+)',
                         lambda m: m[1]+opening+m[2], text, count=1, flags=re.S)
    if count != 1:
        raise ValueError('Abstract location changed; no automatic replacement.')
    path.write_text(text)
    # Recompute the exclusion comparison rather than retaining potentially stale winners.
    notes_path = ROOT/'RevisionResponses/responses.json'; notes = json.loads(notes_path.read_text())
    no_constant = ranked.loc[~ranked.model.str.contains(r'(?:^|, )Constant(?:,|$)')].sort_values('bic')
    monthly = pd.read_csv(ROOT/'ModelComparison/model_ranking_monthly.csv')
    monthly = monthly.loc[~monthly.model.str.contains(r'ARX\([12],0\)|(?:^|, )Constant(?:,|$)', regex=True)]
    mb = monthly.sort_values('bic').iloc[0]
    for note in notes:
        if note['id'] == 'B1':
            top = '; '.join(f'{r.model} (BIC {r.bic:.4f})' for r in no_constant.head(3).itertuples())
            note['response'] = ('The anchored mean remains a benchmark. Excluding it, while retaining '
                'estimated SPF loadings, the three lowest quarterly BICs are: '+top+'. '
                f'The monthly winner under this exclusion is {mb.model} (BIC {mb.bic:.4f}). '
                'These rankings include both new restricted Full BEGE specifications and the four '
                'report means. Mean coefficients remain jointly estimated with volatility; '
                'this is a sensitivity comparison, not a change to the estimation menu.')
    notes_path.write_text(json.dumps(notes, indent=2, ensure_ascii=False)+'\n')


def main():
    if (RUN_DIR/'report_withdrawn.json').exists():
        raise RuntimeError('Restricted-model results were withdrawn from the PDF by the user. Do not republish them without a new instruction.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--skip-build', action='store_true')
    args = parser.parse_args()
    audit = validate_completed_searches()
    if args.check_only:
        print(json.dumps(audit, indent=2)); return
    table = comparison_rows()
    table.to_csv(RUN_DIR/'comparison.csv', index=False)
    COMPARISON.write_text(comparison_text(table))
    update_responses(table)
    update_abstract_and_rankings()
    include_articles()
    methodology = ROOT/'BEGE_GARCH/README.md'
    text = methodology.read_text().replace(
        "The two p-loading restrictions are implemented. Their quarterly and monthly\n"
        "searches are in progress; fitted results will be added after the completed\n"
        "searches pass the collection checks.\n\n", "")
    methodology.write_text(text)
    if not args.skip_build:
        subprocess.run([sys.executable, str(ROOT/'scripts/build_report.py')], cwd=ROOT, check=True)
        # Extraction verifies a complete readable artifact and the new result sections.
        extracted = subprocess.check_output(['pdftotext', str(ROOT/'Inflation-Dynamics.pdf'), '-'], text=True)
        for phrase in ('Symmetric-p/full-n', 'Positive-only-p/full-n', 'Restricted Full BEGE: comparison'):
            if phrase not in extracted:
                raise ValueError(f'PDF missing {phrase!r}')
    (RUN_DIR/'completion.json').write_text(json.dumps(
        {'completed': datetime.now().isoformat(), 'pdf_built': not args.skip_build,
         'pdf_sha256': hashlib.sha256((ROOT/'Inflation-Dynamics.pdf').read_bytes()).hexdigest()
         if not args.skip_build else None, 'searches': audit},
        indent=2)+'\n')
    print('Restricted Full BEGE results validated, comparisons updated, report refined.')


if __name__ == '__main__':
    main()
