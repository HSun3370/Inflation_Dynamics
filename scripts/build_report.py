#!/usr/bin/env python3
"""Build the economics PDF and its online appendix from the canonical documents.

Run: python3 scripts/build_report.py
Requires the MyST and Typst CLIs on PATH. Estimation is never run.

The PDF holds the data, specifications, methods and estimation results. The BEGE
density evaluation is the online appendix, OnlineAppendix/BEGE_Density_Evaluation.md
(GitHub-flavoured Markdown), which the PDF links to on GitHub. Source documentation
and generated estimation reports remain unchanged.
"""
from pathlib import Path
from datetime import date
import json
import os
import re
import shutil
import subprocess
from report_editorial import polish
from report_revision_notes import add_revision_notes, write_checklist
from report_scope import omit_archived_table_rows, scope_report_text

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / '_build' / 'report-source'
PREAMBLE = '```{raw:typst}\n#set page(margin: auto)\n```\n\n'
PREAMBLE_RE = re.compile(r'```\{raw:typst\}\n#set page\(margin: auto\)\n```\n*')
GITHUB = re.search(r'^\s*github:\s*(\S+)', (ROOT / 'myst.yml').read_text(), re.M).group(1).rstrip('/')
GITHUB_BLOB = GITHUB + '/blob/main/'
GREY = '```{raw:typst}\n#set text(fill: rgb("#808080"))\n#show text: set text(fill: rgb("#808080"))\n```\n\n'
APPENDIX_DIR = ROOT / 'OnlineAppendix'
DENSITY_SOURCE = 'BEGE_GARCH/BEGE_Density/BEGE_Density.md'
DENSITY_APPENDIX = 'BEGE_Density_Evaluation.md'
# The BEGE-EGARCH specification is merged into the EGARCH subsection of the BEGE section.
EGARCH_SPEC = 'BEGE_GARCH/BEGE_EGARCH/SPECIFICATION.md'
# Typst layout for the report: sections run on without page breaks, paragraphs are
# compact, and tables may continue on the next page.
LAYOUT = (
    '#show par: set par(spacing: 1em)\n'
    '#show figure: set block(breakable: true)\n'
)


def replace_section(text, start, end, replacement=''):
    """Replace an explicitly named passage; fail if its source has drifted."""
    a = text.index(start)
    b = text.index(end, a) if end else len(text)
    return text[:a] + replacement + text[b:]


def replace_once(text, old, new):
    """Replace a passage that must exist; fail if its source has drifted."""
    if old not in text:
        raise ValueError(f'Report source passage not found: {old[:60]!r}')
    return text.replace(old, new)


def data_date():
    """Date the data run up to, from the record written by the data update routine."""
    vintage = json.loads((ROOT / 'DataSummary' / 'data_vintage.json').read_text())['vintage']
    d = date.fromisoformat(vintage)
    return f'{d:%B} {d.day}, {d.year}'


def remove_code_examples(text):
    # Keep MyST figures, tables, admonitions, math, and raw PDF styling.
    lines = text.splitlines(keepends=True)
    result = []
    i = 0
    while i < len(lines):
        match = re.match(r'^(`{3,}|~{3,})(.*)$', lines[i].rstrip())
        if not match:
            result.append(lines[i]); i += 1; continue
        fence, info = match.groups()
        end = i + 1
        while end < len(lines) and lines[end].strip() != fence:
            end += 1
        if end == len(lines):
            raise ValueError('Unclosed Markdown fence')
        if info.lstrip().startswith('{'):
            result.extend(lines[i:end + 1])
        i = end + 1
    return ''.join(result)


def egarch_specification():
    """BEGE-EGARCH specification as the body of the BEGE section's EGARCH subsection."""
    text = PREAMBLE_RE.sub('', (ROOT / EGARCH_SPEC).read_text())
    text = replace_section(text, '# BEGE-EGARCH Specification', 'The inflation residual is',
        'This extension of Full BEGE replaces the GJR-type shape recursions with coupled '
        'EGARCH-type recursions on the log shapes. ')
    text = replace_section(text, '## Estimation Scripts', None)
    # The report keeps only the persistence restriction of Parameters and Stability.
    # The parameter-order display, the pointer to the BEGE parameter table (which
    # follows in the same section), and the shared-settings sentence are omitted.
    text = replace_once(text, '## Parameters and Stability', '## Persistence Restriction')
    text = replace_section(text, 'Parameter order in estimation output:', 'The persistence restriction')
    text = replace_section(text, ' The parameter bounds, random-start', '\n\n')
    # Subsections of the EGARCH subsection (level 3) are level-4 run-in headings.
    text = re.sub(r'^## ', '#### ', text, flags=re.M)
    return text.strip() + '\n\n'


def report_text(name, text):
    text = re.sub(r'<!--.*?-->', '', text, flags=re.S)
    if name == 'README.md':
        text = text.replace(
            'This computational notebook documents the data construction, model specifications, estimation workflows, and best-model reporting used to study U.S. quarterly inflation dynamics.',
            'This report studies U.S. inflation dynamics through conditional mean, volatility, and regime-switching models. It presents the data, econometric specifications, estimation methods, and empirical results, with quarterly estimates as the primary focus and monthly estimates as supplementary evidence.')
        text = text.replace('This notebook is organized', 'This report is organized')
    elif name == 'DataSummary/README.md':
        text = replace_section(text, '## Routine Data Update', '### Current Vintage')
        # The report states the data date once, in the Data Coverage list; vintage
        # history, reference-workbook checks and cross-vintage notes are omitted.
        text = text.replace('### Current Vintage', '## Data Coverage')
        text = re.sub(r'^- \*\*Data vintage\*\*: .*$', f'- **Data**: up to {data_date()}.', text, count=1, flags=re.M)
        text = re.sub(r'^- \*\*Reference check\*\*.*\n', '', text, flags=re.M)
        text = replace_section(text, '### Comparability of Committed Results', '```{figure} cpi_inflation_quarterly.png')
        text = replace_once(text, 'the value used in the current vintage is listed under [Current Vintage](#current-vintage).',
            'its value is listed under [Data Coverage](#data-coverage).')
        text = replace_once(text, 'its end date and size for the current vintage are listed under [Current Vintage](#current-vintage).',
            'its end date and size are listed under [Data Coverage](#data-coverage).')
        # The GARCH recursion-initialization paragraph is omitted from the report.
        text = replace_section(text, 'For the reported GARCH-family estimates, conditional-variance recursion starts', '\n\n')
        # The effective-sample correction footnote is omitted from the report.
        text = replace_once(text, '[^effective-sample-correction]\n', '\n')
        text = replace_section(text, '[^effective-sample-correction]: ', '\n', '')
        text = replace_section(text, 'These rules were reverse-engineered', '```{raw:typst}')
        # The monthly SPF rule names the original workbook in plain text.
        text = replace_once(text, 'This is the method of the original data, `Aggregate_CPI_inflation_20230513.xls`:',
            'This is the method of the original data (Aggregate_CPI_inflation_20230513.xls):')
        # The report's construction rules omit the inflation-shock and
        # unpublished-CPI-month items (kept in the canonical README).
        text = replace_section(text, '- **Inflation shock**', '\n\n')
        # The report's data-decision box keeps the decision itself; its Why and
        # Scope paragraphs stay only in the canonical DataSummary/README.md.
        text = replace_section(text, '\n\n*Why:*', '\n```')
        text = text.replace('Summary statistics for the trimmed effective sample are reported below (generated by `data_summary.py`).', 'Summary statistics for the effective sample are reported below.')
        text = text.replace('Summary statistics for the trimmed monthly effective sample (generated by `data_summary.py`):', 'Summary statistics for the monthly effective sample:')
        text = replace_section(text, 'The monthly counterpart to the quarterly estimation dataset', '\n\nThe monthly effective sample',
            'For an observation $\\pi_t$, the monthly sample requires both $SPF_{t-1}$ and $SPF_{t-2}$. Forecasts are available for target months starting in **1969M1**, so the effective sample begins in **1969M2** and ends in **2026M9** (**692 observations**). The quarterly professional forecast is geometrically de-compounded into a monthly rate and is constant within each target quarter.')
    elif name == 'MeanProcess/README.md':
        text = text.replace('The OLS results below are generated by `mean_process_ols.py`.\n\n', '')
    elif name == 'GARCH/README.md':
        text = replace_section(
            text, '## Volatility Recursion Specifications',
            '## Estimation Initialization And Collection',
            'The complete mean and volatility equations, with each set of parameter '
            'restrictions stated immediately after its equation, are given with the '
            'quarterly GARCH results. Those specifications also apply to the monthly '
            'results.\n\n',
        )
        text = replace_section(text, '### Running the search on the CBS Grid', '## Log-Likelihood Specifications')
        # The delta-method standard error of the GJR negative-shock coefficient is standard.
        text = replace_section(text, ' For GJR-GARCH, `arch` estimates the indicator form', '## Log-Likelihood Specifications', '\n\n')
        text = replace_section(text, 'The GARCH-family estimation is executed', 'The reported estimate for a combination',
            'Parameters are estimated by constrained maximum likelihood using SLSQP and **5,000 starting points per model combination**. Initial values are drawn within the parameter bounds and stationary region; volatility persistence and inflation-lag AR roots are restricted below $1-10^{-6}$ in modulus.\n\n')
        text = replace_section(text, 'The second check is needed', 'Standard errors are robust',
            'First-order checks screen out steep likelihood ridges that can otherwise be mistaken for maxima. Up to five local restarts are used before rejecting a candidate. The mixture variance-ratio restriction prevents near-degenerate components from generating spurious likelihood improvements. If no estimate passes all checks, the best converged first-order-valid estimate is reported with a qualification. Conditional-variance recursion uses the default initialization of the estimation method.\n\n')
        text = replace_section(text, 'The same estimation menu', None,
            'The monthly analysis uses the same three volatility families, six mean processes, three innovation distributions, and 5,000 starting points per combination as the quarterly analysis. The effective sample is the monthly effective sample described in the Data Summary. All model restrictions and selection criteria are identical across frequencies.\n')
        text = text.replace('## Estimation Initialization And Collection', '## Estimation and Inference')
        text = replace_once(text, "Standard errors are robust (sandwich) and are computed only for each combination's reported estimate.",
            "Standard errors are robust (sandwich) and are computed only for each combination's reported estimate. For estimates on parameter bounds or active constraints, the usual normal approximation may fail, so these standard errors require caution.")
    elif name == 'RegimeSwitching/README.md':
        text = replace_section(text, 'Only Normal specifications', '## Data Frequency and Running the Search',
            'The reported results use normal innovations.\n\n')
        text = replace_section(text, '## Data Frequency and Running the Search', None,
            '## Data Frequency\n\n' +
            'The quarterly estimates reported here use the effective sample **1969Q2--2026Q3 (230 observations)**.\n\n' + GREY +
            'The monthly estimates use **1969M2--2026M9 (692 observations)**. The specification menu, estimation procedure, and admissibility criteria are identical across frequencies.\n')
        # The procedure is the authors' own, so the report describes it without a citation.
        text = replace_section(text, 'The identification and likelihood-maximization procedure follows', '## Model Specification',
            'I estimate Gaussian regime-switching versions of the inflation mean processes below, with '
            'unconditional initial regime probabilities, variance-based regime labeling, multiple initial '
            'parameter vectors, and final-solution perturbation checks.\n\n')
        text = replace_section(text, 'At the row for $\\pi_{t+1}$, the forecast $SPF_t$ is stored in',
            'The mean menu is Constant',
            'All lags and forecasts are observed inputs from the common effective sample; '
            'no additional observations are removed to construct regressors.\n\n')
        text = replace_once(text, 'Following the identification convention in Section 4.1.2 of Bekaert et al.,\nI label', 'I label')
        text = replace_once(text, ' These are the\nmultiple-start and final-solution checks described in Appendix B of\nBekaert et al.', '')
        text = replace_once(text, 'Bekaert et al. do not specify the optimizer, number or distribution of\nstarts, EM initialization, perturbation size, or numerical return tolerance.\n'
                            'The following values document this implementation of their procedure.',
                            'The numerical settings of the implementation are as follows.')
        text = replace_section(text, "Following the user's decision on 2026-11-10 to use the paper's stated",
                               'Without a variance floor, the Gaussian likelihood is unbounded')
        text = replace_section(text, 'On 2026-11-12, I corrected the Gaussian EM mean update',
            'After combining all search blocks for a specification')
        text = replace_section(text, 'Without a variance floor, the Gaussian likelihood is unbounded', 'For the inflation ARX specifications',
            'Without a variance floor, the Gaussian likelihood is unbounded: as a regime variance approaches zero, '
            'that regime can fit a few observations exactly. Such degenerate optima arise in several switching-AR '
            'specifications. They fail the final-solution verification because the perturbation refits do not '
            'return to them, and an optimum whose likelihood is not reproduced by its parameters is not verified. '
            'Specifications without a verified estimate are excluded from the results tables.\n\n')
        text = replace_once(text, "This is an assumption of the inflation application; the paper's benchmark\nreturn equation has no AR coefficients.",
                            'This is an assumption of the inflation application.')
        if re.search(r'Bekaert|the paper', text):
            raise ValueError(f'{name}: citation of the paper remains in the report edition')
        text = text.replace('The CSV files and generated tables use zero-based labels',
                            'The tables use zero-based labels')
        # Present definitions, filtering, and smoothing consecutively in the PDF.
        # Retain the specification menu and implementation audit in the source notes.
        text = replace_section(text, "In the code's observation-date convention",
                               'All lags and forecasts are observed inputs')
        text = replace_section(text, 'All lags and forecasts are observed inputs',
                               '## Filtering and the Likelihood')
        text = replace_section(text, 'In implementation the likelihood is evaluated',
                               '### How Partial Switching Changes the Update')
        moments_start = text.index('### How Partial Switching Changes the Update')
        smoothing_start = text.index('## Smoothing Probability')
        estimation_start = text.index('## Estimation with Shared Coefficients')
        smoothing = text[smoothing_start:estimation_start]
        text = text[:moments_start] + smoothing + text[estimation_start:]
        text = replace_section(text, 'An independent implementation of these equations',
                               '## Identification')
        text = text.replace('The existing implementation uses an expected-count transition update',
                            'The expected-count transition update is used')
        text = text.replace('### Numerical Implementation', '### Numerical Estimation Settings')
        text = text.replace('The numerical settings of the implementation are as follows.',
                            'The numerical estimation settings are as follows.')
        text = text.replace('The script uses the following Gaussian mean and',
                            'The Gaussian mean and')
        text = text.replace('variance updates.', 'variance updates are as follows.')
        text = text.replace('Start 1 uses the\nstatsmodels defaults:', 'The first starting vector uses')
        text = replace_section(text, ' (raised from 50 on',
                               ', after which BFGS')
        text = replace_section(text, ' L-BFGS uses `pgtol=1e-7`',
                               ' A return requires optimizer convergence,')
        text = replace_section(text, ' Each initial start and verification',
                               '## Parameter Validity and Collection', '\n\n')
        # Keep the post-estimation check concise; the source notes retain
        # the full random-start construction and numerical settings.
        text = replace_section(text, '## Perturbation after Estimation',
            '## Parameter Validity and Collection',
            r'''## Perturbation after Estimation

I numerically maximize the likelihood using different initial parameter
vectors to reduce the risk of selecting a local maximum. I then check
the stability of the final solution by randomly deviating from the
estimates and verifying that the optimization routine returns to the
provisional maximum.

For this application, I select the highest-likelihood admissible estimate
from 5,000 starts and perturb all free parameters in unrestricted
optimization coordinates. For each coordinate $a_j$, draw

$$
a_j^*=\hat{a}_j+0.05\max\{1,|\hat{a}_j|\}z_j,
\qquad z_j\stackrel{\mathrm{i.i.d.}}{\sim}N(0,1).
$$

Each batch contains 20 refits, alternating BFGS and L-BFGS, with no EM
initialization and at most 1,500 optimizer iterations per refit.
Every refit in the final batch must converge, pass the model checks,
and return within $10^{-4}$ of the reference log likelihood in absolute value.
If a higher admissible maximum is found, it becomes the new reference
and requires a fresh batch. I allow at most three batches and exclude
unverified estimates from the reported tables.

The perturbation scale, refit counts, and return tolerance are numerical
choices for this application. The refits are additional to the 5,000
initial starts and assess local stability, not global maximization.

''')
        # Replace the opening validity discussion with the parameter/draw table.
        text = replace_section(text, 'The reported estimate must have successful optimizer convergence',
                               'For the inflation ARX specifications')
        # The table now states the regime-specific AR restrictions explicitly.
        text = replace_section(text, 'For the inflation ARX specifications',
                               'The SPF loadings are excluded from the AR polynomial.')
    elif name == 'BEGE_GARCH/README.md':
        text = replace_once(text, ', both from the 2026-10-20 data vintage.', '.')
        # The report states the search size without its job/draw layout.
        text = replace_section(text, 'For each BEGE specification and mean process, I randomly sample',
            '\n\nThe table below summarizes',
            'For each BEGE specification and mean process, I run the optimizer from 10,000 random '
            'starting points and save the best result of every 25 optimizations. Thus a report covering '
            'six mean processes contains 2,400 saved results from 60,000 optimizations.')
        # The parameter table is printed without the log-shape cap row and without
        # its notes; the mean-start symbols are defined in the lead-in instead.
        text = replace_once(text, '| Log-shape cap | $\\ln p_t,\\ln n_t\\le50$ in BEGE-EGARCH | --- |\n', '')
        text = replace_section(text, 'Notes:\n\n- $U[a,b]$ is the uniform distribution', '## Density Evaluation')
        text = replace_once(text, 'parameter. Constant BEGE is maximized',
            'parameter. In the mean-process rows, $\\hat\\theta_j$ and $\\widehat{SE}_j$ are the OLS estimate '
            'of mean parameter $j$ and its HAC(12) standard error. Constant BEGE is maximized')
        # The rough standard errors are described briefly; the README lists the method chain.
        text = replace_section(text, 'Every reported parameter receives a standard error.', '$<0.0001$.',
            'Every reported parameter receives a standard error. An estimate on a bound of its admissible '
            'range, or one whose sandwich standard error is not numerically identified, receives a rough '
            'standard error marked with $\\dagger$, because the usual normal approximation does not apply '
            'there; the note under its parameter table states how it is computed. Standard errors below '
            'the display precision are shown as ')
        text = replace_once(text, 'twice on the 2026-10-20 data vintage, once with', 'twice, once with')
        text = replace_once(text, 'documented in the BEGE Density section.',
            f'documented in the [BEGE density evaluation]({GITHUB_BLOB}OnlineAppendix/{DENSITY_APPENDIX}) of the online appendix.')
        text = text.replace('The by-mean best\nrows remain available in the CSV outputs linked from each report. ', '')
        text = text.replace('using the four-point\ncentered finite-difference method implemented by\n`statsmodels.tools.numdiff.approx_hess`.', 'using four-point centered finite differences.')
        text = text.replace('the collector uses an', 'inference uses an')
        text = replace_section(text, '### BEGE-EGARCH (extension)\n', '## Estimation and Random Starts',
                               '### BEGE-EGARCH (extension)\n\n' + egarch_specification())
    if 'garch_family_best_models.md' in name:
        text = omit_archived_table_rows(text)
        text = re.sub(r', data vintage \d{4}-\d{2}-\d{2}\)', ')', text)
        # The table of every model combination is linked on GitHub instead of printed.
        start, end = '## All Model Results', '## Best Models by Criterion and Volatility Family'
        section = text[text.index(start):text.index(end)]
        n_models = sum(1 for line in section.splitlines()
                       if line.startswith('| ') and not line.startswith('| Distribution |'))
        if n_models == 0:
            raise ValueError(f'{name}: no rows in the All Model Results table')
        frequency = 'Monthly' if 'results_garch_distributions_monthly/' in name else 'Quarterly'
        online_table = f'GARCH_Results_{frequency}.md'
        APPENDIX_DIR.mkdir(exist_ok=True)
        (APPENDIX_DIR / online_table).write_text(
            '<!-- Generated by scripts/build_report.py; do not edit by hand. -->\n\n'
            f'# GARCH Results ({frequency})\n\n'
            'The four reported mean processes are Constant, ARX(1,1), ARX(2,1), and ARX(2,2). '
            'Each combination uses 5,000 optimizer starts. Selection qualifications are given '
            'in the main report.\n\n' + section)
        text = replace_section(text, start, end,
            f'{start}\n\nThe best log likelihood, AIC, and BIC of all {n_models} estimated model '
            f'combinations are listed in the [online results table]({GITHUB_BLOB}OnlineAppendix/{online_table}).\n\n')
        if 'results_garch_distributions_monthly/' in name:
            equations = (
                'The mean equations, volatility equations, innovation distributions, and '
                'parameter restrictions are identical to those stated with the quarterly '
                'GARCH results and are not repeated here.\n\n'
            )
        else:
            equations = (ROOT / 'GARCH' / 'report_model_equations.md').read_text()
            if not equations.startswith(PREAMBLE):
                raise ValueError('GARCH report equations must retain the Typst preamble')
            equations = equations[len(PREAMBLE):] + '\n\n'
        # Random-start ranges: the table is printed with the quarterly results;
        # the monthly results give their own v-bar values and refer to it.
        ranges = text[text.index('## Parameter Ranges and Random Starts'):text.index('## Model Selection')]
        vbar = re.search(r'^\$\\bar v\$ denotes .*$', ranges, re.M).group(0)
        if 'results_garch_distributions_monthly/' in name:
            starts = ('Each model combination uses 5,000 starts. Start 1 is the default start of the estimation '
                      'package; starts 2--5,000 are random draws from the ranges in the random-start table of the '
                      'quarterly GARCH results. ' + vbar.replace('$\\bar v$ denotes', 'On the monthly sample, $\\bar v$,') )
            starts = replace_once(starts, 'Constant mean):', 'Constant mean), is:') + ' Reported standard errors are robust.'
        else:
            table = re.search(r'^\| Parameter \|.*?(?=\n\n)', ranges, re.M | re.S).group(0)
            table = replace_once(table, '| Parameter | `arch` bound / constraint | Random-start draw |',
                                 '| Parameter | Bounds and selection restrictions | Random-start draw |')
            table = replace_once(table, '`arch` estimates $\\gamma_i-\\alpha_i\\in[-1,2]$', 'estimated as $\\gamma_i-\\alpha_i\\in[-1,2]$')
            starts = ('Each model combination uses 5,000 starts. Start 1 is the default start of the estimation '
                      "package (Gaussian mixture: the Student's $t$ estimates plus $(p_1,\\mu_1,\\sigma_1^2)=(0.3,0.1,1)$ "
                      'when that vector is feasible). Starts 2--5,000 are random draws from the third column of Table 1. '
                      "Every draw lies inside the estimation package's bounds and constraints (second column) and inside the "
                      'stationary region, where volatility persistence and the AR roots are below $1-10^{-6}$ in modulus, '
                      'so every random start is used as drawn. Reported standard errors are robust. ' + vbar + '\n\n'
                      '**Table 1: Parameter bounds and random-start draws**\n\n' + table)
        text = replace_section(text, '## Parameter Ranges and Random Starts', '## Model Selection',
            '## Estimation\n\n' + starts + '\n\n' + equations + '\n\n')
        text = re.sub(r' and is flagged in the CSV `selection_status` column', ' and is qualified accordingly', text)
        text = text.replace('Reported in README notation. (`arch` estimates the equivalent indicator form.)', 'Reported in the model notation defined above.')
        text = text.replace('of the README specification', 'of the model specification')
        # The delta-method variance of gamma_i is standard; the report omits it (as in the GARCH section).
        text, n = re.subn(r'; its standard error uses the delta method on the same robust covariance matrix, \$[^$]+\$\.',
                          '.', text)
        if n != 2:
            raise ValueError(f'{name}: expected two GJR delta-method notes, found {n}')
        text = text.replace('For GJR-GARCH, `arch` estimates', 'For GJR-GARCH, the estimation uses')
        text = text.replace("`arch` estimates", 'The estimation uses')
    if re.fullmatch(r'BEGE_GARCH/([^/]+_BEGE|BEGE_EGARCH)/README(_monthly)?\.md', name):
        text = re.sub(r'^Generated:.*\n', '', text, flags=re.M)
        # Link the stored random-search results of this model and frequency on GitHub.
        results = name.rsplit('/', 1)[0] + ('/results_monthly/' if name.endswith('_monthly.md') else '/results/')
        url = lambda f: f'{GITHUB_BLOB}{results}{f}'
        means = [('Constant', 'constant'), ('ARX(1,1)', 'ARX_1_1'),
                 ('ARX(2,1)', 'ARX_2_1'), ('ARX(2,2)', 'ARX_2_2')]
        missing = [f for f in ['all_estimations.csv', 'selection_diagnostics.csv', 'path_quantile_diagnostics.csv',
                               'best_loglik_with_se.csv', *(f'by_mean/{m}.csv' for _, m in means)]
                   if not (ROOT / results / f).exists()]
        if missing:
            raise ValueError(f'{name}: stored BEGE results missing: {missing}')
        by_mean = ', '.join(f'[{label}]({url(f"by_mean/{m}.csv")})' for label, m in means)
        text = replace_section(text, 'CSV outputs:', '## Selected Best Model',
            'Stored results: '
            f'[all saved results]({url("all_estimations.csv")}); by mean: {by_mean}; '
            f'[selection checks]({url("selection_diagnostics.csv")}), '
            f'[path quantiles]({url("path_quantile_diagnostics.csv")}), and '
            f'[best estimate with standard errors]({url("best_loglik_with_se.csv")}).\n\n')
        # The three count lines would otherwise run together as one paragraph.
        text, n = re.subn(r'^Total estimations: `(\d+)`\n(Converged|Successful) estimations: `(\d+)`\n'
                          r'Eligible estimations for best-model selection: `(\d+)`\n',
                          lambda m: (f'Estimations: {int(m[1]):,} in total, {int(m[3]):,} {m[2].lower()}, '
                                     f'{int(m[4]):,} eligible for best-model selection.\n'),
                          text, flags=re.M)
        if n != 1:
            raise ValueError(f'{name}: expected one block of estimation counts')
        # Seed and draw only identify the search job that found the estimate.
        headers = re.findall(r'^\| Mean \| Seed \| Draw \| LogLik \| AIC \| BIC \|.*\n', text, re.M)
        if len(headers) != 1:
            raise ValueError(f'{name}: expected one best-model summary table')
        a = text.index(headers[0])
        b = text.find('\n\n', a)
        b = len(text) if b < 0 else b
        rows = [r.split('|') for r in text[a:b].split('\n')]
        text = text[:a] + '\n'.join('|'.join(r[:2] + r[4:]) for r in rows) + text[b:]
    if name in ('RegimeSwitching/results/regime_switching_best_model.md',
                'RegimeSwitching/results_monthly/regime_switching_best_model.md'):
        text = replace_section(text, '## Unconditional and Sample-Averaged Moments',
                               '## Switching Blocks')
    text = polish(name, text)
    # Remove executable examples, retaining equations, tables, figures and colour rules.
    text = remove_code_examples(text)
    # Replace residual source paths with reader-facing descriptions.
    text = re.sub(r', `Aggregate_CPI_inflation_(Quarterly|Monthly)\.pkl`', '', text)
    text = text.replace('`arch` convergence flag 0', 'a zero convergence flag')
    text = text.replace('`arch`', 'the estimation package')
    text = text.replace('`NA`', 'unavailable')
    text = text.replace(' (Median_PGDP_Level.xlsx)', '').replace(' (`Median_PGDP_Level.xlsx`)', '')
    text = text.replace('`Aggregate_CPI_inflation_20230513.xls`', 'the 2023 reference data')
    text = text.replace('2026-10-20 .', '2026-10-20.')
    text = text.replace('and give statistical summary', 'and presents summary statistics')
    text = text.replace('BEGE_GARCH/README.md', 'the BEGE methodology section')
    # Inline identifiers such as model IDs are labels, not code examples.
    text = re.sub(r'(?<!`)`([^`\n]+)`(?!`)', r'\1', text)
    # The page margin is set once for the whole report (see LAYOUT); a per-article
    # page rule would start every article on a new page.
    text = PREAMBLE_RE.sub('', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text).lstrip()
    return scope_report_text(name, add_revision_notes(name, text))


def to_gfm(text, source):
    """Convert report Markdown (MyST) to GitHub-flavoured Markdown."""
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        match = re.match(r'^(`{3,}|~{3,})\{([\w:-]+)\}\s*(.*)$', lines[i])
        if not match:
            out.append(lines[i]); i += 1; continue
        fence, directive, arg = match.groups()
        end = i + 1
        while end < len(lines) and lines[end].strip() != fence:
            end += 1
        body = [line for line in lines[i + 1:end] if not re.match(r'^:[\w-]+:', line)]
        i = end + 1
        if directive.startswith('raw'):
            continue  # PDF styling only
        if directive == 'table':
            out += ([f'**{arg}**', ''] if arg else []) + body
        elif directive == 'figure':
            caption = ' '.join(line.strip() for line in body if line.strip())
            out += [f'![{caption}]({arg})', '', f'*{caption}*']
        elif directive in ('note', 'important', 'warning', 'tip'):
            out += [f'> **{arg or directive.title()}**', '>'] + [f'> {line}'.rstrip() for line in body]
        else:
            raise ValueError(f'{source}: no GitHub rendering for the {directive} directive')
    text = '\n'.join(out) + '\n'
    text = re.sub(r'^%.*\n', '', text, flags=re.M)  # MyST comments
    # GitHub drops inline colour; the best value in a selection table is shown in bold.
    text = re.sub(r'<span style="color:\s*red">(.*?)</span>', r'**\1**', text)
    base = (ROOT / source).parent

    def relink(match):
        label, target = match.groups()
        if re.match(r'(https?:|mailto:|#)', target):
            return match.group(0)
        path, _, fragment = target.partition('#')
        rel = os.path.relpath(os.path.normpath(base / path), APPENDIX_DIR)
        return f'{label}({rel}{"#" + fragment if fragment else ""})'

    text = re.sub(r'(!?\[[^\]]*\])\(([^)\s]+)\)', relink, text)
    text = re.sub(r'(?<=\w)--(?=\w)', '–', text)  # MyST's en dash, e.g. 1969Q2--2026Q3
    # GitHub renders $$ display math reliably only as a block of its own.
    lines, in_math = [], False
    for line in text.split('\n'):
        if line.strip() == '$$':
            if not in_math and lines and lines[-1].strip():
                lines.append('')
            lines.append(line)
            if in_math:
                lines.append('')
            in_math = not in_math
        else:
            lines.append(line)
    text = '\n'.join(lines)
    # Count lines of the BEGE reports would otherwise run together into one paragraph.
    text = re.sub(r'^((?:Total|Successful|Converged|Eligible) estimations\b.*)$', r'- \1', text, flags=re.M)
    return re.sub(r'\n{3,}', '\n\n', text).strip() + '\n'


def write_appendix():
    """Write the online appendix: the BEGE density evaluation, as GitHub-flavoured Markdown."""
    APPENDIX_DIR.mkdir(exist_ok=True)
    text = to_gfm(report_text(DENSITY_SOURCE, (ROOT / DENSITY_SOURCE).read_text()), DENSITY_SOURCE)
    text = re.sub(r'^# .*\n', '', text, count=1, flags=re.M)
    intro = (
        '<!-- Generated by scripts/build_report.py from BEGE_GARCH/BEGE_Density/BEGE_Density.md; do not edit by hand. -->\n\n'
        '# Online Appendix: BEGE Density Evaluation\n\n'
        f'This online appendix to the report [*Inflation Dynamics*]({GITHUB_BLOB}Inflation-Dynamics.pdf) '
        'documents the evaluation of the BEGE density used in likelihood estimation.\n\n'
    )
    (APPENDIX_DIR / DENSITY_APPENDIX).write_text(intro + text)
    print('Wrote online appendix:', APPENDIX_DIR / DENSITY_APPENDIX, flush=True)


def compact_layout(typ):
    """Lay out MyST's Typst export compactly; only the title page keeps the template's wide margin."""
    if '\n#pagebreak()\n' not in typ:
        raise ValueError('MyST Typst export no longer separates articles with page breaks')
    typ = typ.replace('\n#pagebreak()\n', '\n')
    typ = replace_once(typ, '#import "myst-imports.typ": *\n', '#import "myst-imports.typ": *\n' + LAYOUT)
    first = re.search(r'^#include "(.+)"$', typ, re.M)
    # Keep the title/abstract/roadmap on one page. Put the notation at the
    # beginning of the normal-width body, rather than spilling the title-page
    # margin onto a mostly empty second page.
    intro = STAGE / 'exports' / first[1]
    before, marker, notation = intro.read_text().partition('The following notation is used throughout:')
    if not marker:
        raise ValueError('Introductory notation block not found')
    intro.write_text(before.rstrip() + '\n')
    (STAGE / 'exports' / 'report-notation.typ').write_text(
        '#import "myst-imports.typ": *\n\n*Notation and reporting conventions.*\n\n'
        + marker + notation)
    return typ.replace(first[0] + '\n', first[0] +
        '\n\n#set page(margin: auto)\n#include "report-notation.typ"\n', 1)


def prepare():
    write_checklist()
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    config = (ROOT / 'myst.yml').read_text()
    # The BEGE density evaluation is published as the online appendix instead.
    config = replace_once(config, f'        - file: {DENSITY_SOURCE}\n', '')
    # The BEGE-EGARCH specification appears inside the BEGE section instead.
    config = replace_once(config, f'        - file: {EGARCH_SPEC}\n', '')
    # Export Typst source, which main() lays out before compiling the PDF.
    config = replace_once(config, '    - format: pdf\n', '    - format: typst\n')
    config = replace_once(config, '      output: Inflation-Dynamics.pdf\n', '      output: exports/Inflation-Dynamics.typ\n')
    config = config.replace('Inflation Dynamics - Computational Notebook', 'Inflation Dynamics')
    config = config.replace('kind: "Computation Notebook"', 'kind: "Econometric Report"')
    config = config.replace('This repository contains computational notebooks and resources related to the study of inflation dynamics in economics.', 'An econometric analysis of U.S. inflation dynamics.')
    report_date = os.environ.get('REPORT_DATE')
    if report_date:
        # Explicit dates make exports reproducible across server time zones.
        report_date = date.fromisoformat(report_date).isoformat()
        config = config.replace('date: null', f'date: "{report_date}"')
    (STAGE / 'myst.yml').write_text(config)
    articles = re.findall(r'^        - file: (.+?)\s*$', config, re.M)
    for name in articles:
        source = ROOT / name
        text = source.read_text()
        if not text.strip():
            print('Empty source retained without invented content:', name)
        target = STAGE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report_text(name, text))
    # Mirror image paths so figure references and relative cross-references survive.
    for directory, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {'.git', '_build', '.venv', 'venv', '.claude', '.agents', '__pycache__'}]
        for filename in files:
            source = Path(directory) / filename
            if source.suffix.lower() in {'.png', '.jpg', '.jpeg', '.svg'}:
                target = STAGE / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    print('Prepared report sources:', STAGE, flush=True)
    print('Omitted implementation examples and operational instructions.', flush=True)


def fix_links(typ):
    """Repair link targets in MyST's Typst export.

    MyST escapes Markdown characters such as `_` and `#` inside link URLs, and
    Typst keeps those backslashes in the PDF link, which breaks GitHub links.
    MyST also auto-links abbreviations that look like domain names (`Sw.AR`).
    """
    typ = re.sub(r'#link\("http://([^"/\s]+)"\)\[\1\]', r'\1', typ)
    typ = re.sub(r'(#link\(")([^"]*)("\))',
                 lambda m: m[1] + re.sub(r'\\([_#*$@~`<>\[\]])', r'\1', m[2]) + m[3], typ)
    if re.search(r'#link\("[^"]*\\', typ):
        raise ValueError('A link target in the Typst export still contains a backslash')
    return typ


def main():
    write_appendix()
    prepare()
    subprocess.run(['myst', 'build', '--typst'], cwd=STAGE, check=True)
    for part in (STAGE / 'exports').glob('*.typ'):
        text = fix_links(part.read_text())
        # Standalone bold instructions and table captions should introduce
        # content on the same page, including the filtering/smoothing steps.
        text = re.sub(r'^\*[^*\n]+\*$',
            lambda m: '#block(sticky: true)[' + m[0] + ']', text, flags=re.M)
        if part.name == 'lapreprint.typ':
            # The template's final counter can be stale when the long document
            # reaches Typst's layout iteration limit. Show the actual page only.
            text = replace_once(text,
                '#counter(page).display() of #counter(page).final().first()',
                '#counter(page).display()')
            text = replace_once(text, '(short-title, short-citation).join(spacer)',
                '(short-title, short-citation).filter(x => x != none).join(spacer)')
            text = replace_once(text, 'show heading: it => context {',
                'show heading: it => block(sticky: true, context {')
            text = replace_once(text, '\n  }\n\n\n  if (logo != none)',
                '\n  })\n\n\n  if (logo != none)')
        if part.name == 'myst-imports.typ':
            text = replace_once(text,
                '#import "@preview/tablex:0.0.9": tablex, cellx, hlinex, vlinex',
                (ROOT / 'scripts' / 'report_tables.typ').read_text())
        part.write_text(text)
    typ = STAGE / 'exports' / 'Inflation-Dynamics.typ'
    typ.write_text(compact_layout(typ.read_text()))
    pdf = STAGE / 'Inflation-Dynamics.pdf'
    subprocess.run(['typst', 'compile', str(typ), str(pdf)], cwd=STAGE, check=True)
    if not pdf.is_file() or pdf.stat().st_size == 0:
        raise RuntimeError('PDF compilation failed; the previous report was preserved.')
    shutil.copy2(pdf, ROOT / pdf.name)
    print('Updated:', ROOT / pdf.name)


if __name__ == '__main__':
    main()
