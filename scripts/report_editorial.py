"""Presentation-only edits to the report edition; never alter fitted values."""
import re


def compact_selection_tables(text):
    """Turn the wide, three-family GARCH panels into five-column tables.

    Family and Order are adjacent because together they name the volatility
    specification; the mean process follows.
    """
    pattern = r'^\| Distribution \| GARCH Mean \|.*?(?=\n\n)'

    def reshape(match):
        rows = [line.strip().strip('|').split('|') for line in match[0].splitlines()]
        rows = [[cell.strip() for cell in row] for row in rows]
        assert all(len(row) == 10 for row in rows)
        criterion = rows[0][3].split()[-1]
        result = [f'| Distribution | Family | Order | Mean | {criterion} |',
                  '|---|---|---|---|---:|']
        for row in rows[2:]:
            for i, family in enumerate(('GARCH', 'GJR-GARCH', 'EGARCH')):
                mean, order, value = row[1 + 3*i:4 + 3*i]
                result.append('| ' + ' | '.join(
                    [row[0], family, order, mean, value]) + ' |')
        return '\n'.join(result)

    return re.sub(pattern, reshape, text, flags=re.M | re.S)


def wrap_fitted_equations(text):
    """Wrap the long fitted GJR and ARX equations at a term boundary."""
    def wrap(match):
        body = match[1].strip()
        if '\n' in body or '=' not in body:
            return match[0]
        # Five or more terms in a fitted volatility equation, or the six-term
        # ARX(2,2) observation equation, can collide with the equation number.
        if not (body.startswith(r'\hat{\sigma}_t^2 =')
                or body.startswith(r'\ln \hat{\sigma}_t^2 =')
                or body.startswith(r'\pi_{t+1} =')):
            return match[0]
        # GARCH's signed coefficients have no space after the sign. Restrict
        # splitting to signed numeric coefficients, leaving t-1 indices alone.
        terms = re.split(r'\s+(?=[+-]\s*\d)', body)
        if len(terms) < 5:
            return match[0]
        cut = 3
        first = ' '.join(terms[:cut]).replace(' = ', ' &= ', 1)
        second = ' '.join(terms[cut:])
        return '\n$$\n\\begin{aligned}\n' + first + '\\\\\n&' + second + '\n\\end{aligned}\n$$\n'

    return re.sub(r'\n\$\$\n(.*?)\n\$\$\n', wrap, text, flags=re.S)


def polish(name, text):
    if name == 'DataSummary/README.md':
        text = text.replace('Inflation ($\\pi$)', r'Inflation ($\pi_t$)')
        text = text.replace('$SPF$ |', r'$SPF_{t-1}$ |')
        text = text.replace(r'$\pi - SPF$', r'$\pi_t-SPF_{t-1}$')
        text = re.sub(r' This is the method of the original data \(Aggregate_CPI_inflation_20230513.xls\):.*?(?=\n)', '', text)
    if name == 'GARCH/README.md':
        text = text.replace('first-order (KKT)', 'first-order Karush–Kuhn–Tucker (KKT)')
        text = text.replace('(iv) the sample log-likelihood', '(iv) their sum, the sample log-likelihood')
        # The sum of contributions is already defined once for all distributions.
        text = re.sub(r',\n\$\$\nand the sample log-likelihood is\n\$\$\n.*?\n\$\$',
                      '.\n$$', text, flags=re.S)
        # Put the mixture density and its component density on separate lines.
        text = text.replace(r'f_{\mathrm{mix}}(z) \;=\;',
                            '\\begin{aligned}\nf_{\\mathrm{mix}}(z) &\\;=\\;', 1)
        text = text.replace(r', \qquad \varphi(z;\mu,s^{2}) \;=\;',
                            ',\\\\\n\\varphi(z;\\mu,s^{2}) &\\;=\\;', 1)
        text = text.replace(r'\exp \left\{-\frac{(z-\mu)^{2}}{2 s^{2}}\right\}.',
                            '\\exp \\left\\{-\\frac{(z-\\mu)^{2}}{2 s^{2}}\\right\\}.\n\\end{aligned}', 1)
    if name == 'MeanProcess/README.md':
        # Monthly estimates immediately follow in their own article.
        text = re.sub(r'## Monthly Results\n.*', '', text, flags=re.S)
    if name == 'BEGE_GARCH/README.md':
        text, n = re.subn(r'The BEGE exercises estimate all six mean processes:\n.*?(?=The bounds and starting-value draws of the mean parameters)',
            'Each BEGE volatility specification is paired with the six mean processes defined in '
            'Mean Process Specification. Mean and volatility parameters are estimated jointly.\n\n',
            text, flags=re.S)
        if n != 1:
            raise ValueError(f'{name}: BEGE mean-process list not found')
    if 'garch_family_best_models.md' in name:
        frequency = 'Monthly' if 'monthly/' in name else 'Quarterly'
        text = text.replace('# Model Selection Report', f'# GARCH Model Selection ({frequency})')
        text = compact_selection_tables(text)
        text = re.sub(r'\*\*Table [1-4]: ', '**', text)
        text = text.replace('third column of Table 1', 'third column of the parameter-bounds table below')
        text = text.replace('Table 4 repeats the BIC selection', 'The next table repeats the BIC selection')
        text = text.replace('Because Table 3 selects no Constant-mean model, Table 4 coincides with Table 3.',
                            'Because the BIC table above selects no Constant-mean model, the two tables coincide.')
        # Table numbers are dropped in the report, so no numbered reference may remain.
        leftover = re.findall(r'\bTables? \d', text)
        if leftover:
            raise ValueError(f'{name}: numbered table references remain: {leftover}')
    if name.endswith('regime_switching_results.md'):
        frequency = 'Monthly' if 'results_monthly/' in name else 'Quarterly'
        # The generated selection tables have no heading of their own.
        start = text.index('Each model is selected')
        text = (text[:start] + '```{raw:typst}\n#pagebreak(weak: true)\n```\n\n'
                + f'# Regime-Switching Results ({frequency})\n\n' + text[start:])
        text = text.replace('`Sw.Dist` reflects whether $\\sigma^2$ is regime-specific.',
            '$K$ is the number of regimes. Sw.AR includes the intercept and inflation lags; '
            'Sw.SPF denotes the forecast coefficients; Sw.Dist denotes the innovation variance. '
            'Y means switching and N means shared.')
    if name.endswith('regime_switching_best_model.md'):
        frequency = 'Monthly' if 'results_monthly/' in name else 'Quarterly'
        text = text.replace('# Best Regime-Switching Model', f'# Best Regime-Switching Model ({frequency})')
    if re.fullmatch(r'BEGE_GARCH/([^/]+_BEGE|BEGE_EGARCH)/README(_monthly)?\.md', name):
        text = re.sub(r'^This report shows only .*?\n\n', '', text, flags=re.M)
        text = re.sub(r'^Best admissible (?:model|estimate) ranked by .*?\n\n', '', text, flags=re.M)
        text = text.replace('Estimations:', 'Saved results:')
        text = text.replace('Empirical path quantiles:', 'Empirical path quantiles (definitions in Residual Dynamics):')
    # Keep short lead-ins with the equation or table they introduce.
    labels = (r'Mean process:|BEGE volatility process:|Volatility process:|'
              r'Parameter table:|Distribution parameters:|Empirical path quantiles[^\n]*:')
    text = re.sub(r'^(' + labels + r')$',
                  lambda m: '```{raw:typst}\n#block(sticky: true)[' + m[0] + ']\n```',
                  text, flags=re.M)
    text = text.replace(r'1-10^{-6}', r'1-\delta')
    text = text.replace(r'\frac12', r'\frac{1}{2}')
    return wrap_fitted_equations(text)
