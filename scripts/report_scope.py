"""Limit the PDF to the four SPF-based means; preserve estimation archives."""
import csv
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ARCHIVED_MEANS = {'ARX(1,0)', 'ARX(2,0)'}


def omit_archived_table_rows(text):
    return '\n'.join(line for line in text.split('\n')
                     if not (line.startswith('|') and any(m in line for m in ARCHIVED_MEANS)))


def scope_report_text(name, text):
    # OLS specification bullets, fitted equations and diagnostics.
    text = re.sub(r'^- \*\*ARX\([12],0\)\*\*:.*\n', '', text, flags=re.M)
    text = re.sub(r'^\*\*ARX\([12],0\)\*\*:\n.*?(?=^\*\*|\Z)',
                  '', text, flags=re.M | re.S)
    # The GARCH equation appendix gives each pure AR model its own block.
    text = re.sub(r'^\*\*ARX\([12],0\)\.\*\*\n.*?(?=^\*\*ARX|\Z)',
                  '', text, flags=re.M | re.S)
    text = text.replace('Constant, ARX(1,0), ARX(1,1), ARX(2,0), ARX(2,1), and ARX(2,2)',
                        'Constant, ARX(1,1), ARX(2,1), and ARX(2,2)')
    text = text.replace('; ARX(1,0) and ARX(2,0) are pure autoregressions without the professional forecast', '')
    text = re.sub(r' ARX\(1,0\) and ARX\(2,0\) are pure\nautoregressions.*?inflation lags\.',
                  '', text, flags=re.S)
    text = text.replace('For ARX(1,0) and ARX(1,1)', 'For ARX(1,1)')
    text = text.replace('For ARX(2,0), ARX(2,1) and ARX(2,2)', 'For ARX(2,1) and ARX(2,2)')
    text = re.sub(r', ARX\([12],0\) [0-9.]+', '', text)
    text = text.replace('Six types of mean processes', 'Four types of mean processes')
    text = text.replace('six mean', 'four mean')
    text = text.replace('contains 2,400 saved results from 60,000 optimizations',
                        'contains 1,600 saved results from 40,000 optimizations')
    text = text.replace('216 reported model combinations', '144 reported model combinations')
    if re.fullmatch(r'BEGE_GARCH/([^/]+_BEGE|BEGE_EGARCH)/README(_monthly)?\.md', name):
        results = Path(name).parent / ('results_monthly' if name.endswith('_monthly.md') else 'results')
        with (ROOT / results / 'selection_diagnostics.csv').open() as f:
            rows = [r for r in csv.DictReader(f) if r['mean_type'] not in ARCHIVED_MEANS]
        true = lambda x: str(x).strip().lower() in {'true', '1', 'y'}
        converged = sum(true(r['success']) for r in rows)
        eligible = sum(true(r['selection_eligible']) for r in rows)
        text, count = re.subn(r'^Saved results: [\d,]+ in total, [\d,]+ (converged|successful), '
                             r'[\d,]+ eligible for best-model selection\.',
                             lambda m: f'Saved results for the four reported means: {len(rows):,} in total, '
                             f'{converged:,} {m[1]}, {eligible:,} eligible for best-model selection.',
                             text, flags=re.M)
        if count != 1:
            raise ValueError(f'{name}: BEGE result-count paragraph changed')
    if re.search(r'ARX\([12],\s*0\)|ARX_[12]_0|six mean|60,000|2,400', text):
        raise ValueError(f'{name}: archived mean or outdated search count remains in PDF text')
    return text
