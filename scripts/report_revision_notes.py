"""Attach the advisor-response footnotes to the report edition, after editing.

The responses live outside generated estimation reports so collection cannot
erase them. This module changes presentation only, never estimates or data.
"""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / 'RevisionResponses' / 'responses.json'
PREAMBLE = '```{raw:typst}\n#set page(margin: auto)\n```\n\n'


def load_notes():
    notes = json.loads(NOTES.read_text())
    expected = {f'Q{i}' for i in range(1, 12)} | {f'B{i}' for i in range(1, 6)}
    ids = [note['id'] for note in notes]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError('Revision responses must cover Q1–Q11 and B1–B5 exactly once')
    return notes


def add_revision_notes(name, text):
    for note in load_notes():
        # "footnote": false keeps a response in the audit checklist but out of the PDF.
        if name != note['file'] or not note.get('footnote', True):
            continue
        body = f"**Response to {note['id']} — {note['topic']}.** {note['response']}"
        if 'existing_footnote' in note:
            # Consolidate the old EWMA explanation into its response footnote.
            label = note['existing_footnote']
            pattern = rf'^\[\^{re.escape(label)}\]: .*?(?=\n\n|\Z)'
            text, count = re.subn(pattern, lambda _: f'[^{label}]: {body}',
                                  text, flags=re.M | re.S)
            if count != 1 or f'[^{label}]' not in text:
                raise ValueError(f'{name}: existing footnote {label} not found once')
            continue
        anchor = note['anchor']
        if text.count(anchor) != note.get('anchor_count', 1):
            raise ValueError(f"{name}: response {note['id']} anchor changed")
        label = 'revision-' + note['id'].lower()
        if 'lead' in note:
            replacement = anchor + '\n' + note['lead'] + f'[^{label}]\n'
        else:
            replacement = anchor + f'[^{label}]'
        text = text.replace(anchor, replacement, 1)
        text = text.rstrip() + f'\n\n[^{label}]: {body}\n'
    return text


def write_checklist():
    notes = load_notes()
    audit_only = [n['id'] for n in notes if not n.get('footnote', True)]
    if audit_only:
        placement = ('The PDF places each response in a footnote beside its relevant '
                     'discussion, except ' + ', '.join(audit_only[:-1])
                     + (', and ' if len(audit_only) > 2 else ' and ' if len(audit_only) == 2 else '')
                     + audit_only[-1] + ', which are recorded only here. ')
    else:
        placement = 'The PDF places each response in a footnote beside its relevant discussion. '
    parts = [PREAMBLE + '# Revision response audit',
             'Reviewed on 2026-11-14. Q1–Q11 refer to Questions/Remarks; B1–B5 '
             'refer to Bigger Picture/Finalization in the supplied letter. ' + placement +
             'This audit distinguishes completed work, explanations added in this '
             'revision, changes from the proposed method, and remaining work. '
             'The initial audit used saved estimates. Subsequent restricted Full BEGE '
             'estimation is tracked in Q11 and its supporting run records.']
    table = ['| Item | Concern or suggestion | Status | Remaining work |',
             '|---|---|---|---|']
    for n in notes:
        table.append(f"| {n['id']} | {n['topic']} | {n['status']} | {n['remaining']} |")
    parts.append('\n'.join(table))
    parts.append('\n## Responses and supporting evidence')
    for n in notes:
        parts.extend([f"\n### {n['id']}: {n['topic']}", n['response'],
                      '**Evidence:** ' + '; '.join(
                          f'[{p}](../{p})' if not p.startswith('/') else f'[{Path(p).name}]({p})'
                          for p in n['evidence']) + '.'])
    parts.extend(['\n## Maintaining these notes',
                  'Edit `responses.json`, then run `python scripts/build_report.py`. '
                  'The builder checks each insertion point and regenerates this checklist. '
                  'Re-audit numerical statements when results change. The saved archived '
                  'Full BEGE candidate was evaluated with the current collector on the '
                  'current quarterly data; its parameters and diagnostics are in '
                  '`archived_full_candidate_check.csv`. It is evidence for the outstanding '
                  'search-reconciliation task, not a replacement reported estimate.'])
    (NOTES.parent / 'README.md').write_text('\n\n'.join(parts) + '\n')


if __name__ == '__main__':
    write_checklist()
