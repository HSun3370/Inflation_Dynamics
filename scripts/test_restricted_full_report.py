"""Checks for the deferred report's criterion comparisons and missing-input guard."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from scripts import refine_restricted_full_report as report


class ReportTests(unittest.TestCase):
    def test_missing_search_does_not_publish(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(report, 'ROOT', Path(tmp)):
            with self.assertRaises(FileNotFoundError):
                report.read_results('SymmetricP_BEGE', '')
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_comparison_qualifies_incomplete_unrestricted_search(self):
        rows = []
        for frequency in report.FREQUENCIES:
            for folder, (_, label) in report.SPECS.items():
                for mean in report.MEANS:
                    rows.append(dict(frequency=frequency, specification=folder, label=label,
                        mean=mean, loglik=10., AIC=8., BIC=20., delta_loglik=.2,
                        delta_AIC=-2.4, delta_BIC=-5.4))
        text = report.comparison_text(pd.DataFrame(rows))
        self.assertTrue(text.startswith(report.PREAMBLE))
        self.assertIn('incomplete unrestricted numerical search', text)
        self.assertIn('lower AIC than Full BEGE in 4 of the four means', text)
        self.assertIn('| 10.0000 | 0.2000 | -2.4000 | -5.4000 |', text)
        self.assertNotIn('|\n\n|', text)
        self.assertNotIn('ARX(1,0)', text)


if __name__ == '__main__':
    unittest.main()
