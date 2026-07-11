"""Regression tests for deterministic caption-audit semantics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.grounding_audit import audit_caption as audit_grounding
from eval.quality_audit import audit_caption as audit_quality
from eval.self_check import _tech_hits


class AuditSemanticsTests(unittest.TestCase):
    def test_low_taste_terms_use_word_boundaries(self) -> None:
        caption = (
            "Cars cross the boulevard in a serious demonstration of bidirectional traffic."
        )

        self.assertEqual(audit_quality("v1", "sarcastic", caption), [])

    def test_everyday_frames_verb_is_not_tech_bleed(self) -> None:
        caption = "A leafy plant frames the desk, a thrilling triumph of office decoration."

        self.assertEqual(audit_quality("v3", "sarcastic", caption), [])

    def test_tech_style_allows_figurative_code_terms(self) -> None:
        caption = (
            "A woman types at a keyboard while tangled cables resemble legacy code."
        )

        self.assertEqual(audit_grounding("v3", "humorous_tech", caption), [])

    def test_everyday_staging_is_not_classified_as_tech_jargon(self) -> None:
        caption = "The vegetable is staging a heroic last stand against the knife."

        self.assertEqual(_tech_hits(caption), [])
        self.assertEqual(audit_quality("7963468", "sarcastic", caption), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
