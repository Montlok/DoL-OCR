# -*- coding: utf-8 -*-

"""Guards for ``Model.config`` special-token / segment contracts (L2)."""

from __future__ import annotations

import unittest
from dataclasses import asdict

from Model.config import (
    _FALLBACK_SEGMENT,
    _FALLBACK_SPECIAL_TOKENS,
    _REMOVED_INACTIVE_RDT_CONFIG_FIELDS,
    rdt_config_from_dict,
    two_stage_tiny_config,
)


class SpecialTokenFallbackConsistencyTest(unittest.TestCase):
    def test_fallback_matches_canonical_vocab(self) -> None:
        try:
            from Tokenizer.unified.vocab import SEGMENT, SPECIAL_TOKENS
        except ImportError:  # pragma: no cover
            self.skipTest("Tokenizer package not importable")

        self.assertEqual(
            _FALLBACK_SPECIAL_TOKENS,
            dict(SPECIAL_TOKENS),
            "Model.config fallback SPECIAL_TOKENS drifted from "
            "Tokenizer/unified/vocab.py — keep them in sync.",
        )
        self.assertEqual(
            _FALLBACK_SEGMENT,
            {k: tuple(v) for k, v in SEGMENT.items()},
            "Model.config fallback SEGMENT drifted from "
            "Tokenizer/unified/vocab.py — keep them in sync.",
        )


class RemovedExperimentCompatibilityTest(unittest.TestCase):
    def test_reviewed_inactive_metadata_restores_current_config(self) -> None:
        current = two_stage_tiny_config()
        legacy = {**asdict(current), **_REMOVED_INACTIVE_RDT_CONFIG_FIELDS}
        self.assertEqual(rdt_config_from_dict(legacy), current)

    def test_removed_active_experiments_fail_closed(self) -> None:
        current = asdict(two_stage_tiny_config())
        for field, active in (
            ("use_act", True),
            ("use_mol", True),
            ("recurrent_random_r", True),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "reviewed inactive value"
            ):
                rdt_config_from_dict({**current, field: active})
        with self.assertRaisesRegex(ValueError, "segmented"):
            rdt_config_from_dict({**current, "core_type": "segmented"})


if __name__ == "__main__":
    unittest.main()
