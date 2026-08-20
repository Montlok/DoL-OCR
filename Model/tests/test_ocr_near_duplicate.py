# -*- coding: utf-8 -*-

from __future__ import annotations

import io
import unittest

from PIL import Image, ImageDraw

from Model.ocr.near_duplicate import (
    cluster_perceptual_hashes,
    near_duplicate_cluster_id,
    perceptual_hash,
    phash_hamming_distance,
)


def _encoded(format_name: str, *, shifted: bool = False) -> bytes:
    image = Image.new("L", (96, 128), 255)
    draw = ImageDraw.Draw(image)
    offset = 2 if shifted else 0
    draw.rectangle((25 + offset, 12, 38 + offset, 112), fill=0)
    draw.rectangle((55 + offset, 28, 68 + offset, 118), fill=50)
    buffer = io.BytesIO()
    image.save(buffer, format=format_name, quality=85)
    return buffer.getvalue()


class OCRNearDuplicateTest(unittest.TestCase):
    def test_reencoding_and_small_shift_are_close_and_clustered(self) -> None:
        hashes = {
            "png": perceptual_hash(_encoded("PNG")),
            "jpeg": perceptual_hash(_encoded("JPEG")),
            "shifted": perceptual_hash(_encoded("PNG", shifted=True)),
        }
        self.assertLessEqual(phash_hamming_distance(hashes["png"], hashes["jpeg"]), 4)
        self.assertLessEqual(phash_hamming_distance(hashes["png"], hashes["shifted"]), 8)
        clusters = cluster_perceptual_hashes(hashes, max_hamming_distance=8)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].member_ids, ("jpeg", "png", "shifted"))
        self.assertEqual(
            clusters[0].cluster_id,
            near_duplicate_cluster_id(("jpeg", "png", "shifted")),
        )

    def test_content_addressed_cluster_ids_do_not_renumber_unrelated_groups(self):
        base = cluster_perceptual_hashes(
            {"a": "0" * 16, "z": "f" * 16},
            max_hamming_distance=0,
        )
        extended = cluster_perceptual_hashes(
            {"a": "0" * 16, "m": "a" * 16, "z": "f" * 16},
            max_hamming_distance=0,
        )
        base_ids = {cluster.member_ids: cluster.cluster_id for cluster in base}
        extended_ids = {
            cluster.member_ids: cluster.cluster_id for cluster in extended
        }
        self.assertEqual(base_ids[("a",)], extended_ids[("a",)])
        self.assertEqual(base_ids[("z",)], extended_ids[("z",)])

    def test_invalid_hash_and_bounded_review_set_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "16 lowercase"):
            phash_hamming_distance("bad", "0" * 16)
        with self.assertRaisesRegex(ValueError, "max_records"):
            cluster_perceptual_hashes(
                {"a": "0" * 16, "b": "f" * 16},
                max_hamming_distance=1,
                max_records=1,
            )


if __name__ == "__main__":
    unittest.main()
