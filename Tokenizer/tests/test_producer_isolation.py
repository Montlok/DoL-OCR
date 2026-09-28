import json
from pathlib import Path

import pytest

from Tokenizer.pretraining.data_contract import (
    PRETRAINING_PRODUCER_GENERIC_BUILDER,
    PRETRAINING_PRODUCER_OCR_ALIGN_TEXT,
    build_pretraining_data_contract,
    canonical_json_sha256,
    file_sha256,
    pretraining_producer_algorithm_contract,
    validate_pretraining_data_contract_payload,
)
from Tokenizer.tests.test_pretraining_builder import build_smoke_bundle
from Tokenizer.unified.contract import tokenizer_algorithm_contract, tokenizer_bundle_contract


# Released main's identity at e57cb5e. Producer-only windowing must not
# invalidate unchanged OCR encoding or checkpoints through this fingerprint.
BASE_TOKENIZER_FILES = "018a5b87d6df0346fa2fddec697ed9aeb8eb97fb01a1293de747d913d231fd12"
BASE_OCR_PRODUCER = {
    "contract_version": 1,
    "source_file": "scripts/build_text_rows_from_align.py",
    "source_sha256": "9ae9c3104011a4e5f089c8212a491f9a0ef5d06ce6b3e5e218f100b7411e2922",
}


def test_windowing_keeps_shared_tokenizer_and_ocr_producer_identity():
    assert tokenizer_algorithm_contract()["files_canonical_sha256"] == BASE_TOKENIZER_FILES
    assert pretraining_producer_algorithm_contract(PRETRAINING_PRODUCER_OCR_ALIGN_TEXT) == BASE_OCR_PRODUCER


def test_generic_fingerprint_covers_the_complete_producer_module():
    root = Path(__file__).resolve().parents[2]
    identity = pretraining_producer_algorithm_contract(PRETRAINING_PRODUCER_GENERIC_BUILDER)
    assert set(identity) == {"contract_version", "source_file", "source_sha256"}
    assert identity["source_file"] == "Tokenizer/pretraining/producer.py"
    assert identity["source_sha256"] == file_sha256(root / identity["source_file"])


def test_unchanged_ocr_receipt_remains_valid_and_old_generic_is_rejected(tmp_path):
    build_smoke_bundle(str(tmp_path))
    bundle = tokenizer_bundle_contract(tmp_path / "bundle")
    algorithm = tokenizer_algorithm_contract()
    shard = tmp_path / "data.jsonl"
    shard.write_text(json.dumps({"input_ids": [2, 3], "labels": [-100, 3]}) + "\n")
    for kind in (PRETRAINING_PRODUCER_OCR_ALIGN_TEXT, PRETRAINING_PRODUCER_GENERIC_BUILDER):
        receipt = build_pretraining_data_contract(
            shard, producer_kind=kind, tokenizer_bundle=bundle, tokenizer_algorithm=algorithm,
        )
        if kind == PRETRAINING_PRODUCER_OCR_ALIGN_TEXT:
            receipt["producer_algorithm"] = BASE_OCR_PRODUCER
        else:
            receipt["producer_algorithm"] = {
                "contract_version": 1,
                "source_file": "Tokenizer/tools/build_pretraining_data.py",
                "source_sha256": "8a3d9a2fa46f6a6d5a4ba77b629a41d802ed2c9da301efd6dbb55ee0a28d884d",
            }
        receipt.pop("contract_canonical_sha256")
        receipt["contract_canonical_sha256"] = canonical_json_sha256(receipt)
        kwargs = {"tokenizer_bundle": bundle, "tokenizer_algorithm": algorithm}
        if kind == PRETRAINING_PRODUCER_OCR_ALIGN_TEXT:
            validate_pretraining_data_contract_payload(receipt, **kwargs)
        else:
            with pytest.raises(ValueError, match="producer_algorithm differs"):
                validate_pretraining_data_contract_payload(receipt, **kwargs)
