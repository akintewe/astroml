"""Comprehensive data quality validation tests.

Covers completeness, uniqueness, consistency, and hash-integrity checks
across the astroml.validation pipeline.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from astroml.validation import dedupe, integrity, validator
from astroml.validation.validator import CorruptionType


class TestCompleteness:
    """Required fields must be present and non-null."""

    def test_all_required_fields_present(self, valid_transactions):
        v = validator.TransactionValidator(required_fields={"id", "source_account"})
        for tx in valid_transactions:
            result = v.validate(tx)
            assert result.is_valid, f"Expected valid: {tx}"

    def test_missing_field_flagged(self):
        v = validator.TransactionValidator(required_fields={"id", "source_account", "amount"})
        tx = {"id": "tx_missing", "source_account": "GABC"}
        result = v.validate(tx)
        assert not result.is_valid
        assert any(e.error_type == CorruptionType.MISSING_FIELD for e in result.errors)

    def test_null_required_field_flagged(self):
        v = validator.TransactionValidator(required_fields={"id"})
        result = v.validate({"id": None, "source_account": "GABC"})
        assert not result.is_valid

    def test_empty_dict_fails_required_fields(self):
        v = validator.TransactionValidator(required_fields={"id"})
        assert not v.validate({}).is_valid

    def test_batch_completeness_surfaces_invalid_rows(self, transactions_with_missing_fields):
        v = validator.TransactionValidator(required_fields={"id", "source_account"})
        results = v.validate_batch(transactions_with_missing_fields)
        assert len(results) == len(transactions_with_missing_fields)
        assert all(not r.is_valid for r in results)

    def test_non_dict_flagged_as_malformed(self):
        v = validator.TransactionValidator()
        for bad in [None, "string", 42, [1, 2]]:
            result = v.validate(bad)
            assert not result.is_valid
            assert any(e.error_type == CorruptionType.MALFORMED_STRUCTURE for e in result.errors)


class TestUniqueness:
    """Duplicate transactions must be detected reliably."""

    def test_clean_batch_has_no_duplicates(self, valid_transactions):
        result = dedupe.deduplicate(valid_transactions)
        assert len(result.duplicates) == 0
        assert len(result.unique) == len(valid_transactions)

    def test_exact_duplicate_detected(self, duplicate_transactions):
        result = dedupe.deduplicate(duplicate_transactions)
        assert len(result.duplicates) == 1
        assert len(result.unique) == len(duplicate_transactions) - 1

    def test_first_occurrence_is_kept(self):
        tx_a = {"id": "tx_a", "amount": 10}
        tx_b = {"id": "tx_b", "amount": 20}
        result = dedupe.deduplicate([tx_a, tx_b, {**tx_a}])
        assert len(result.unique) == 2
        assert len(result.duplicates) == 1

    def test_deduplicator_stateful_across_batches(self):
        d = dedupe.Deduplicator()
        tx = {"id": "tx_1", "amount": 100}
        r1 = d.process([tx])
        r2 = d.process([{**tx}])
        assert len(r1.unique) == 1
        assert len(r2.duplicates) == 1

    def test_deduplicator_reset_clears_state(self):
        d = dedupe.Deduplicator()
        tx = {"id": "tx_reset", "amount": 50}
        d.process([tx])
        d.reset()
        result = d.process([{**tx}])
        assert len(result.unique) == 1
        assert len(result.duplicates) == 0

    def test_integrity_validator_detects_duplicates(self):
        v = integrity.IntegrityValidator(required_fields={"id"})
        result = v.process([{"id": "dup"}, {"id": "dup"}])
        assert len(result.duplicates) == 1
        assert len(result.valid) == 1


class TestConsistency:
    """Field types must match declared expectations."""

    def test_correct_types_pass(self):
        v = validator.TransactionValidator(field_types={"id": str, "amount": float})
        assert v.validate({"id": "tx_001", "amount": 100.0}).is_valid

    def test_wrong_type_flagged(self):
        v = validator.TransactionValidator(field_types={"id": str})
        result = v.validate({"id": 12345})
        assert not result.is_valid
        assert any(e.error_type == CorruptionType.INVALID_TYPE for e in result.errors)

    def test_multiple_type_violations_all_reported(self):
        v = validator.TransactionValidator(field_types={"id": str, "amount": float})
        result = v.validate({"id": 123, "amount": "not_a_float"})
        type_errors = [e for e in result.errors if e.error_type == CorruptionType.INVALID_TYPE]
        assert len(type_errors) == 2

    def test_batch_surfaces_type_inconsistent_rows(self):
        v = validator.TransactionValidator(field_types={"id": str})
        results = v.validate_batch([{"id": "ok_1"}, {"id": "ok_2"}, {"id": 999}])
        assert sum(1 for r in results if not r.is_valid) == 1


class TestAccuracy:
    """Hash-based integrity checks catch tampered data."""

    def test_correct_hash_passes(self):
        from astroml.validation.hashing import compute_transaction_hash

        v = validator.TransactionValidator()
        tx = {"id": "tx_hash_ok", "payload": "original"}
        result = v.validate(tx, stored_hash=compute_transaction_hash(tx))
        assert result.is_valid

    def test_wrong_hash_flagged(self):
        v = validator.TransactionValidator()
        tx = {"id": "tx_hash_bad", "payload": "tampered"}
        result = v.validate(tx, stored_hash="00000000_wrong_hash")
        assert not result.is_valid
        assert any(e.error_type == CorruptionType.HASH_MISMATCH for e in result.errors)

    def test_integrity_processor_flags_corrupted_rows(self):
        v = integrity.IntegrityValidator(required_fields={"id", "amount"})
        result = v.process([{"id": "ok", "amount": 50}, {"id": "bad"}])
        assert len(result.corrupted) == 1
        assert result.corrupted[0]["id"] == "bad"


class TestDataQualityPipeline:
    """End-to-end data quality across completeness + uniqueness + integrity."""

    def test_clean_batch_passes_full_pipeline(self, valid_transactions):
        v = validator.TransactionValidator(required_fields={"id", "source_account"})
        assert all(r.is_valid for r in v.validate_batch(valid_transactions))
        assert len(dedupe.deduplicate(valid_transactions).duplicates) == 0
        assert integrity.IntegrityValidator(required_fields={"id"}).process(valid_transactions).is_valid

    def test_strict_mode_raises_on_first_corruption(self):
        v = integrity.IntegrityValidator(required_fields={"id"}, strict=True)
        with pytest.raises(integrity.IntegrityError):
            v.process([{"id": "ok"}, {}])

    def test_filter_returns_only_clean_unique_rows(self):
        txs = [
            {"id": "tx_1", "amount": 100},
            {"id": "tx_2", "amount": 200},
            {"id": "tx_1", "amount": 100},
        ]
        valid = integrity.filter_valid_transactions(txs)
        assert len(valid) == 2

    def test_verify_integrity_false_on_duplicates(self, duplicate_transactions):
        v = integrity.IntegrityValidator(required_fields={"id"})
        assert not v.verify_integrity(duplicate_transactions)

    def test_verify_integrity_true_for_clean_batch(self, valid_transactions):
        v = integrity.IntegrityValidator(required_fields={"id"})
        assert v.verify_integrity(valid_transactions)

    def test_check_integrity_convenience_function(self, valid_transactions):
        result = integrity.check_integrity(valid_transactions)
        assert result.is_valid
