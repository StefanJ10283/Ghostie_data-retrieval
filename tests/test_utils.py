"""
Unit tests for pure utility functions in DataRetrieval.py.

Level of abstraction: UNIT
These tests exercise business logic in complete isolation — no network calls,
no database connections, no HTTP layer. Each test invokes a single function
and asserts on its return value only.

Functions under test:
  - compute_hash(data)         : SHA-256 fingerprinting of a data list
  - make_business_key(...)     : normalised composite lookup key
  - floats_to_decimals(obj)    : recursive float → Decimal conversion
  - decimals_to_floats(obj)    : recursive Decimal → float/int conversion
"""
import hashlib
import json
from decimal import Decimal

import pytest

from DataRetrieval import (
    compute_hash,
    decimals_to_floats,
    floats_to_decimals,
    make_business_key,
)


# ── compute_hash ─────────────────────────────────────────────────────────────


class TestComputeHash:
    def test_returns_64_char_hex_string(self):
        result = compute_hash([{"text": "hello"}])
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_same_data_produces_same_hash(self):
        data = [{"title": "News article", "score": 1}]
        assert compute_hash(data) == compute_hash(data)

    def test_different_data_produces_different_hash(self):
        assert compute_hash([{"a": 1}]) != compute_hash([{"a": 2}])

    def test_empty_list_produces_valid_hash(self):
        result = compute_hash([])
        assert isinstance(result, str)
        assert len(result) == 64

    def test_key_order_does_not_affect_hash(self):
        data_a = [{"b": 2, "a": 1}]
        data_b = [{"a": 1, "b": 2}]
        assert compute_hash(data_a) == compute_hash(data_b)

    def test_matches_manual_sha256(self):
        data = [{"type": "review", "rating": 5}]
        expected = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        assert compute_hash(data) == expected

    def test_single_item_list(self):
        result = compute_hash([{"body": "Great place"}])
        assert isinstance(result, str)
        assert len(result) == 64

    def test_large_data_list(self):
        data = [{"id": i, "text": f"item {i}"} for i in range(100)]
        result = compute_hash(data)
        assert len(result) == 64


# ── make_business_key ─────────────────────────────────────────────────────────


class TestMakeBusinessKey:
    def test_lowercases_all_parts(self):
        result = make_business_key("Subway", "SYDNEY", "Restaurant")
        assert result == "subway_sydney_restaurant"

    def test_strips_leading_and_trailing_whitespace(self):
        result = make_business_key("  subway  ", "  sydney  ", "  restaurant  ")
        assert result == "subway_sydney_restaurant"

    def test_underscore_separator_between_parts(self):
        result = make_business_key("a", "b", "c")
        assert result == "a_b_c"

    def test_preserves_internal_spaces_in_name(self):
        result = make_business_key("KFC Australia", "New South Wales", "fast food")
        assert result == "kfc australia_new south wales_fast food"

    def test_already_lowercase_input_unchanged(self):
        result = make_business_key("mcdonald's", "brisbane", "fast food")
        assert result == "mcdonald's_brisbane_fast food"

    def test_numeric_characters_preserved(self):
        result = make_business_key("7-Eleven", "Perth", "convenience")
        assert result == "7-eleven_perth_convenience"

    def test_mixed_case_category(self):
        result = make_business_key("Woolworths", "Melbourne", "SUPERMARKET")
        assert result == "woolworths_melbourne_supermarket"


# ── floats_to_decimals ────────────────────────────────────────────────────────


class TestFloatsToDecimals:
    def test_float_converted_to_decimal(self):
        result = floats_to_decimals(3.14)
        assert isinstance(result, Decimal)
        assert result == Decimal("3.14")

    def test_integer_passes_through_unchanged(self):
        result = floats_to_decimals(42)
        assert result == 42
        assert isinstance(result, int)

    def test_string_passes_through_unchanged(self):
        result = floats_to_decimals("hello")
        assert result == "hello"

    def test_none_passes_through_unchanged(self):
        assert floats_to_decimals(None) is None

    def test_dict_float_values_converted(self):
        result = floats_to_decimals({"score": 0.75, "count": 5})
        assert isinstance(result["score"], Decimal)
        assert result["count"] == 5

    def test_list_of_floats_all_converted(self):
        result = floats_to_decimals([1.0, 2.5, 3])
        assert isinstance(result[0], Decimal)
        assert isinstance(result[1], Decimal)
        assert isinstance(result[2], int)

    def test_nested_dict_in_list_converted(self):
        data = [{"score": 0.9, "nested": {"value": 1.5}}]
        result = floats_to_decimals(data)
        assert isinstance(result[0]["score"], Decimal)
        assert isinstance(result[0]["nested"]["value"], Decimal)

    def test_zero_float_converted(self):
        result = floats_to_decimals(0.0)
        assert isinstance(result, Decimal)
        assert result == Decimal("0.0")


# ── decimals_to_floats ────────────────────────────────────────────────────────


class TestDecimalsToFloats:
    def test_whole_decimal_converts_to_int(self):
        result = decimals_to_floats(Decimal("5"))
        assert result == 5
        assert isinstance(result, int)

    def test_fractional_decimal_converts_to_float(self):
        result = decimals_to_floats(Decimal("3.14"))
        assert isinstance(result, float)
        assert abs(result - 3.14) < 1e-9

    def test_dict_decimal_values_converted(self):
        result = decimals_to_floats({"score": Decimal("0.5"), "label": "positive"})
        assert isinstance(result["score"], float)
        assert result["label"] == "positive"

    def test_list_elements_converted(self):
        result = decimals_to_floats([Decimal("1"), Decimal("2.5")])
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)

    def test_nested_structure_fully_converted(self):
        data = {"items": [{"value": Decimal("99"), "ratio": Decimal("0.5")}]}
        result = decimals_to_floats(data)
        assert result["items"][0]["value"] == 99
        assert isinstance(result["items"][0]["value"], int)

    def test_non_decimal_types_pass_through(self):
        assert decimals_to_floats("text") == "text"
        assert decimals_to_floats(42) == 42
        assert decimals_to_floats(None) is None

    def test_float_roundtrip_preserves_value(self):
        original = {"rating": 4.5, "count": 10, "label": "good"}
        result = decimals_to_floats(floats_to_decimals(original))
        assert result["rating"] == pytest.approx(4.5)
        assert result["count"] == 10
        assert result["label"] == "good"

    def test_zero_decimal_converts_to_int(self):
        result = decimals_to_floats(Decimal("0"))
        assert result == 0
        assert isinstance(result, int)
