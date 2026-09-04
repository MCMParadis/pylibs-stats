import logging

import numpy as np
import pytest

from pylibs.core.exceptions import ExpressionError
from pylibs.core.features.expression import evaluate_expression, split_ratio, validate_expression
from pylibs.core.features.identity import feature_identity_hash
from pylibs.core.features.peak_table import PeakEntry

TABLE = {
    "Fe438": PeakEntry(element="Fe", id="Fe438", peak=438.4, continuum=437.8),
    "Fe266-1": PeakEntry(element="Fe", id="Fe266-1", peak=266.6, continuum=265.3),
    "Fe266": PeakEntry(element="Fe", id="Fe266", peak=266.5, continuum=265.8),
}

WAVELENGTHS = np.array([200.0, 201.0, 202.0, 203.0, 204.0, 205.0])
EVAL_TABLE = {
    "A100": PeakEntry(element="A", id="A100", peak=200.0, continuum=None),
    "A101": PeakEntry(element="A", id="A101", peak=201.0, continuum=None),
    "A100-1": PeakEntry(element="A", id="A100-1", peak=202.0, continuum=None),
}
DATA = np.array([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]])


def test_valid_expression_logs_nothing(caplog):
    with caplog.at_level(logging.ERROR):
        validate_expression("Fe438 / Fe266", TABLE)

    assert caplog.records == []


def test_hyphenated_id_is_recognized_as_single_token(caplog):
    with caplog.at_level(logging.ERROR):
        validate_expression("Fe266-1 - Fe266", TABLE)

    assert caplog.records == []


def test_unknown_id_logs_error_without_raising(caplog):
    with caplog.at_level(logging.ERROR):
        validate_expression("Fe438-1 / Fe266", TABLE)

    assert len(caplog.records) == 1
    assert "Fe438-1" in caplog.records[0].message
    assert "Fe438-1 / Fe266" in caplog.records[0].message


def test_multiple_unknown_ids_each_log_once(caplog):
    with caplog.at_level(logging.ERROR):
        validate_expression("Foo123 / Bar456", TABLE)

    assert len(caplog.records) == 2


def test_evaluate_single_peak():
    result = evaluate_expression("A100", EVAL_TABLE, DATA, WAVELENGTHS)

    assert result == pytest.approx([10.0])


def test_evaluate_ratio():
    # +1 on both sides (see _eval's Div handling): (20+1)/(10+1), not 20/10
    result = evaluate_expression("A101 / A100", EVAL_TABLE, DATA, WAVELENGTHS)

    assert result == pytest.approx([21 / 11])


def test_evaluate_subtraction_between_hyphenated_ids():
    result = evaluate_expression("A100-1 - A101", EVAL_TABLE, DATA, WAVELENGTHS)

    assert result == pytest.approx([10.0])


def test_evaluate_addition_and_multiplication():
    assert evaluate_expression("A100 + A101", EVAL_TABLE, DATA, WAVELENGTHS) == pytest.approx(
        [30.0]
    )
    assert evaluate_expression("A100 * A101", EVAL_TABLE, DATA, WAVELENGTHS) == pytest.approx(
        [200.0]
    )


def test_evaluate_log_and_exp():
    result = evaluate_expression("log(A101)", EVAL_TABLE, DATA, WAVELENGTHS)
    assert result == pytest.approx([np.log(20.0)])

    result = evaluate_expression("exp(A100)", EVAL_TABLE, DATA, WAVELENGTHS)
    assert result == pytest.approx([np.exp(10.0)])


def test_evaluate_nested_parens():
    # numerator (A100-1 - A101) = 30-20 = 10, denominator (A100) = 10, so the
    # +1 smoothing gives (10+1)/(10+1) == 1.0 too -- unchanged here only
    # because numerator and denominator were already equal
    result = evaluate_expression("(A100-1 - A101) / A100", EVAL_TABLE, DATA, WAVELENGTHS)

    assert result == pytest.approx([1.0])


def test_evaluate_unknown_id_logs_and_nan_propagates(caplog):
    with caplog.at_level(logging.ERROR):
        result = evaluate_expression("A999 / A100", EVAL_TABLE, DATA, WAVELENGTHS)

    assert np.all(np.isnan(result))
    assert len(caplog.records) == 1
    assert "A999" in caplog.records[0].message


def test_evaluate_repeated_unknown_id_only_logs_once(caplog):
    with caplog.at_level(logging.ERROR):
        evaluate_expression("A999 / A999", EVAL_TABLE, DATA, WAVELENGTHS)

    assert len(caplog.records) == 1


def test_evaluate_rejects_unsupported_syntax():
    # power operator, unregistered functions, and bare numeric literals are
    # all outside the expression mini-language's supported grammar
    for expression in ["A100 ** 2", "sqrt(A100)", "A100 * 2"]:
        with pytest.raises(ExpressionError):
            evaluate_expression(expression, EVAL_TABLE, DATA, WAVELENGTHS)


def test_split_ratio_returns_none_for_non_ratio_expressions():
    assert split_ratio("Fe438") is None
    assert split_ratio("Fe438 + Li670") is None


def test_split_ratio_simple_ratio():
    assert split_ratio("Fe438/Li670") == ("Fe438", "Li670")
    assert split_ratio("Fe438 / Li670") == ("Fe438", "Li670")  # spacing doesn't matter


def test_split_ratio_nested_parenthesized_ratio():
    # the surrounding parens are grouping only -- the AST node's own span
    # (and thus the restored sub-expression) doesn't include them, though the
    # result is an equivalent expression once re-parsed.
    expression = "(Fe438/Li670)/(Fe438/Li670+P214/Li670)"
    assert split_ratio(expression) == ("Fe438/Li670", "Fe438/Li670+P214/Li670")


def test_split_ratio_chained_division_splits_at_outermost():
    # left-associative: (Fe438/Li670)/Na589 -- the AST root's right side is
    # Na589, so that's the denominator.
    assert split_ratio("Fe438/Li670/Na589") == ("Fe438/Li670", "Na589")


def test_feature_identity_hash_same_expression_and_table_matches():
    assert feature_identity_hash("Fe438 / Fe266", TABLE) == feature_identity_hash(
        "Fe438 / Fe266", TABLE
    )


def test_feature_identity_hash_changes_with_expression():
    assert feature_identity_hash("Fe438", TABLE) != feature_identity_hash("Fe266", TABLE)


def test_feature_identity_hash_changes_with_peak_table_entry():
    # same expression, same peak id -- but the peak table's own definition of
    # that id changed (e.g. a corrected wavelength), so the hash must too
    moved_table = {
        **TABLE,
        "Fe438": PeakEntry(element="Fe", id="Fe438", peak=438.5, continuum=437.8),
    }
    assert feature_identity_hash("Fe438", TABLE) != feature_identity_hash("Fe438", moved_table)


def test_feature_identity_hash_unknown_peak_id_still_hashes():
    # an expression referencing a peak id not in this table is a valid (if
    # degenerate) identity -- shouldn't raise, and differs from a table
    # where that id IS known
    assert feature_identity_hash("Unknown999", TABLE) != feature_identity_hash(
        "Unknown999",
        {
            **TABLE,
            "Unknown999": PeakEntry(element="U", id="Unknown999", peak=999.0, continuum=None),
        },
    )
