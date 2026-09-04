"""Feature expression handling: `ADDFEATURE` expressions combine peak ids
with `+ - * /` and `log()`/`exp()`. Peak ids can contain hyphens (e.g.
`Fe266-1`), so `-` is only ever read as subtraction *between* two peak-id
tokens -- never a literal number -- which is what makes id-shaped tokens
unambiguous to find directly, before any real parsing happens.

Every division adds 1 to both sides first (see `_eval`'s `ast.Div` branch) --
a ratio's numerator/denominator are raw intensities that can be genuinely
zero for real (mostly-background) LIBS data, so this avoids a divide-by-zero
(and, for a `log(ratio)` composite, `log(0)`) without aborting the batch.
"""

import ast
import re
import warnings

import numpy as np

from pylibs.core.exceptions import ExpressionError
from pylibs.core.features.extraction import extract_peak
from pylibs.core.features.peak_table import PeakTable
from pylibs.core.logging import get_logger

logger = get_logger(__name__)

_PEAK_ID_PATTERN = re.compile(r"[A-Za-z]+[0-9]+(?:-[0-9]+)?")
_ALLOWED_FUNCS = {"log": np.log, "exp": np.exp}
_PLACEHOLDER_PATTERN = re.compile(r"_peak\d+")


def validate_expression(expression: str, peak_table: PeakTable) -> None:
    """Log an error (without raising) for every peak-id-shaped token in
    `expression` that isn't in `peak_table` -- most likely a typo."""
    for match in _PEAK_ID_PATTERN.finditer(expression):
        token = match.group(0)
        if token not in peak_table:
            logger.error(
                "Unknown peak id %r in expression %r -- check for a typo, "
                "or add it to the peak table.",
                token,
                expression,
            )


def referenced_peak_ids(expression: str) -> list[str]:
    """Every distinct peak-id-shaped token in `expression`, in first-seen
    order -- regardless of whether it's actually in any peak table (see
    `features.identity.feature_identity_hash`, the one caller that cares
    about unknown tokens too)."""
    seen: dict[str, None] = {}
    for match in _PEAK_ID_PATTERN.finditer(expression):
        seen.setdefault(match.group(0), None)
    return list(seen)


def peaks_in_range(expression: str, peak_table: PeakTable, wavelengths: np.ndarray) -> list[str]:
    """Peak ids referenced in `expression` (via `peak_table` lookup) whose
    `peak` or `continuum` wavelength falls outside `wavelengths`' range.
    Unknown tokens (not in `peak_table` at all) aren't reported here -- that's
    a typo, already handled by `validate_expression`/`evaluate_expression`'s
    own NaN-degrade path, not a range problem."""
    wl_min, wl_max = float(wavelengths.min()), float(wavelengths.max())
    out_of_range = []
    for match in _PEAK_ID_PATTERN.finditer(expression):
        token = match.group(0)
        entry = peak_table.get(token)
        if entry is None:
            continue
        if not wl_min <= entry.peak <= wl_max:
            out_of_range.append(token)
        elif entry.continuum is not None and not wl_min <= entry.continuum <= wl_max:
            out_of_range.append(token)
    return out_of_range


def _rewrite_ids(expression: str) -> tuple[str, dict[str, str]]:
    """Replace every peak-id-shaped token with a safe placeholder name, so the
    result is valid Python syntax (peak ids may contain hyphens, e.g. Fe266-1)."""
    placeholders: dict[str, str] = {}

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        placeholder = f"_peak{len(placeholders)}"
        placeholders[placeholder] = token
        return placeholder

    rewritten = _PEAK_ID_PATTERN.sub(_replace, expression)
    return rewritten, placeholders


def split_ratio(expression: str) -> tuple[str, str] | None:
    """If `expression`'s outermost operation is a division, return its
    (numerator, denominator) as sub-expressions in the original syntax --
    e.g. "Fe438 / Li670" -> ("Fe438", "Li670"), and
    "(Fe438/Li670)/(Fe438/Li670+P214/Li670)" ->
    ("(Fe438/Li670)", "(Fe438/Li670+P214/Li670)"). Uses the same parser as
    `evaluate_expression`, so parentheses (not just the first "/") determine
    what's outermost. Returns None for a non-ratio expression (a bare peak
    id, a sum, etc.) -- there's nothing to split."""
    rewritten, placeholders = _rewrite_ids(expression)
    try:
        tree = ast.parse(rewritten, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Can't parse expression {expression!r}: {exc}") from exc

    root = tree.body
    if not (isinstance(root, ast.BinOp) and isinstance(root.op, ast.Div)):
        return None

    def _restore(node: ast.AST) -> str:
        segment = ast.get_source_segment(rewritten, node)
        if segment is None:
            raise ExpressionError(f"Can't isolate a sub-expression of {expression!r}")
        return _PLACEHOLDER_PATTERN.sub(lambda m: placeholders[m.group(0)], segment)

    return _restore(root.left), _restore(root.right)


def evaluate_expression(
    expression: str, peak_table: PeakTable, data: np.ndarray, wavelengths: np.ndarray
) -> np.ndarray:
    """Evaluate `expression` (peak ids combined with `+ - * /` and
    `log()`/`exp()`) for every spectrum in `data` (shape (n, n_pix)).
    Returns an array of shape (n,).

    An unknown peak id is logged (like `validate_expression`) and evaluates
    to NaN for that term, matching how `extract_peak` handles an out-of-range
    peak -- it doesn't abort the whole batch. Malformed expressions (bad
    syntax, unsupported operators/functions, numeric literals) raise
    `ExpressionError` instead, since no per-spectrum fallback makes sense
    for those."""
    rewritten, placeholders = _rewrite_ids(expression)

    try:
        tree = ast.parse(rewritten, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Can't parse expression {expression!r}: {exc}") from exc

    cache: dict[str, np.ndarray] = {}

    def _peak_value(placeholder: str) -> np.ndarray:
        token = placeholders[placeholder]
        if token not in cache:
            entry = peak_table.get(token)
            if entry is None:
                logger.error(
                    "Unknown peak id %r in expression %r -- check for a typo, "
                    "or add it to the peak table.",
                    token,
                    expression,
                )
                cache[token] = np.full(data.shape[0], np.nan)
            else:
                cache[token] = extract_peak(data, wavelengths, entry)
        return cache[token]

    def _eval(node: ast.AST) -> np.ndarray:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Name):
            return _peak_value(node.id)
        if isinstance(node, ast.BinOp):
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                # +1 on both sides -- a ratio's numerator/denominator are raw
                # intensities that can be genuinely zero for a real (mostly-
                # background) LIBS raster, which would otherwise divide by
                # zero here or feed log(0) in a log(ratio) composite
                return (left + 1) / (right + 1)
            raise ExpressionError(
                f"Unsupported operator {type(node.op).__name__!r} in expression {expression!r}"
            )
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in _ALLOWED_FUNCS
                and len(node.args) == 1
                and not node.keywords
            ):
                arg = _eval(node.args[0])
                # e.g. log(0) for a scan whose numerator peak is genuinely
                # zero -- expected for real spectroscopy data (mostly-
                # background rasters), not a bug -- yields -inf/nan for that
                # scan rather than aborting the batch (matches how an
                # unknown peak id degrades to NaN above), so just log it
                # at DEBUG instead of letting numpy's RuntimeWarning reach
                # the console
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = _ALLOWED_FUNCS[node.func.id](arg)
                for warning in caught:
                    logger.debug(
                        "%s() in expression %r: %s", node.func.id, expression, warning.message
                    )
                return result
            raise ExpressionError(f"Unsupported function call in expression {expression!r}")
        raise ExpressionError(
            f"Unsupported syntax ({type(node).__name__}) in expression {expression!r}"
        )

    return _eval(tree)
