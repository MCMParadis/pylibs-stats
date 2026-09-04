"""Sample-filename schemes: derive a sample's `sample_name` (the correlations
join key, see `sample_properties`) and any other structured fields from the
filename itself, so registering a directory doesn't mean restating per-file
metadata that's already encoded in the name.

A scheme is a template string over `{field}` placeholders, not a registry
entry or a regex, so a new filename layout is a different *argument* rather
than new code -- which is what lets a future GUI expose it as an editable
text field. `{sample_name}` is the one required field; every other field is
carried along in the parse result (and stored as `SampleConfig.name_parts`).

    "{sample_name}_{layer}"                        (the default)
        "sample1_2"            -> sample_name="sample1", layer="2"
    "{sample_name}_{param}-{info1}_{layer}"
        "sampleA_temp300-run2_1"
                               -> sample_name="sampleA", param="temp300",
                                  info1="run2", layer="1"

Fields are greedy left-to-right, so where a stem could split several ways the
*leftmost* field absorbs the ambiguity. That's what makes the default scheme
split at the LAST underscore -- `sample_name` is everything before the final
`_<layer>`, so "batch2_sample1_3" reads as sample_name="batch2_sample1",
layer="3" rather than the other way around.
"""

import re

from pylibs.core.exceptions import FilenameSchemeError

DEFAULT_SCHEME = "{sample_name}_{layer}"
SAMPLE_NAME_FIELD = "sample_name"

_FIELD_PATTERN = re.compile(r"\{([^{}]*)\}")


def compile_scheme(scheme: str) -> re.Pattern[str]:
    """Compile a scheme template into an anchored regex whose named groups are
    the template's fields. Raises `FilenameSchemeError` if the template is
    malformed -- callers should compile once, up front, so a bad template
    fails before any file is touched."""
    literals_only = _FIELD_PATTERN.sub("", scheme)
    if "{" in literals_only or "}" in literals_only:
        raise FilenameSchemeError(
            f"Unbalanced braces in filename scheme {scheme!r}: "
            "every field must be written as '{field_name}'."
        )

    pattern = ""
    fields: list[str] = []
    position = 0
    for match in _FIELD_PATTERN.finditer(scheme):
        field = match.group(1).strip()
        if not field.isidentifier():
            raise FilenameSchemeError(
                f"Invalid field {match.group(0)!r} in filename scheme {scheme!r}: "
                "a field name must be a plain identifier, e.g. '{sample_name}'."
            )
        if field in fields:
            raise FilenameSchemeError(
                f"Field {field!r} appears more than once in filename scheme {scheme!r}; "
                "each field must be unique."
            )
        fields.append(field)
        pattern += re.escape(scheme[position : match.start()]) + f"(?P<{field}>.+)"
        position = match.end()
    pattern += re.escape(scheme[position:])

    if SAMPLE_NAME_FIELD not in fields:
        raise FilenameSchemeError(
            f"Filename scheme {scheme!r} has no {{{SAMPLE_NAME_FIELD}}} field; "
            f"known fields in this scheme: {fields or 'none'}."
        )
    return re.compile(rf"\A{pattern}\Z")


def parse_stem(stem: str, pattern: re.Pattern[str]) -> dict[str, str] | None:
    """Every field `pattern` captures from `stem`, or None if it doesn't match.
    A non-match isn't an error here -- what to do with an unparseable filename
    is the caller's policy (see `api.add_samples_from_dir`, which skips it)."""
    match = pattern.fullmatch(stem)
    return match.groupdict() if match is not None else None
