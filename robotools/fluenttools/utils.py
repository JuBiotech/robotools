"""Generic utility functions."""
import re

from robotools.liquidhandling import Labware

_WELLID_MATCHER = re.compile(r"^([a-zA-Z]+?)(\d+?)$")
"""Compiled RegEx for matching well row & column from alphanumeric IDs."""


def get_well_position(labware: Labware, well: str) -> int:
    """Calculate the EVO-style well position from the alphanumeric ID."""
    # Extract row & column number from the alphanumeric ID
    m = _WELLID_MATCHER.match(well)
    if m is None:
        raise ValueError(f"This is not an alphanumeric well ID: '{well}'.")
    row = m.group(1)
    column = int(m.group(2))

    c = labware.column_ids.index(column)

    # The Fluent does NOT count rows inside troughs!
    if labware.is_trough:
        return 1 + c

    # Therefore the row number is only relevant for non-trough labware.
    row = well[0]
    r = labware.row_ids.index(row)
    return 1 + c * labware.n_rows + r
