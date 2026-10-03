"""Generic utility functions."""

import re

from robotools.liquidhandling import Labware


def to_hex(dec: int):
    """Method from stackoverflow to convert decimal to hex.

    Link: https://stackoverflow.com/questions/5796238/python-convert-decimal-to-hex
    Solution posted by user "Chunghee Kim" on 21.11.2020.
    """
    digits = "0123456789ABCDEF"
    x = dec % 16
    rest = dec // 16
    if rest == 0:
        return digits[x]
    return to_hex(rest) + digits[x]


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

    r = labware.row_ids.index(row)
    c = labware.column_ids.index(column)

    # Calculate the position from the row & column number.
    # The EVO counts virtual rows in troughs too.
    if labware.virtual_rows is not None:
        return 1 + c * labware.virtual_rows + r
    return 1 + c * labware.n_rows + r
