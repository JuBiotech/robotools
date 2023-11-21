"""Enums and classes, and related helper functions."""
import enum

__all__ = (
    "Labwares",
    "Tip",
    "int_to_tip",
)


class Labwares(str, enum.Enum):
    """Built-in EVOware labware identifiers."""

    SystemLiquid = "Systemliquid"


class Tip(enum.IntEnum):
    """Enumeration of LiHa tip IDs."""

    Any = -1
    T1 = 1
    T2 = 2
    T3 = 4
    T4 = 8
    T5 = 16
    T6 = 32
    T7 = 64
    T8 = 128


def int_to_tip(tip_int: int) -> Tip:
    """Checks and convert a tip number [1-8] to the Tecan Tip ID."""
    if tip_int == 1:
        return Tip.T1
    elif tip_int == 2:
        return Tip.T2
    elif tip_int == 3:
        return Tip.T3
    elif tip_int == 4:
        return Tip.T4
    elif tip_int == 5:
        return Tip.T5
    elif tip_int == 6:
        return Tip.T6
    elif tip_int == 7:
        return Tip.T7
    elif tip_int == 8:
        return Tip.T8
    raise ValueError(
        f"Tip is {tip_int} with type {type(tip_int)}, but should be an int between 1 and 8 for _int_to_tip conversion."
    )
