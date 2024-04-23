"""Exceptions that indicate problems in liquid handling."""

from typing import Optional

__all__ = (
    "VolumeOverflowError",
    "VolumeUnderflowError",
    "VolumeViolationException",
)


class VolumeViolationException(Exception):
    """Error indicating a violation of volume constraints."""


class VolumeOverflowError(VolumeViolationException):
    """Error that indicates the planned overflow of a well."""

    def __init__(
        self,
        labware: str,
        well: str,
        current: float,
        change: float,
        threshold: float,
        label: Optional[str] = None,
    ) -> None:
        if label:
            super().__init__(
                f'Too much volume for "{labware}".{well}: {current} + {change} > {threshold} in step {label}'
            )
        else:
            super().__init__(f'Too much volume for "{labware}".{well}: {current} + {change} > {threshold}')


class VolumeUnderflowError(VolumeViolationException):
    """Error that indicates the planned underflow of a well."""

    def __init__(
        self,
        labware: str,
        well: str,
        current: float,
        change: float,
        threshold: float,
        label: Optional[str] = None,
    ) -> None:
        if label:
            super().__init__(
                f'Too little volume in "{labware}".{well}: {current} - {change} < {threshold} in step {label}'
            )
        else:
            super().__init__(f'Too little volume in "{labware}".{well}: {current} - {change} < {threshold}')
