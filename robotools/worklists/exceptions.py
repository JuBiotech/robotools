"""Exceptions related to liquid handling with worklists."""

__all__ = (
    "CompatibilityError",
    "InvalidOperationError",
)


class CompatibilityError(NotImplementedError):
    """Exception that's thrown when device-specific implementations are required."""


class InvalidOperationError(Exception):
    """When an operation cannot be performed under the present circumstances."""
