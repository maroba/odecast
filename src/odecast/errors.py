"""
Custom error types for odecast
"""


class OdecastError(Exception):
    """Base exception for odecast errors."""

    pass


class MissingInitialConditionError(OdecastError):
    """Raised when required initial conditions are missing."""

    pass


class OrderMismatchError(OdecastError):
    """Raised when variable order doesn't match equation requirements."""

    pass


class CompilationError(OdecastError):
    """Raised when symbolic to numeric compilation fails."""

    pass


class BackendError(OdecastError):
    """Raised when a backend-specific error occurs."""

    pass
