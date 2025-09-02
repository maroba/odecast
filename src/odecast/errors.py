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


class OverdeterminedConditionsError(OdecastError):
    """Raised when too many initial/boundary conditions are provided."""

    pass


class NonSolvableFormError(OdecastError):
    """Raised when the equation cannot be solved in the required form."""

    pass
