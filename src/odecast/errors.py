"""
Custom error types for odecast
"""


class OdecastError(Exception):
    """Base exception for odecast errors."""


class MissingInitialConditionError(OdecastError):
    """Raised when required initial conditions are missing."""


class OrderMismatchError(OdecastError):
    """Raised when variable order doesn't match equation requirements."""


class CompilationError(OdecastError):
    """Raised when symbolic to numeric compilation fails."""


class BackendError(OdecastError):
    """Raised when a backend-specific error occurs."""


class OverdeterminedConditionsError(OdecastError):
    """Raised when too many initial/boundary conditions are provided."""


class NonSolvableFormError(OdecastError):
    """Raised when the equation cannot be solved in the required form."""
