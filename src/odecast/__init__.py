"""
Odecast: A DSL for ordinary differential equations
"""

from .symbols import t
from .api import var, solve, BC
from .equation import Eq

__version__ = "0.1.0"
__all__ = ["t", "var", "Eq", "solve", "BC"]
