"""
Odecast: A DSL for ordinary differential equations
"""

from .api import BC, solve, var
from .equation import Eq
from .symbols import t

__version__ = "0.1.1"
__all__ = ["BC", "Eq", "solve", "t", "var"]
