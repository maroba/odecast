"""
Odecast: A DSL for ordinary differential equations
"""

from .api import BC, solve, var
from .equation import Eq
from .events import Event
from .fitting import fit
from .printing import latex
from .stability import equilibria
from .symbols import t

__version__ = "0.2.0"
__all__ = ["BC", "Eq", "Event", "equilibria", "fit", "latex", "solve", "t", "var"]
