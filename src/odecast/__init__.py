"""
Odecast: A DSL for ordinary differential equations
"""

from .api import BC, solve, var
from .continuation import continuation
from .equation import Eq
from .events import Event
from .fitting import fit
from .linearize import linearize
from .periodic import periodic_orbit
from .portrait import phase_portrait
from .printing import latex
from .stability import equilibria
from .symbols import t

__version__ = "0.3.0"
__all__ = [
    "BC",
    "Eq",
    "Event",
    "continuation",
    "equilibria",
    "fit",
    "latex",
    "linearize",
    "periodic_orbit",
    "phase_portrait",
    "solve",
    "t",
    "var",
]
