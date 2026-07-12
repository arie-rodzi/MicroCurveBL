"""MicroCurveBL: dual-solution boundary-layer analysis of unsteady micropolar
fluid over a permeable curved stretching/shrinking surface."""
from .solver import solve, Solution
from .postproc import skin_friction, couple_stress, profiles
from .continuation import sweep, critical_point

__version__ = "1.0.0"
__all__ = ["solve", "Solution", "skin_friction", "couple_stress",
           "profiles", "sweep", "critical_point"]
