"""Engineering quantities of interest."""
import numpy as np


def skin_friction(sol):
    """-C_f Re_x^{1/2}. Sign convention matches Tables 1-3 of the source paper
    (the solver's eta orientation flips the printed +lam/k to -lam/k)."""
    p = sol.params
    return (1 + p["K"]) * (-sol.fpp0 + p["lam"] / p["k"])


def couple_stress(sol):
    """C_m Re_x."""
    p = sol.params
    return (1 + p["K"] / 2) * (sol.gp0 - p["n"] * sol.fpp0 / p["k"])


def profiles(sol):
    """Return (eta, f'(eta), g(eta)) for plotting boundary-layer profiles."""
    return sol.eta, sol.y[1], sol.y[4]
