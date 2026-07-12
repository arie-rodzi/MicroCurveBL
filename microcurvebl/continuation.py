"""Dual-solution continuation: trace a solution branch through a swept
parameter and locate the critical (turning) point where the first and
second branches merge and no solution exists beyond it.
"""
import numpy as np
from .solver import solve


def sweep(param, values, branch="first", fixed=None, eta_max=25.0, nmesh=2000):
    """Trace one branch as `param` varies over `values`.

    param  : one of 'lam','S','beta','k','K','n'
    branch : 'first' (stable, small |f''(0)| seed) or
             'second' (unstable, large |f''(0)| seed)
    fixed  : dict of the remaining parameters
    Returns list of Solution objects (converged ones), warm-started along
    the branch. The last value before divergence brackets the critical point.
    """
    fixed = dict(fixed or {})
    seed = -1.0 if branch == "first" else -5.0
    out, warm = [], None
    for v in values:
        kw = dict(fixed); kw[param] = v
        kw.setdefault("n", 0.0); kw.setdefault("k", 50.0); kw.setdefault("K", 1.0)
        kw.setdefault("beta", -2.0); kw.setdefault("S", 2.0); kw.setdefault("lam", -0.1)
        if warm is None:
            sol = solve(**kw, eta_max=eta_max, nmesh=nmesh, guess_fpp0=seed)
        else:
            sol = solve(**kw, eta_max=eta_max, nmesh=nmesh,
                        eta_init=warm.eta, y_init=warm.y)
        if not sol.converged:
            break
        out.append(sol); warm = sol
    return out


def critical_point(param, lo, hi, branch="first", fixed=None, tol=1e-4,
                   eta_max=25.0, nmesh=1500):
    """Bisection for the critical value of `param` between a converging value
    `lo` and a non-converging value `hi`."""
    fixed = dict(fixed or {})
    def ok(v):
        return bool(sweep(param, [v], branch, fixed, eta_max, nmesh))
    assert ok(lo) and not ok(hi), "need lo converging and hi diverging"
    while abs(hi - lo) > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return 0.5 * (lo + hi)
