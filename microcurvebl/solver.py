"""BVP solver (shooting-equivalent collocation via scipy.solve_bvp)."""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_bvp
from .equations import rhs, bc


@dataclass
class Solution:
    eta: np.ndarray
    y: np.ndarray            # shape (6, N): f, f', f'', f''', g, g'
    params: dict
    converged: bool
    status: int

    @property
    def fpp0(self):  return self.y[2, 0]   # f''(0)
    @property
    def gp0(self):   return self.y[5, 0]   # g'(0)


def solve(n, k, K, beta, S, lam, eta_max=25.0, nmesh=2000,
          guess_fpp0=-1.0, y_init=None, eta_init=None, tol=1e-8):
    """Solve the similarity system for one parameter set.

    guess_fpp0 seeds the branch: a mildly negative value converges to the
    first (stable) solution; a large-magnitude seed targets the second branch.
    Pass y_init/eta_init to warm-start from a neighbouring solution (used by
    the continuation module to trace a branch through parameter space).
    """
    if eta_init is None:
        eta = np.linspace(0.0, eta_max, nmesh)
        fp = lam * np.exp(-eta)
        f = S + lam * (1 - np.exp(-eta))
        fpp = guess_fpp0 * np.exp(-eta)
        y0 = np.vstack([f, fp, fpp, -guess_fpp0 * np.exp(-eta),
                        -n * guess_fpp0 * np.exp(-eta),
                        n * guess_fpp0 * np.exp(-eta)])
    else:
        eta, y0 = eta_init, y_init

    sol = solve_bvp(lambda e, y: rhs(e, y, n, k, K, beta),
                    lambda a, b: bc(a, b, n, S, lam),
                    eta, y0, tol=tol, max_nodes=200000)
    return Solution(sol.x, sol.y, dict(n=n, k=k, K=K, beta=beta, S=S, lam=lam),
                    sol.status == 0, sol.status)
