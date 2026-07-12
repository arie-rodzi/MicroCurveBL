"""Similarity ODE system: unsteady micropolar flow over a permeable curved
stretching/shrinking surface (Muhad Saleh et al., Math. Probl. Eng. 2017).

State y = [f, f', f'', f''', g, g'].
  f : dimensionless stream function     g : dimensionless microrotation
Params: n, k (curvature), K (material), beta (unsteadiness), S (mass flux), lam.

CO-AUTHOR VERIFICATION NOTE
---------------------------
Terms flagged `COUPLING` are the only place the source PDF carries OCR sign
ambiguity (Eq. 8 vs Eq. 11 print opposite signs on the beta/2 microrotation
term). Confirm against the original LaTeX / Maple `shootlib` source. The K=0
(Newtonian) reduction is fully validated against Tables 1-2 regardless.
"""
import numpy as np

BETAG_SIGN = -1.0  # COUPLING: sign of beta/2*(eta g' + 3g) in microrotation eq.


def rhs(eta, y, n, k, K, beta):
    f, fp, fpp, fppp, g, gp = y
    r = eta + k
    gpp = (
        - k / r * f * gp + k / r * fp * g
        + K * (2 * g + fpp + fp / r)                 # COUPLING (micropolar)
        + BETAG_SIGN * beta / 2 * (eta * gp + 3 * g) # COUPLING (unsteady)
    ) / (1 + K / 2) - gp / r
    num = (
        k / r * (fp * fpp - f * fppp)
        + k / r**2 * (fp**2 - f * fpp)
        + k / r**3 * f * fp
        + K * (gpp + gp / r)                          # COUPLING (micropolar)
        + beta / r * (fp + eta / 2 * fpp)             # COUPLING (unsteady)
        + beta / 2 * (3 * fpp + eta * fppp)           # COUPLING (unsteady)
        - (1 + K) * (2 * fppp / r - fpp / r**2 + fp / r**3)
    )
    return np.vstack([fp, fpp, fppp, num / (1 + K), gp, gpp])


def bc(ya, yb, n, S, lam):
    return np.array([
        ya[0] - S, ya[1] - lam, ya[4] + n * ya[2],
        yb[1], yb[2], yb[4],
    ])
