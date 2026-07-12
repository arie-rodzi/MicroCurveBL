"""Regression tests against the benchmark tables of Muhad Saleh et al. (2017).

Table 1 (Newtonian curved stretching) and Table 2 (Newtonian flat shrinking
with suction) are asserted to <2e-3. These are the cases the original authors
used to validate their method against Abbas et al. and Rosca & Pop.
"""
import numpy as np
from microcurvebl import solve, skin_friction

TABLE1 = {5: 1.15167, 10: 1.07348, 20: 1.03505, 30: 1.02317, 40: 1.01731,
          50: 1.01381, 100: 1.00687, 200: 1.00342, 1000: 1.00068}
TABLE2 = {-0.5: 0.85355, -0.6: 0.97947, -0.7: 1.08340, -0.75: 1.12500,
          -0.8: 1.15777, -0.9: 1.18460}  # 'Present (Numerical)' column


def test_table1_curved_stretching():
    for k, ref in TABLE1.items():
        em = 40 if k <= 10 else 25
        s = solve(n=0.5, k=k, K=0, beta=0, S=0, lam=1.0, eta_max=em, nmesh=2500)
        assert s.converged
        assert abs(skin_friction(s) - ref) < 2e-3, (k, skin_friction(s), ref)


def test_table2_flat_shrinking_suction():
    for lam, ref in TABLE2.items():
        s = solve(n=0.5, k=1e6, K=0, beta=0, S=2.0, lam=lam,
                  eta_max=20, nmesh=1500, guess_fpp0=1.0)
        assert s.converged
        assert abs(abs(s.fpp0) - ref) < 2e-3, (lam, abs(s.fpp0), ref)
