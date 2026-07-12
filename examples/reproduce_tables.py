"""Reproduce Table 1 of the source paper and print a comparison."""
from microcurvebl import solve, skin_friction

REF = {5: 1.15167, 10: 1.07348, 20: 1.03505, 50: 1.01381,
       100: 1.00687, 1000: 1.00068}
print(f"{'k':>6}{'-Cf Re^0.5':>14}{'paper':>10}{'err':>10}")
for k, ref in REF.items():
    em = 40 if k <= 10 else 25
    s = solve(n=0.5, k=k, K=0, beta=0, S=0, lam=1.0, eta_max=em, nmesh=2500)
    v = skin_friction(s)
    print(f"{k:>6}{v:>14.5f}{ref:>10.5f}{abs(v-ref):>10.5f}")
