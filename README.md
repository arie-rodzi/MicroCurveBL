# MicroCurveBL

Dual-solution boundary-layer analysis of **unsteady micropolar fluid over a
permeable curved stretching/shrinking surface** (model of Muhad Saleh et al.,
*Math. Probl. Eng.* 2017).

## Install
```bash
pip install -r requirements.txt
```

## Use (headless core)
```python
from microcurvebl import solve, skin_friction, couple_stress
sol = solve(n=0.0, k=50, K=1.0, beta=-2.0, S=2.0, lam=-0.1, guess_fpp0=-1.0)
print(sol.converged, skin_friction(sol), couple_stress(sol))
```

## Interactive GUI
```bash
streamlit run app.py
```

## Reproduce benchmarks / run tests
```bash
PYTHONPATH=. python examples/reproduce_tables.py
PYTHONPATH=. pytest tests/          # Tables 1 & 2 asserted < 2e-3
```

## Layout
```
microcurvebl/   equations.py  solver.py  continuation.py  postproc.py
tests/          test_validation.py        (Newtonian benchmarks)
examples/       reproduce_tables.py
app.py          Streamlit interface
paper/          SoftwareX manuscript (.tex + .pdf)
```

## Validation status
- **K = 0 (Newtonian):** Tables 1 & 2 reproduced to <2e-3 (exact to 5 d.p. for k>=20). Enforced by the test suite.
- **K > 0 with beta != 0 (full micropolar + unsteady):** the `COUPLING`-flagged
  terms in `equations.py` must be sign-confirmed against the original LaTeX /
  Maple `shootlib` source before the Table 3 rows lock to full precision. The
  source PDF prints inconsistent signs (Eq. 8 vs Eq. 11) on the beta/2
  microrotation term.
