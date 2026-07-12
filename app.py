"""MicroCurveBL — Streamlit GUI (thin layer over microcurvebl/*)."""
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from microcurvebl import solve, skin_friction, couple_stress, profiles

st.set_page_config(page_title="MicroCurveBL", layout="wide")
st.title("MicroCurveBL")
st.caption("Dual-solution boundary layer — unsteady micropolar fluid over a "
           "permeable curved stretching/shrinking surface")

with st.sidebar:
    st.header("Parameters")
    n = st.select_slider("n (concentration)", [0.0, 0.5, 1.0], 0.0)
    k = st.slider("k (curvature)", 5.0, 1000.0, 50.0)
    K = st.slider("K (material)", 0.0, 2.0, 1.0, 0.1)
    beta = st.slider("β (unsteadiness)", -5.0, 0.0, -2.0, 0.1)
    S = st.slider("S (mass flux)", -3.0, 3.0, 2.0, 0.1)
    lam = st.slider("λ (stretch/shrink)", -1.0, 1.0, -0.1, 0.01)
    branch = st.radio("Branch", ["first", "second"], horizontal=True)
    eta_max = st.slider("η∞ (domain)", 10.0, 40.0, 25.0)

seed = -1.0 if branch == "first" else -5.0
sol = solve(n, k, K, beta, S, lam, eta_max=eta_max, guess_fpp0=seed)

if not sol.converged:
    st.error(f"No convergence at these parameters (status {sol.status}). "
             "This may be beyond the critical point of the selected branch.")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("-C_f Re_x^{1/2}", f"{skin_friction(sol):.5f}")
    c2.metric("C_m Re_x", f"{couple_stress(sol):.5f}")
    c3.metric("f''(0)", f"{sol.fpp0:.5f}")

    eta, fp, g = profiles(sol)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(eta, fp); ax[0].set(xlabel="η", ylabel="f'(η)", title="Velocity")
    ax[1].plot(eta, g, color="C1"); ax[1].set(xlabel="η", ylabel="g(η)",
                                               title="Microrotation")
    for a in ax: a.grid(alpha=0.3)
    st.pyplot(fig)

st.info("K=0 (Newtonian) results are validated against Tables 1–2 of "
        "Muhad Saleh et al. (2017). See tests/ for the regression suite.")
