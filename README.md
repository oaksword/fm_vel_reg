# Circle Jacobian diagnostics

Population-level numerical checks accompanying the paper *Geometric Velocity Regularity for Flow Matching on Manifold-Concentrated Data*. The script evaluates the flow-matching velocity Jacobian on the unit circle through the exact posterior-covariance identity. No velocity field is trained.

The source is uniform on $S^1$ by default, with one nonuniform density $f(\omega)\propto 1+0.5\cos\omega$. For $\sigma\in\{0.2,0.1,0.05,0.025,0.0125\}$ at the switching time $t^\star=1/(1+\sigma)$, the diagnostics are:

- centerline normal and tangential forms, and the operator norm, for the uniform circle
- tangential, normal, and mixed blocks for the nonuniform density
- time integrals of the centerline operator norm against the $\sigma_t^{-1}$ and $\sigma_t^{-3}$ envelopes
- operator-norm quantiles on held-out samples from the late marginal
- quadrature checked against the von Mises closed form

Running `python3 run_circle_jacobian.py` rewrites `results/`, including the CSV tables, `results/summary.md`, and the PDF figures. Dependencies are listed in `requirements.txt`.
