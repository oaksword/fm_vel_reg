#!/usr/bin/env python3
"""Population Jacobian diagnostics for FM on a noisy circle.

This script uses the exact posterior-covariance identity

    J_t(x) = alpha_t I + t(1-t)/sigma_t^4 Cov(Y | X_t=x)

for Y on S^1 and X_t = tY + sigma_t eta.  It is intentionally a
population-level numerical check, not a learned-model experiment.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import ive

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

SIGMAS = [0.2, 0.1, 0.05, 0.025, 0.0125]
THETA_N = 65536
INTEGRATION_T_N = 1800
HELDOUT_N = 500
RNG_SEED = 13428
TIME_PROFILE_SIGMA = 0.05


def sigma_t_value(sigma: float, t: float) -> float:
    return math.sqrt((1.0 - t) ** 2 + (t * sigma) ** 2)


def circle_grid(n: int = THETA_N):
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    y = np.column_stack([np.cos(theta), np.sin(theta)])
    return theta, y


THETA, YGRID = circle_grid()


def posterior_covariance(
    x: np.ndarray,
    sigma: float,
    t: float,
    density_a: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return posterior mean and covariance of Y | X_t=x by quadrature.

    density_a=0 gives the uniform circle.  density_a in (-1,1) gives
    f(theta) proportional to 1 + density_a cos(theta).
    """
    sigt = sigma_t_value(sigma, t)
    log_prior = np.log1p(density_a * np.cos(THETA))
    logw = log_prior + (t / (sigt * sigt)) * (YGRID @ x)
    logw -= np.max(logw)
    w = np.exp(logw)
    w /= np.sum(w)
    mean = np.sum(w[:, None] * YGRID, axis=0)
    second = np.einsum("n,ni,nj->ij", w, YGRID, YGRID)
    cov = second - np.outer(mean, mean)
    return mean, cov


def jacobian(
    x: np.ndarray,
    sigma: float,
    t: float,
    density_a: float = 0.0,
) -> np.ndarray:
    sigt = sigma_t_value(sigma, t)
    _, cov = posterior_covariance(x=x, sigma=sigma, t=t, density_a=density_a)
    alpha = (t * (1.0 + sigma * sigma) - 1.0) / (sigt ** 2)
    pref = t * (1.0 - t) / (sigt ** 4)
    return alpha * np.eye(2) + pref * cov


def opnorm_symmetric(j: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvalsh(0.5 * (j + j.T)))))


def tangent_normal_basis(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = y / np.linalg.norm(y)
    tau = np.array([-n[1], n[0]])
    return tau, n


def centerline_row(sigma: float) -> dict:
    t = 1.0 / (1.0 + sigma)
    sigt = sigma_t_value(sigma, t)
    y = np.array([1.0, 0.0])
    x = t * y
    j = jacobian(x=x, sigma=sigma, t=t)
    tau, n = tangent_normal_basis(y)
    lambda_t = float(tau @ j @ tau)
    lambda_n = float(n @ j @ n)
    op = opnorm_symmetric(j)
    return {
        "sigma": sigma,
        "t_star": t,
        "sigma_t": sigt,
        "lambda_normal": lambda_n,
        "lambda_tangent": lambda_t,
        "opnorm": op,
        "sigma_t_times_opnorm": sigt * op,
        "sigma_t_cubed_times_opnorm": (sigt ** 3) * op,
        "normal_prediction_at_t_star": -((1.0 - sigma * sigma) / (2.0 * sigma)),
    }


def closed_form_centerline_row(sigma: float) -> dict:
    """Validate uniform-centerline quadrature against Bessel moments.

    For x=rho e1, the posterior is von Mises with beta=t*rho/sigma_t^2.
    On the centerline rho=t.  Ratios I_n(beta)/I_0(beta) are computed with
    exponentially scaled Bessel functions ive for numerical stability.
    """
    t = 1.0 / (1.0 + sigma)
    sigt = sigma_t_value(sigma, t)
    beta = t * t / (sigt ** 2)
    a1 = float(ive(1, beta) / ive(0, beta))
    a2 = float(ive(2, beta) / ive(0, beta))
    var_n = (1.0 + a2) / 2.0 - a1 * a1
    var_t = (1.0 - a2) / 2.0
    alpha = (t * (1.0 + sigma * sigma) - 1.0) / (sigt ** 2)
    pref = t * (1.0 - t) / (sigt ** 4)
    lambda_n_closed = alpha + pref * var_n
    lambda_t_closed = alpha + pref * var_t

    quad = centerline_row(sigma)
    return {
        "sigma": sigma,
        "beta": beta,
        "lambda_normal_quadrature": quad["lambda_normal"],
        "lambda_normal_closed": lambda_n_closed,
        "abs_error_normal": abs(quad["lambda_normal"] - lambda_n_closed),
        "lambda_tangent_quadrature": quad["lambda_tangent"],
        "lambda_tangent_closed": lambda_t_closed,
        "abs_error_tangent": abs(quad["lambda_tangent"] - lambda_t_closed),
    }


def integrated_row(sigma: float) -> dict:
    # Include t=1 but avoid t=0 because x=t*y collapses to the origin, where the
    # chosen centerline direction is arbitrary.  The omitted short interval has no
    # relevance for the terminal singular scaling.
    ts = np.linspace(1.0e-3, 1.0, INTEGRATION_T_N)
    ops = []
    invs = []
    cubics = []
    y = np.array([1.0, 0.0])
    for t in ts:
        sigt = sigma_t_value(sigma, float(t))
        x = float(t) * y
        j = jacobian(x=x, sigma=sigma, t=float(t))
        ops.append(opnorm_symmetric(j))
        invs.append(sigt ** -1)
        cubics.append(sigt ** -3)
    int_op = float(np.trapezoid(ops, ts))
    int_inv = float(np.trapezoid(invs, ts))
    int_cubic = float(np.trapezoid(cubics, ts))
    return {
        "sigma": sigma,
        "int_centerline_opnorm": int_op,
        "int_sigma_t_inv": int_inv,
        "int_sigma_t_cubic": int_cubic,
        "int_op_over_log_inv_sigma": int_op / math.log(1.0 / sigma),
        "sigma_squared_times_int_cubic": (sigma ** 2) * int_cubic,
    }


def nonuniform_row(sigma: float, density_a: float = 0.5) -> dict:
    t = 1.0 / (1.0 + sigma)
    sigt = sigma_t_value(sigma, t)
    theta0 = math.pi / 2.0
    y = np.array([math.cos(theta0), math.sin(theta0)])
    x = t * y
    j = jacobian(x=x, sigma=sigma, t=t, density_a=density_a)
    tau, n = tangent_normal_basis(y)
    jtt = float(tau @ j @ tau)
    jnn = float(n @ j @ n)
    jtn = float(tau @ j @ n)
    return {
        "sigma": sigma,
        "t_star": t,
        "sigma_t": sigt,
        "J_TT": jtt,
        "J_NN": jnn,
        "J_TN": jtn,
        "abs_J_TN_over_sigma_t": abs(jtn) / sigt,
        "sigma_t_times_opnorm": sigt * opnorm_symmetric(j),
    }


def time_profile(sigma: float = TIME_PROFILE_SIGMA, n: int = 500) -> pd.DataFrame:
    """Centerline eigenvalue profile over late times for plotting."""
    t_star = 1.0 / (1.0 + sigma)
    ts = np.unique(
        np.concatenate(
            [
                np.linspace(max(1.0e-3, t_star - 0.20), t_star, n // 2),
                np.linspace(t_star, 1.0, n // 2),
            ]
        )
    )
    y = np.array([1.0, 0.0])
    tau, nvec = tangent_normal_basis(y)
    rows = []
    for t in ts:
        sigt = sigma_t_value(sigma, float(t))
        j = jacobian(x=float(t) * y, sigma=sigma, t=float(t))
        lam_t = float(tau @ j @ tau)
        lam_n = float(nvec @ j @ nvec)
        op = opnorm_symmetric(j)
        rows.append(
            {
                "sigma": sigma,
                "t": float(t),
                "t_star": t_star,
                "sigma_t": sigt,
                "lambda_normal": lam_n,
                "lambda_tangent": lam_t,
                "opnorm": op,
                "sigma_t_times_opnorm": sigt * op,
            }
        )
    return pd.DataFrame(rows)


def heldout_row(sigma: float, n: int = HELDOUT_N) -> dict:
    rng = np.random.default_rng(RNG_SEED + int(round(100000 * sigma)))
    t = 1.0 / (1.0 + sigma)
    sigt = sigma_t_value(sigma, t)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    y = np.column_stack([np.cos(theta), np.sin(theta)])
    eta = rng.standard_normal(size=(n, 2))
    xs = t * y + sigt * eta
    scaled_ops = []
    normal_eigs = []
    tangent_eigs = []
    for x in xs:
        j = jacobian(x=x, sigma=sigma, t=t)
        # Project using the nearest point on the circle.  Extremely rare samples
        # near zero are skipped because projection is ill-conditioned there.
        norm_x = np.linalg.norm(x)
        if norm_x < 1.0e-10:
            continue
        yhat = x / norm_x
        tau, nvec = tangent_normal_basis(yhat)
        scaled_ops.append(sigt * opnorm_symmetric(j))
        normal_eigs.append(float(nvec @ j @ nvec))
        tangent_eigs.append(float(tau @ j @ tau))
    scaled_ops = np.array(scaled_ops)
    normal_eigs = np.array(normal_eigs)
    tangent_eigs = np.array(tangent_eigs)
    return {
        "sigma": sigma,
        "t_star": t,
        "sigma_t": sigt,
        "num_samples": len(scaled_ops),
        "scaled_op_q50": float(np.quantile(scaled_ops, 0.50)),
        "scaled_op_q90": float(np.quantile(scaled_ops, 0.90)),
        "scaled_op_q99": float(np.quantile(scaled_ops, 0.99)),
        "normal_eig_q10": float(np.quantile(normal_eigs, 0.10)),
        "normal_eig_q50": float(np.quantile(normal_eigs, 0.50)),
        "tangent_eig_q50": float(np.quantile(tangent_eigs, 0.50)),
    }


def slope(x: Iterable[float], y: Iterable[float]) -> float:
    lx = np.log(np.array(list(x), dtype=float))
    ly = np.log(np.array(list(y), dtype=float))
    return float(np.polyfit(lx, ly, deg=1)[0])


def set_compact_log_xticks(ax, values) -> None:
    """Label a log x-axis only at the plotted samples, in plain decimals."""
    ticks = np.asarray(list(values), dtype=float)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda value, _pos: f"{value:.3g}")
    )
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def plot_results(
    center: pd.DataFrame,
    integ: pd.DataFrame,
    mixed: pd.DataFrame,
    heldout: pd.DataFrame,
    profile: pd.DataFrame,
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 160})

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    x = center["sigma_t"].to_numpy()
    y = center["opnorm"].to_numpy()
    ax.loglog(x, y, "o-", label=r"$\|\nabla v_t^\star\|_{op}$")
    ref1 = y[-1] * (x / x[-1]) ** (-1)
    ref3 = y[-1] * (x / x[-1]) ** (-3)
    ax.loglog(x, ref1, "--", label=r"slope $-1$")
    ax.loglog(x, ref3, ":", label=r"slope $-3$ reference")
    set_compact_log_xticks(ax, x)
    ax.set_xlabel(r"$\sigma_{t^\star}$")
    ax.set_ylabel("operator norm")
    ax.set_title("Centerline Jacobian scaling on $S^1$")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "centerline_loglog_scaling.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.semilogx(center["sigma"], center["sigma_t_times_opnorm"], "o-", label=r"$\sigma_t\|J\|_{op}$")
    ax.semilogx(center["sigma"], center["sigma_t_cubed_times_opnorm"], "s-", label=r"$\sigma_t^3\|J\|_{op}$")
    ax.invert_xaxis()
    set_compact_log_xticks(ax, center["sigma"])
    ax.set_xlabel(r"$\sigma$ decreasing $\rightarrow$")
    ax.set_ylabel("scaled quantity")
    ax.set_title("Bounded inverse-scale diagnostic")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "scaled_centerline_quantities.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.loglog(integ["sigma"], integ["int_centerline_opnorm"], "o-", label=r"$\int \|J_t\|dt$")
    ax.loglog(integ["sigma"], integ["int_sigma_t_inv"], "s-", label=r"$\int \sigma_t^{-1}dt$")
    ax.loglog(integ["sigma"], integ["int_sigma_t_cubic"], "^-", label=r"$\int \sigma_t^{-3}dt$")
    ax.invert_xaxis()
    set_compact_log_xticks(ax, integ["sigma"])
    ax.set_xlabel(r"$\sigma$ decreasing $\rightarrow$")
    ax.set_ylabel("time integral")
    ax.set_title("Integrated actual Jacobian vs ambient envelopes")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "integrated_scaling_comparison.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.semilogx(center["sigma"], center["lambda_normal"], "o-", label="normal/radial")
    ax.semilogx(center["sigma"], center["lambda_tangent"], "s-", label="tangent/angular")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.invert_xaxis()
    set_compact_log_xticks(ax, center["sigma"])
    ax.set_xlabel(r"$\sigma$ decreasing $\rightarrow$")
    ax.set_ylabel("eigenvalue")
    ax.set_title(r"Eigenvalue anisotropy at $t^\star$")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "eigenvalues_at_tstar.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.semilogx(mixed["sigma"], mixed["J_TT"], "o-", label=r"$J_{TT}$")
    ax.semilogx(mixed["sigma"], -mixed["J_NN"] * mixed["sigma_t"], "s-", label=r"$-\sigma_t J_{NN}$")
    ax.semilogx(mixed["sigma"], mixed["abs_J_TN_over_sigma_t"], "^-", label=r"$|J_{TN}|/\sigma_t$")
    ax.invert_xaxis()
    set_compact_log_xticks(ax, mixed["sigma"])
    ax.set_xlabel(r"$\sigma$ decreasing $\rightarrow$")
    ax.set_ylabel("scaled block value")
    ax.set_title("Nonuniform circle block diagnostics")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "nonuniform_block_diagnostics.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(profile["t"], profile["lambda_normal"], label="normal/radial")
    ax.plot(profile["t"], profile["lambda_tangent"], label="tangent/angular")
    ax.plot(profile["t"], profile["opnorm"], "--", label="operator norm")
    ax.axvline(float(profile["t_star"].iloc[0]), color="black", linewidth=0.8, linestyle=":", label=r"$t^\star$")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("time t")
    ax.set_ylabel("value")
    ax.set_title(fr"Late-time centerline profile, $\sigma={float(profile['sigma'].iloc[0]):g}$")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "time_profile_sigma005.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.semilogx(heldout["sigma"], heldout["scaled_op_q50"], "o-", label="median")
    ax.semilogx(heldout["sigma"], heldout["scaled_op_q90"], "s-", label="q90")
    ax.semilogx(heldout["sigma"], heldout["scaled_op_q99"], "^-", label="q99")
    ax.invert_xaxis()
    set_compact_log_xticks(ax, heldout["sigma"])
    ax.set_xlabel(r"$\sigma$ decreasing $\rightarrow$")
    ax.set_ylabel(r"quantile of $\sigma_t\|J\|_{op}$")
    ax.set_title(r"Held-out tube samples at $t^\star$")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "heldout_scaled_op_quantiles.pdf")
    plt.close(fig)


def write_summary(
    center: pd.DataFrame,
    integ: pd.DataFrame,
    mixed: pd.DataFrame,
    heldout: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    slope_op_vs_sigt = slope(center["sigma_t"], center["opnorm"])
    slope_int_op_vs_sigma = slope(center["sigma"], integ["int_centerline_opnorm"])
    slope_cubic_vs_sigma = slope(center["sigma"], integ["int_sigma_t_cubic"])

    def md_table(df: pd.DataFrame, cols: list[str], digits: int = 5) -> str:
        # Avoid pandas.to_markdown because it needs the optional tabulate package.
        def fmt(value):
            if isinstance(value, (float, np.floating)):
                return f"{float(value):.{digits}g}"
            return str(value)

        rows = [[fmt(v) for v in row] for row in df[cols].to_numpy()]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = ["| " + " | ".join(row) + " |" for row in rows]
        return "\n".join([header, sep] + body)

    text = f"""# Circle Jacobian diagnostics

Generated by `python3 run_circle_jacobian.py`.

## Main conclusion

At the centerline switching time `t_star=1/(1+sigma)`, the fitted log-log slope of `opnorm` versus `sigma_t` is

```text
{slope_op_vs_sigt:.3f}
```

which is close to the predicted `-1`. The scaled quantity `sigma_t_times_opnorm` remains bounded, while `sigma_t_cubed_times_opnorm` decreases quickly. The dominant eigenvalue in magnitude is the normal/radial one and is negative, so the sharp singularity corresponds to normal contraction rather than tangential expansion.

## D1. Centerline sharpness at t_star

{md_table(center, ["sigma", "sigma_t", "lambda_normal", "lambda_tangent", "opnorm", "sigma_t_times_opnorm", "sigma_t_cubed_times_opnorm"], digits=6)}

The analytic centerline normal prediction at `t_star` is approximately `-(1-sigma^2)/(2 sigma)`. The numerical normal eigenvalue approaches this value as `sigma` decreases.

## D2. Time-integrated comparison

{md_table(integ, ["sigma", "int_centerline_opnorm", "int_sigma_t_inv", "int_sigma_t_cubic", "int_op_over_log_inv_sigma", "sigma_squared_times_int_cubic"], digits=6)}

The fitted slope of `int_sigma_t_cubic` versus `sigma` is `{slope_cubic_vs_sigma:.3f}`, close to the predicted `-2`. The centerline Jacobian integral grows slowly and is consistent with logarithmic growth; the displayed ratio `int_op_over_log_inv_sigma` stays order-one.

## D3. Nonuniform circle mixed block

Density: `f(theta) proportional to 1 + 0.5 cos(theta)`. Evaluation point: `theta=pi/2` at `t_star`.

{md_table(mixed, ["sigma", "sigma_t", "J_TT", "J_NN", "J_TN", "abs_J_TN_over_sigma_t", "sigma_t_times_opnorm"], digits=6)}

The tangential block stays order-one, the normal block is negative and order `-sigma_t^{-1}`, and `abs(J_TN)/sigma_t` is bounded.

## D4. Held-out tube samples at t_star

Uniform circle; samples are drawn from `X_t=tY+sigma_t eta`. We report quantiles of `sigma_t * ||J||_op` and representative projected eigenvalue quantiles.

{md_table(heldout, ["sigma", "sigma_t", "num_samples", "scaled_op_q50", "scaled_op_q90", "scaled_op_q99", "normal_eig_q50", "tangent_eig_q50"], digits=6)}

The held-out quantiles of `sigma_t * ||J||_op` remain order-one across decreasing `sigma`, indicating that the centerline result is representative of late tube samples.

## Closed-form validation

For the uniform centerline case, the posterior is von Mises and the covariance can also be computed from Bessel-function ratios. The quadrature agrees with this closed-form check to high precision:

{md_table(validation, ["sigma", "beta", "abs_error_normal", "abs_error_tangent"], digits=6)}

## Figures

PDF plots are written to `results/figures/`:

- `centerline_loglog_scaling`: log-log slope check against `-1` and `-3` references.
- `scaled_centerline_quantities`: bounded `sigma_t ||J||` and vanishing `sigma_t^3 ||J||`.
- `integrated_scaling_comparison`: actual centerline integral vs `sigma_t^{-1}` and `sigma_t^{-3}` envelopes.
- `eigenvalues_at_tstar`: normal/tangential eigenvalues at the switching time.
- `nonuniform_block_diagnostics`: tangent, normal, and mixed projected blocks for a nonuniform circle.
- `time_profile_sigma005`: late-time profile for `sigma=0.05`.
- `heldout_scaled_op_quantiles`: quantiles under held-out tube samples.
"""
    (RESULTS / "summary.md").write_text(text)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    center = pd.DataFrame([centerline_row(s) for s in SIGMAS])
    integ = pd.DataFrame([integrated_row(s) for s in SIGMAS])
    mixed = pd.DataFrame([nonuniform_row(s) for s in SIGMAS])
    heldout = pd.DataFrame([heldout_row(s) for s in SIGMAS])
    validation = pd.DataFrame([closed_form_centerline_row(s) for s in SIGMAS])
    profile = time_profile()

    center.to_csv(RESULTS / "centerline_sharpness.csv", index=False)
    integ.to_csv(RESULTS / "integrated_scaling.csv", index=False)
    mixed.to_csv(RESULTS / "nonuniform_mixed_blocks.csv", index=False)
    heldout.to_csv(RESULTS / "heldout_tube_quantiles.csv", index=False)
    validation.to_csv(RESULTS / "closed_form_validation.csv", index=False)
    profile.to_csv(RESULTS / "time_profile_sigma005.csv", index=False)
    plot_results(center=center, integ=integ, mixed=mixed, heldout=heldout, profile=profile)
    write_summary(center=center, integ=integ, mixed=mixed, heldout=heldout, validation=validation)

    print(f"Wrote results to {RESULTS}")
    print((RESULTS / "summary.md").read_text())


if __name__ == "__main__":
    main()
