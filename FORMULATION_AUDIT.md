# RKHS Formulation Audit

This repository uses ridgeless RKHS collocation: each independent kernel
function is identified by including its squared RKHS norm, `alpha' K alpha`,
in the objective. Algebraic helper variables introduced for solver reformulation
are not penalized; they must be pinned by model equations and residual checks.

## Asset Pricing

- RKHS functions: asset price/costate `p`.
- Algebraic helpers: none.
- Objective: `||p||_H^2`.
- Constraints: collocated asset-pricing ODE `dp/dt = r p - x(t)`.
- Pyomo comparison: current CVXPY QP matches the old Pyomo objective and
  equality constraint.

## Basic Neoclassical Growth

- RKHS functions: capital `k`, consumption `c`, costate `mu`.
- Algebraic helpers: none.
- Objective: `||k||_H^2 + ||c||_H^2 + ||mu||_H^2`.
- Constraints: resource equation, Euler equation, `mu*c = 1`, and positive
  domain guards for `k`, `c`, and `mu`.
- Pyomo comparison: matches the old economic constraints; adding the `c` norm
  removes the former under-identification of consumption coefficients.

## Concave-Convex Growth

- RKHS functions: capital `k`, costate `mu`.
- Algebraic helpers: consumption `c`, production power `z`, output `Y`, and
  marginal product `P`.
- Objective: `||k||_H^2 + ||mu||_H^2`.
- Constraints: solve smooth active-branch candidates for each branch of
  `A max(k^a, b_1 k^a - b_2)`, then accept only candidates that validate
  against the original max problem: resource residual, Euler residual,
  `c*mu = 1`, output binding, and the subgradient interval `m1 <= P <= m2`
  using the marginal product implied by the costate equation. Residuals are
  checked on the collocation grid and a denser validation grid over the training
  horizon; the plotted extrapolation tail is checked for finite positive values.
- Pyomo comparison: preserves the same reduced RKHS functions as the old Pyomo
  model while replacing `Expr_if` with conservative active-branch enumeration
  and ex-post validation. Ambiguous or invalid candidates are rejected rather
  than silently plotted.

## Human Capital

- RKHS functions: physical capital `k`, human capital `h`, physical investment
  `i_k`, human investment `i_h`, consumption `c`, and costates `mu_k`, `mu_h`.
- Algebraic helpers: none.
- Objective: sum of all seven squared RKHS norms.
- Constraints: two accumulation equations, two Euler equations, resource
  feasibility, `mu_k*c = 1`, and `mu_k = mu_h`.
- Pyomo comparison: matches the old constraints and promotes the former small
  smoothing terms for `i_k`, `i_h`, and `c` into the ridgeless norm objective.

## Optimal Advertising

- RKHS functions: market share `x`, costate `mu`, advertising control `u`.
- Algebraic helpers: none.
- Objective: `||x||_H^2 + ||mu||_H^2 + ||u||_H^2`.
- Constraints: market-share dynamics, costate equation, advertising marginal
  condition, and positive control guard `u >= 1e-8`.
- Pyomo comparison: preserves the old equations and adds the missing RKHS norm
  for the kernel-represented control.
