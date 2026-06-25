# RKHS Formulation Audit

This repository uses ridgeless RKHS collocation: each independent kernel
function is identified by including its squared RKHS norm, `alpha' K alpha`,
in the objective. The active implementation uses direct JAX callbacks into UNO.

## Asset Pricing

- RKHS functions: asset price/costate `p`.
- Algebraic helpers: none.
- Objective: `||p||_H^2`.
- Constraints: collocated asset-pricing ODE `dp/dt = r p - x(t)`.
- Solver implementation: direct JAX/UNO quadratic model.

## Basic Neoclassical Growth

- RKHS functions: capital `k`, consumption `c`, costate `mu`.
- Algebraic helpers: none.
- Objective: `||k||_H^2 + ||c||_H^2 + ||mu||_H^2`.
- Constraints: resource equation, Euler equation, `mu*c = 1`, and positive
  domain guards for `k`, `c`, and `mu`.
- Solver implementation: direct JAX/UNO nonlinear model; adding the `c` norm
  removes the former under-identification of consumption coefficients.

## Concave-Convex Growth

- RKHS functions: capital `k`, consumption `c`, costate `mu`.
- Algebraic helpers: diagnostic production power `z`, output `Y`, and marginal
  product `P` returned after the solve.
- Objective: `||k||_H^2 + ||c||_H^2 + ||mu||_H^2`.
- Constraints: same economic equations as basic growth with production replaced
  by `A max(k^a, b_1 k^a - b_2)`. JAX traces the max production function and its
  marginal product directly; the solver does not use steady states, basin
  thresholds, or branch enumeration. Residuals are checked on the collocation
  grid and a denser validation grid; ambiguous crossing/local-solve failures
  are rejected quickly rather than silently plotted.
- Solver implementation: thin public wrapper around `neoclassical_growth_matern`
  with kinked production parameters.

## Human Capital

- RKHS functions: physical capital `k`, human capital `h`, physical investment
  `i_k`, human investment `i_h`, consumption `c`, and costates `mu_k`, `mu_h`.
- Algebraic helpers: none.
- Objective: sum of all seven squared RKHS norms.
- Constraints: two accumulation equations, two Euler equations, resource
  feasibility, `mu_k*c = 1`, and `mu_k = mu_h`.
- Solver implementation: direct JAX/UNO nonlinear model with the small
  initial-condition system solved in JAX through `nlls_gram`.

## Optimal Advertising

- RKHS functions: market share `x`, costate `mu`, advertising control `u`.
- Algebraic helpers: none.
- Objective: `||x||_H^2 + ||mu||_H^2 + ||u||_H^2`.
- Constraints: market-share dynamics, costate equation, advertising marginal
  condition, and positive control guard `u >= 1e-8`.
- Solver implementation: direct JAX/UNO nonlinear model including the RKHS norm
  for the kernel-represented control.
