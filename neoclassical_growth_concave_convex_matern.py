from typing import List, Optional

import jsonargparse

from neoclassical_growth_matern import neoclassical_growth_matern


def neoclassical_growth_concave_convex_matern(
    a: float = 1 / 3,
    delta: float = 0.1,
    rho_hat: float = 0.11,
    A: float = 0.5,
    b_1: float = 3.0,
    b_2: float = 2.5,
    k_0: float = 1.0,
    nu: float = 0.5,
    sigma: float = 1.0,
    rho: float = 10,
    train_T: float = 40.0,
    train_points: int = 41,
    test_T: float = 50.0,
    test_points: int = 41,
    benchmark_T: float = 60.0,
    benchmark_points: int = 300,
    train_points_list: Optional[List[float]] = None,
    verbose: bool = False,
):
    return neoclassical_growth_matern(
        a=a,
        delta=delta,
        rho_hat=rho_hat,
        k_0=k_0,
        A=A,
        b_1=b_1,
        b_2=b_2,
        nu=nu,
        sigma=sigma,
        rho=rho,
        train_T=train_T,
        train_points=train_points,
        test_T=test_T,
        test_points=test_points,
        benchmark_T=benchmark_T,
        benchmark_points=benchmark_points,
        train_points_list=train_points_list,
        verbose=verbose,
    )


if __name__ == "__main__":
    jsonargparse.CLI(neoclassical_growth_concave_convex_matern)
