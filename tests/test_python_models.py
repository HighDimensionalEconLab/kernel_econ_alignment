import importlib
import unittest

import jax.numpy as jnp

from asset_pricing_matern import asset_pricing_matern
from neoclassical_growth_matern import neoclassical_growth_matern
from neoclassical_growth_concave_convex_matern import (
    neoclassical_growth_concave_convex_matern,
)
from neoclassical_human_capital_matern import (
    human_capital_initial_conditions,
    human_capital_matern,
)
from optimal_advertising_matern import optimal_advertising_matern


def assert_all_finite(testcase, array):
    testcase.assertTrue(bool(jnp.all(jnp.isfinite(array))))


def assert_rkhs_norms(testcase, sol, expected_keys):
    testcase.assertEqual(set(sol["rkhs_norms"]), set(expected_keys))
    for value in sol["rkhs_norms"].values():
        testcase.assertTrue(bool(jnp.isfinite(jnp.asarray(value))))
        testcase.assertGreaterEqual(value, -1e-10)


class ModelSmokeTests(unittest.TestCase):
    def test_figure_modules_import(self):
        for module_name in [
            "figures_asset_pricing",
            "figures_neoclassical_growth_baseline",
            "figures_neoclassical_growth_robustness",
            "figures_neoclassical_growth_concave_convex",
            "figures_optimal_advertising",
            "figures_neoclassical_human_capital",
            "tables_neoclassical_growth",
        ]:
            with self.subTest(module_name=module_name):
                importlib.import_module(module_name)

    def test_human_capital_initial_conditions(self):
        h_0, c_0, residual, _ = human_capital_initial_conditions()

        self.assertEqual(h_0.dtype, jnp.float64)
        self.assertEqual(c_0.dtype, jnp.float64)
        self.assertAlmostEqual(float(h_0), 1.3745155888757778, places=12)
        self.assertLess(float(jnp.linalg.norm(residual, ord=jnp.inf)), 1e-12)

    def test_asset_pricing_smoke(self):
        sol = asset_pricing_matern(train_points=11, test_points=10)

        assert_all_finite(self, sol["p_test"])
        assert_rkhs_norms(self, sol, {"p"})
        self.assertLess(float(sol["p_rel_error"].mean()), 5e-3)

    def test_neoclassical_growth_smoke(self):
        sol = neoclassical_growth_matern(
            train_points=11,
            test_points=10,
            benchmark_points=80,
        )

        assert_all_finite(self, sol["k_test"])
        assert_all_finite(self, sol["c_test"])
        assert_rkhs_norms(self, sol, {"k", "c", "mu"})
        self.assertTrue(bool(jnp.all(sol["k_test"] > 0.0)))
        self.assertTrue(bool(jnp.all(sol["c_test"] > 0.0)))
        self.assertLess(float(sol["k_rel_error"].mean()), 2e-2)
        self.assertLess(float(sol["c_rel_error"].mean()), 2e-2)

    def test_concave_convex_smoke(self):
        sol = neoclassical_growth_concave_convex_matern(
            train_points=11,
            test_points=10,
            benchmark_points=80,
        )

        assert_all_finite(self, sol["k_test"])
        assert_all_finite(self, sol["c_test"])
        assert_rkhs_norms(self, sol, {"k", "mu"})
        self.assertTrue(bool(jnp.all(sol["k_test"] > 0.0)))
        self.assertTrue(bool(jnp.all(sol["c_test"] > 0.0)))
        for residual in sol["helper_residuals"].values():
            assert_all_finite(self, residual)
            self.assertLess(float(jnp.max(jnp.abs(residual))), 1e-5)

    def test_optimal_advertising_smoke(self):
        sol = optimal_advertising_matern(
            train_points=11,
            test_points=10,
            benchmark_points=80,
        )

        assert_all_finite(self, sol["x_test"])
        assert_all_finite(self, sol["mu_test"])
        assert_all_finite(self, sol["u_test"])
        assert_rkhs_norms(self, sol, {"x", "mu", "u"})
        self.assertTrue(bool(jnp.all(sol["u_test"] > 0.0)))

    def test_human_capital_smoke(self):
        sol = human_capital_matern(
            train_points=11,
            test_points=10,
            benchmark_points=80,
        )

        for key in [
            "k_test",
            "h_test",
            "c_test",
            "i_k_test",
            "i_h_test",
            "mu_k_test",
            "mu_h_test",
            "feasibility_test",
            "hidden_dae_residual_test",
        ]:
            assert_all_finite(self, sol[key])
        assert_rkhs_norms(
            self, sol, {"k", "h", "i_k", "i_h", "c", "mu_k", "mu_h"}
        )
        self.assertLess(float(jnp.max(jnp.abs(sol["feasibility_test"]))), 5e-2)
        self.assertLess(
            float(jnp.max(jnp.abs(sol["hidden_dae_residual_test"]))), 1e-3
        )
        self.assertLess(
            float(jnp.max(jnp.abs(sol["mu_k_test"] - sol["mu_h_test"]))), 1e-6
        )


if __name__ == "__main__":
    unittest.main()
