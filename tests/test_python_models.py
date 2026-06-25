import importlib
import unittest

import jax.numpy as jnp

from asset_pricing_matern import asset_pricing_matern
from neoclassical_growth_concave_convex_matern import (
    neoclassical_growth_concave_convex_matern,
)
from neoclassical_growth_matern import neoclassical_growth_matern
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
        sol = asset_pricing_matern()

        assert_all_finite(self, sol["p_test"])
        assert_rkhs_norms(self, sol, {"p"})
        self.assertEqual(sol["solver_status"], "SUCCESS")
        self.assertLess(sol["max_train_residual"], 1e-10)
        self.assertLess(float(sol["p_rel_error"].mean()), 5e-3)

    def test_neoclassical_growth_smoke(self):
        sol = neoclassical_growth_matern()

        assert_all_finite(self, sol["k_test"])
        assert_all_finite(self, sol["c_test"])
        assert_rkhs_norms(self, sol, {"k", "c", "mu"})
        self.assertEqual(sol["solver_status"], "SUCCESS")
        self.assertTrue(sol["valid_solution"], sol["rejection_reason"])
        self.assertTrue(bool(jnp.all(sol["k_test"] > 0.0)))
        self.assertTrue(bool(jnp.all(sol["c_test"] > 0.0)))
        self.assertLess(float(sol["k_rel_error"].mean()), 2e-2)
        self.assertLess(float(sol["c_rel_error"].mean()), 2e-2)
        self.assertLess(sol["max_train_residual"], 1e-8)

    def test_neoclassical_growth_robustness_cases(self):
        cases = [
            {},
            {"nu": 1.5},
            {"nu": 2.5},
            {"rho": 2},
            {"rho": 20},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                sol = neoclassical_growth_matern(**kwargs)

                self.assertEqual(sol["solver_status"], "SUCCESS")
                self.assertTrue(sol["valid_solution"], sol["rejection_reason"])
                self.assertLess(sol["max_train_residual"], 1e-8)
                self.assertLess(float(jnp.mean(sol["k_rel_error"])), 2e-2)
                self.assertLess(float(jnp.mean(sol["c_rel_error"])), 2e-2)

    def test_concave_convex_smoke(self):
        sol = neoclassical_growth_concave_convex_matern()

        assert_all_finite(self, sol["k_test"])
        assert_all_finite(self, sol["c_test"])
        assert_rkhs_norms(self, sol, {"k", "c", "mu"})
        self.assertTrue(sol["valid_solution"], sol["rejection_reason"])
        self.assertEqual(sol["solver_status"], "SUCCESS")
        self.assertTrue(bool(jnp.all(sol["k_test"] > 0.0)))
        self.assertTrue(bool(jnp.all(sol["c_test"] > 0.0)))
        self.assertLess(sol["max_train_residual"], 1e-5)
        self.assertLess(sol["max_validation_residual"], 5e-3)
        self.assertLess(sol["p_lower_violation"], 1e-3)
        self.assertLess(sol["p_upper_violation"], 1e-3)
        self.assertLess(sol["solve_time"], 0.5)

    def test_concave_convex_interior_cases(self):
        for k_0 in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
            with self.subTest(k_0=k_0):
                sol = neoclassical_growth_concave_convex_matern(k_0=k_0)

                self.assertTrue(sol["valid_solution"], sol["rejection_reason"])
                assert_all_finite(self, sol["k_test"])
                assert_all_finite(self, sol["c_test"])
                self.assertTrue(bool(jnp.all(sol["k_test"] > 0.0)))
                self.assertTrue(bool(jnp.all(sol["c_test"] > 0.0)))
                self.assertLess(sol["max_train_residual"], 1e-5)
                self.assertLess(sol["max_validation_residual"], 5e-3)
                self.assertLess(sol["p_lower_violation"], 1e-3)
                self.assertLess(sol["p_upper_violation"], 1e-3)
                self.assertLess(sol["solve_time"], 0.5)

    def test_concave_convex_boundary_cases_fail_fast_or_validate(self):
        for k_0 in [1.0384615384615383, 1.04, 1.05]:
            with self.subTest(k_0=k_0):
                sol = neoclassical_growth_concave_convex_matern(k_0=k_0)

                if sol["valid_solution"]:
                    self.assertLess(sol["max_train_residual"], 1e-5)
                    self.assertLess(sol["max_validation_residual"], 5e-3)
                    self.assertLess(sol["p_lower_violation"], 1e-3)
                    self.assertLess(sol["p_upper_violation"], 1e-3)
                else:
                    self.assertNotEqual(sol["rejection_reason"], "accepted")
                self.assertLess(sol["solve_time"], 0.5)

    def test_optimal_advertising_smoke(self):
        sol = optimal_advertising_matern()

        assert_all_finite(self, sol["x_test"])
        assert_all_finite(self, sol["mu_test"])
        assert_all_finite(self, sol["u_test"])
        assert_rkhs_norms(self, sol, {"x", "mu", "u"})
        self.assertEqual(sol["solver_status"], "SUCCESS")
        self.assertTrue(bool(jnp.all(sol["u_test"] > 0.0)))
        self.assertLess(sol["max_train_residual"], 1e-8)

    def test_human_capital_smoke(self):
        sol = human_capital_matern()

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
        self.assertTrue(sol["valid_solution"], sol["rejection_reason"])
        self.assertEqual(sol["solver_status"], "SUCCESS")
        self.assertAlmostEqual(float(sol["h_0"]), 1.3745155888757778, places=12)
        self.assertLess(
            float(jnp.linalg.norm(sol["initial_condition_residual"], ord=jnp.inf)),
            1e-12,
        )
        self.assertLess(sol["max_train_residual"], 1e-8)
        self.assertLess(sol["max_validation_residual"], 1e-3)
        self.assertLess(float(jnp.max(jnp.abs(sol["feasibility_test"]))), 1e-3)
        self.assertLess(
            float(jnp.max(jnp.abs(sol["hidden_dae_residual_test"]))), 1e-5
        )
        self.assertLess(
            float(jnp.max(jnp.abs(sol["mu_k_test"] - sol["mu_h_test"]))), 1e-4
        )


if __name__ == "__main__":
    unittest.main()
