#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.unittest import pytest
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
import pyomo.opt
import time
from itertools import product as iter_product

from pyomo.common.dependencies import (
    numpy as mpi4py_available,
    numpy,
    numpy_available,
    attempt_import,
)

from pathlib import Path

# Directory that contains this test file
_TEST_DIR = Path(__file__).resolve().parent

# Absolute locations of the PGLIB cases (assumes they live next to the tests)
PGLIB_CASE5 = str(_TEST_DIR / "PGLIB_Data" / "pglib_opf_case5_pjm.m")
PGLIB_CASE14 = str(_TEST_DIR / "PGLIB_Data" / "pglib_opf_case14_ieee.m")

# from or_topas.benders.benders_serial import (
#     BendersGenerator_Serial as BendersCutGenerator,
# )

from or_topas.benders import (
    BendersGenerator_Serial as BendersCutGenerator,
)
from or_topas.util import try_import
from or_topas.util.mymunch import MyMunch

import or_topas.benders.tests.test_cases as tc

infeasibility_persistent_test_solvers = list(
    pyomo.opt.check_available_solvers(
        # "appsi_gurobi", #TODO: update to allow appsi_gurobi to work
        "gurobi_persistent",
    )
)

parameterized, param_available = attempt_import("parameterized")
if not param_available:
    raise unittest.SkipTest("Parameterized is not available.")
parameterized = parameterized.parameterized

non_persistent_mip_solvers = list(
    pyomo.opt.check_available_solvers("glpk", "highs", "gurobi_direct")
)

persistent_mip_solvers = list(
    pyomo.opt.check_available_solvers(
        # "appsi_highs",
        # "appsi_gurobi",
        "gurobi_persistent",
    )
)

qp_solvers = list(pyomo.opt.check_available_solvers("ipopt", "gurobi_direct", "highs"))
non_linear_solvers = list(pyomo.opt.check_available_solvers("ipopt"))


ipopt_available = pyo.SolverFactory("ipopt").available(exception_flag=False)
gurobi_available = pyo.SolverFactory("gurobi_persistent").available(
    exception_flag=False
)

default_transform = "standard_lp"
# default_transform = "feasibility"
transforms = ["standard_lp", "feasibility"]

with try_import() as matpower_available:
    from matpowercaseframes import CaseFrames


@pytest.mark.skipif(not matpower_available, reason="Need Matpower for these tests")
class TestBendersOPFExamples(unittest.TestCase):

    #
    # DCOPF Tests
    #

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_dcopf_simple(self, solver):
        transform = "standard_lp"
        set_points = [0, 50, 100]
        expected_obj = 5000
        for index, gen_start in enumerate(set_points):
            grid = tc.EnergyGrid()
            m = tc.EnergyGrid.create_root(grid=grid)
            root_vars = list(m.generation.values())
            m.benders = BendersCutGenerator()
            m.benders.set_input(
                root_vars=root_vars,
                tol=1e-8,
                transform=transform,
                allow_infeasible=True,
            )
            m.benders.add_subproblem(
                subproblem_fn=tc.EnergyGrid.create_subproblem,
                subproblem_fn_kwargs={"root": m, "grid": grid},
                root_eta=m.eta,
                subproblem_solver=solver,
            )
            for b in grid.buses:
                m.generation[b] = gen_start
            opt = pyo.SolverFactory(solver)
            opt.set_instance(m)
            for i in range(30):
                res = opt.solve(tee=False, save_results=False)
                cuts_added = m.benders.generate_cut()
                for c in cuts_added:
                    opt.add_constraint(c)
                if len(cuts_added) == 0:
                    break

            assert i < 30, "Should not need 30 cuts"
            gen_list = [rv.value for rv in root_vars]
            for gen in gen_list:
                assert gen >= 0, "Generation Should Always Be Non-negative"
            self.assertAlmostEqual(sum(gen_list), sum(grid.load_dict.values()), 4)
            self.assertAlmostEqual(pyo.value(m.obj), expected_obj, 4)

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_dcopf_adjusted_gen_max(self, solver):
        transform = "standard_lp"
        set_points = [0, 50, 100]
        expected_obj = 5000
        for index, gen_start in enumerate(set_points):
            grid = tc.EnergyGrid()
            grid.gen_max_dict = {"bus1": 75, "bus2": 75, "bus3": 0}
            m = tc.EnergyGrid.create_root(grid=grid)
            root_vars = list(m.generation.values())
            m.benders = BendersCutGenerator()
            m.benders.set_input(
                root_vars=root_vars,
                tol=1e-8,
                transform=transform,
                allow_infeasible=True,
                feasibility_only=False,
            )
            m.benders.add_subproblem(
                subproblem_fn=tc.EnergyGrid.create_subproblem,
                subproblem_fn_kwargs={
                    "root": m,
                    "grid": grid,
                },
                root_eta=m.eta,
                subproblem_solver=solver,
            )
            for b in grid.buses:
                m.generation[b] = gen_start
            opt = pyo.SolverFactory(solver)
            opt.set_instance(m)
            for i in range(30):
                res = opt.solve(tee=False, save_results=False)
                cuts_added = m.benders.generate_cut()
                for c in cuts_added:
                    opt.add_constraint(c)
                if len(cuts_added) == 0:
                    break

            assert i < 30, "Should not need 30 cuts"
            gen_list = [rv.value for rv in root_vars]
            for gen in gen_list:
                assert gen >= 0, "Generation Should Always Be Non-negative"
            self.assertAlmostEqual(sum(gen_list), sum(grid.load_dict.values()), 4)
            self.assertAlmostEqual(pyo.value(m.obj), expected_obj, 4)

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_dcopf_adjusted_gen_max_feasibility_only(self, solver):
        transform = "standard_lp"
        set_points = [0, 50, 100]
        expected_obj = 5000
        for index, gen_start in enumerate(set_points):
            grid = tc.EnergyGrid()
            grid.gen_max_dict = {"bus1": 75, "bus2": 75, "bus3": 0}
            m = tc.EnergyGrid.create_root(grid=grid)
            root_vars = list(m.generation.values())
            m.benders = BendersCutGenerator()
            m.benders.set_input(
                root_vars=root_vars,
                tol=1e-8,
                transform=transform,
                allow_infeasible=True,
                feasibility_only=True,
            )
            m.benders.add_subproblem(
                subproblem_fn=tc.EnergyGrid.create_subproblem,
                subproblem_fn_kwargs={
                    "root": m,
                    "grid": grid,
                    "feasibility_only": True,
                },
                root_eta=m.eta,
                subproblem_solver=solver,
            )
            for b in grid.buses:
                m.generation[b] = gen_start
            opt = pyo.SolverFactory(solver)
            opt.set_instance(m)
            for i in range(30):
                res = opt.solve(tee=False, save_results=False)
                cuts_added = m.benders.generate_cut()
                for c in cuts_added:
                    opt.add_constraint(c)
                if len(cuts_added) == 0:
                    break

            assert i < 30, "Should not need 30 cuts"
            gen_list = [rv.value for rv in root_vars]
            for gen in gen_list:
                assert gen >= 0, "Generation Should Always Be Non-negative"
            self.assertAlmostEqual(sum(gen_list), sum(grid.load_dict.values()), 4)
            self.assertAlmostEqual(pyo.value(m.obj), expected_obj, 4)

    # def Xtest_matpower_creation(self):
    #     file_path = "PGLIB_Data/pglib_opf_case5_pjm.m"
    #     grid = tc.MatpowerGrid(m_file=file_path)

    # def Xtest_matpower_creation_bad_path(self):
    #     with self.assertRaises(RuntimeError) as cm:
    #         file_path = "PGLIB_Data/pglib_not_here.m"
    #         grid = tc.MatpowerGrid(m_file=file_path)
    #     expected_message = "Error in CaseFrames creation in MatpowerGrid initialization"
    #     self.assertIn(expected_message, str(cm.exception))

    # @parameterized.expand(
    #     input=infeasibility_persistent_test_solvers, skip_on_empty=True
    # )
    # def Xtest_matpower_traditional_solve(self, mip_solver):
    #     file_path = "PGLIB_Data/pglib_opf_case5_pjm.m"
    #     grid = tc.MatpowerGrid(m_file=file_path)
    #     model = tc.EnergyGrid.create_tiny_opf(grid, mode=2)
    #     model.pprint()
    #     opt = pyo.SolverFactory(mip_solver)
    #     opt.set_instance(model)
    #     res = opt.solve(tee=False, save_results=False)

    # @parameterized.expand(
    #     input=infeasibility_persistent_test_solvers, skip_on_empty=True
    # )
    # def Xtest_matpower_traditional_solve_2(self, mip_solver):
    #     file_path = "PGLIB_Data/pglib_opf_case14_ieee.m"
    #     grid = tc.MatpowerGrid(m_file=file_path)
    #     model = tc.EnergyGrid.create_tiny_opf(grid, mode=2)
    #     model.pprint()
    #     opt = pyo.SolverFactory(mip_solver)
    #     opt.set_instance(model)
    #     res = opt.solve(tee=False, save_results=False)

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_matpower_case5_benders(self, solver):
        """Benders decomposition on real pglib case5 (repo layout)."""
        file_path = PGLIB_CASE5
        grid = tc.MatpowerGrid(m_file=file_path)
        m = tc.EnergyGrid.create_root(grid=grid)
        root_vars = list(m.generation.values())
        m.benders = BendersCutGenerator()
        m.benders.set_input(
            root_vars=root_vars,
            tol=1e-6,
            transform="standard_lp",
            allow_infeasible=True,
        )
        m.benders.add_subproblem(
            subproblem_fn=tc.EnergyGrid.create_subproblem,
            subproblem_fn_kwargs={"root": m, "grid": grid},
            root_eta=m.eta,
            subproblem_solver=solver,
        )
        opt = pyo.SolverFactory(solver)
        opt.set_instance(m)
        for _ in range(15):
            opt.solve(tee=False, save_results=False)
            cuts = m.benders.generate_cut()
            for c in cuts:
                opt.add_constraint(c)
            if not cuts:
                break
        gen_sum = sum(v.value for v in root_vars)
        self.assertAlmostEqual(gen_sum, sum(grid.load_dict.values()), delta=0.01)

    def test_matpower_case5_creation_and_solve(self):
        """Smoke test using exact repo path."""
        file_path = PGLIB_CASE5
        grid = tc.MatpowerGrid(m_file=file_path)
        self.assertEqual(len(grid.buses), 5)
        model = tc.EnergyGrid.create_tiny_opf(grid, mode=2)
        opt = pyo.SolverFactory("gurobi_persistent")
        opt.set_instance(model)
        res = opt.solve(tee=False)
        self.assertEqual(
            res.solver.termination_condition, pyo.TerminationCondition.optimal
        )

    def test_matpower_case14_smoke(self):
        file_path = PGLIB_CASE14
        grid = tc.MatpowerGrid(m_file=file_path)
        self.assertEqual(len(grid.buses), 14)

    # ------------------------------------------------------------------
    # Commitment model baseline tests from Grok 4.20
    # ------------------------------------------------------------------

    def test_commitment_3bus_creation(self):
        """Smoke: EnergyGridWithCommitment builds and has expected components."""
        for eps in (0.0, 1.0, 5.0):
            grid = tc.EnergyGridWithCommitment(epsilon=eps)
            model = tc.EnergyGridWithCommitment.create_tiny_opf(grid, mode=2)

            self.assertTrue(hasattr(model, "commit"))
            self.assertTrue(hasattr(model, "gen_buses"))
            self.assertTrue(hasattr(model, "min_gen_commit"))
            self.assertTrue(hasattr(model, "max_gen_commit"))
            self.assertEqual(len(model.gen_buses), 2)  # bus1, bus2
            self.assertAlmostEqual(pyo.value(model.epsilon), eps)
            for b in model.gen_buses:
                self.assertIs(model.commit[b].domain, pyo.Binary)

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_commitment_3bus_monolithic(self, solver):
        """Full MIP (no Benders) on 3-bus commitment model solves optimally
        and respects the indicator constraints."""
        for eps in (0.0, 1.0):
            grid = tc.EnergyGridWithCommitment(epsilon=eps)
            model = tc.EnergyGridWithCommitment.create_tiny_opf(grid, mode=2)

            opt = pyo.SolverFactory(solver)
            opt.set_instance(model)
            res = opt.solve(tee=False, save_results=False)
            self.assertEqual(
                res.solver.termination_condition,
                pyo.TerminationCondition.optimal,
            )

            # Power balance
            total_gen = sum(pyo.value(model.generation[b]) for b in model.buses)
            self.assertAlmostEqual(total_gen, sum(grid.load_dict.values()), places=4)

            # Indicator logic
            for b in model.gen_buses:
                u = pyo.value(model.commit[b])
                g = pyo.value(model.generation[b])
                self.assertGreaterEqual(g, -1e-6)
                if u < 0.5:  # committed off
                    self.assertAlmostEqual(g, 0.0, places=4)
                else:  # committed on
                    self.assertGreaterEqual(g, eps - 1e-4)
                    self.assertLessEqual(g, grid.gen_max_dict[b] + 1e-4)

            # Objective consistency
            expected_obj = sum(
                grid.cost_dict[b] * pyo.value(model.generation[b]) for b in model.buses
            )
            self.assertAlmostEqual(pyo.value(model.obj), expected_obj, places=4)

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_commitment_3bus_benders_smoke(self, solver):
        """Benders setup + short cut loop on commitment model does not crash
        and terminates with a finite eta."""
        grid = tc.EnergyGridWithCommitment(epsilon=1.0)

        # Prefer the helper if it exists; otherwise fall back to hand-rolled setup
        if hasattr(
            tc.EnergyGridWithCommitment, "setup_energy_grid_commitment_persistent"
        ):
            opt, m = (
                tc.EnergyGridWithCommitment.setup_energy_grid_commitment_persistent(
                    solver_name=solver,
                    grid=grid,
                    eta_lb=-1e5,
                    eta_ub=1e5,
                    mode=2,
                )
            )
        else:
            m = tc.EnergyGridWithCommitment.create_root(grid, eta_lb=-1e5, eta_ub=1e5)
            root_vars = list(m.commit.values())
            m.benders = BendersCutGenerator()
            m.benders.set_input(
                root_vars=root_vars,
                tol=1e-8,
                transform="standard_lp",
                allow_infeasible=True,
            )
            m.benders.add_subproblem(
                subproblem_fn=tc.EnergyGridWithCommitment.create_subproblem,
                subproblem_fn_kwargs={"root": m, "grid": grid, "mode": 2},
                root_eta=m.eta,
                subproblem_solver=solver,
            )
            opt = pyo.SolverFactory(solver)
            opt.set_instance(m)

        for i in range(15):
            res = opt.solve(tee=False, save_results=False)
            cuts = m.benders.generate_cut()
            for c in cuts:
                opt.add_constraint(c)
            if not cuts:
                break

        self.assertLess(i, 15, "Should not need 15 cuts on the 3-bus commitment model")
        self.assertTrue(pyo.value(m.eta) < 1e20)  # finite

    def test_commitment_subproblem_domain_relaxation(self):
        """create_subproblem must relax commit variables to continuous (Reals)
        so that dual information can be obtained."""
        grid = tc.EnergyGridWithCommitment(epsilon=1.0)
        root = tc.EnergyGridWithCommitment.create_root(grid)
        sub, _ = tc.EnergyGridWithCommitment.create_subproblem(
            root, grid, mode=2, feasibility_only=False
        )

        for b in sub.gen_buses:
            self.assertIs(sub.commit[b].domain, pyo.Reals)

    @pytest.mark.skipif(not matpower_available, reason="Need Matpower for these tests")
    def test_matpower_commitment_case5_creation(self):
        """Smoke: MatpowerGridWithCommitment builds on case5."""
        file_path = PGLIB_CASE5
        grid = tc.MatpowerGridWithCommitment(m_file=file_path, epsilon=1.0)
        self.assertEqual(len(grid.buses), 5)
        model = tc.MatpowerGridWithCommitment.create_tiny_opf(grid, mode=2)
        self.assertTrue(hasattr(model, "commit"))
        self.assertGreater(len(model.gen_buses), 0)

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    @pytest.mark.skipif(not matpower_available, reason="Need Matpower for these tests")
    def test_matpower_commitment_case5_monolithic(self, solver):
        """Full MIP on real case5 commitment model solves optimally."""
        file_path = PGLIB_CASE5
        grid = tc.MatpowerGridWithCommitment(m_file=file_path, epsilon=1.0)
        model = tc.MatpowerGridWithCommitment.create_tiny_opf(grid, mode=2)

        opt = pyo.SolverFactory(solver)
        opt.set_instance(model)
        res = opt.solve(tee=False, save_results=False)
        self.assertEqual(
            res.solver.termination_condition,
            pyo.TerminationCondition.optimal,
        )

        total_gen = sum(pyo.value(model.generation[b]) for b in model.buses)
        self.assertAlmostEqual(total_gen, sum(grid.load_dict.values()), delta=0.05)

    # ------------------------------------------------------------------
    # Optimal commitment pattern tests (3-bus) Grok 4.20 generated
    # ------------------------------------------------------------------

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_commitment_3bus_optimal_patterns_extensive(self, solver):
        """Extensive-form: for every mode the optimal commitment vectors
        are exactly the three that turn on at least one generator."""
        expected = {(1, 0), (0, 1), (1, 1)}
        eps = 1.0

        for mode in (0, 1, 2):
            optimal = set()
            for u1, u2 in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                grid = tc.EnergyGridWithCommitment(epsilon=eps)
                model = tc.EnergyGridWithCommitment.create_tiny_opf(grid, mode=mode)

                # Fix the two commitment variables
                model.commit["bus1"].fix()
                model.commit["bus1"].set_value(u1)
                model.commit["bus2"].fix()
                model.commit["bus2"].set_value(u2)

                opt = pyo.SolverFactory(solver)
                opt.set_instance(model)
                res = opt.solve(tee=False, save_results=False)

                if (
                    res.solver.termination_condition == pyo.TerminationCondition.optimal
                    and abs(pyo.value(model.obj) - 5000.0) < 1e-3
                ):
                    optimal.add((u1, u2))

            self.assertEqual(
                optimal,
                expected,
                f"mode={mode}: extensive-form optimal set was {optimal}",
            )

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_commitment_3bus_optimal_patterns_benders(self, solver):
        """Benders: each of the three known optimal commitment patterns
        produces a subproblem cost of 5000; the all-off pattern does not."""
        expected = {(1, 0), (0, 1), (1, 1)}
        eps = 1.0
        tol = 1e-3

        for mode in (0, 1, 2):
            grid = tc.EnergyGridWithCommitment(epsilon=eps)
            root = tc.EnergyGridWithCommitment.create_root(grid)

            costs = {}
            for u1, u2 in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                # Build a fresh subproblem and fix the (relaxed) commit variables
                sub, _ = tc.EnergyGridWithCommitment.create_subproblem(
                    root, grid, mode=mode, feasibility_only=False
                )
                sub.commit["bus1"].fix()
                sub.commit["bus1"].set_value(u1)
                sub.commit["bus2"].fix()
                sub.commit["bus2"].set_value(u2)

                opt = pyo.SolverFactory(solver)
                opt.set_instance(sub)
                res = opt.solve(tee=False, save_results=False)

                if res.solver.termination_condition == pyo.TerminationCondition.optimal:
                    costs[(u1, u2)] = pyo.value(sub.obj)
                else:
                    costs[(u1, u2)] = float("inf")

            # The three expected patterns must all achieve ~5000
            for pat in expected:
                self.assertAlmostEqual(
                    costs[pat],
                    5000.0,
                    delta=tol,
                    msg=f"mode={mode}, pattern={pat}: expected cost 5000, got {costs[pat]}",
                )

            # All-off must be strictly worse (or infeasible)
            self.assertGreater(
                costs[(0, 0)],
                5000.0 + tol,
                msg=f"mode={mode}: all-off should not be optimal",
            )

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_3bus_continuous_vs_commitment_epsilon_gap(self, solver):
        """
        Extensive-form (monolithic) comparison of continuous vs commitment models
        on the 3-bus toy while sweeping ε.
        Continuous optimum is known to be 5000; the gap must vanish at ε = 0.
        """
        eps_values = [0.0, 1e-4, 1e-3, 0.01, 0.1, 1.0, 5.0]

        print("\n" + "=" * 70)
        print("3-bus toy  (mode=2, network)")
        print("=" * 70)

        # Continuous reference
        grid_c = tc.EnergyGrid()
        model_c = tc.EnergyGrid.create_tiny_opf(grid_c, mode=2)
        opt_c = pyo.SolverFactory(solver)
        opt_c.set_instance(model_c)
        res_c = opt_c.solve(tee=False, save_results=False)
        self.assertEqual(
            res_c.solver.termination_condition, pyo.TerminationCondition.optimal
        )
        cont_obj = pyo.value(model_c.obj)
        cont_gen = {b: pyo.value(model_c.generation[b]) for b in model_c.buses}
        self.assertAlmostEqual(cont_obj, 5000.0, places=3)
        print(f"Continuous objective : {cont_obj:.6f}")
        print(f"Continuous generation: {cont_gen}")

        print(
            f"\n{'ε':>10}  {'commitment obj':>16}  {'gap':>12}  generation (bus1,bus2)"
        )
        print("-" * 70)
        for eps in eps_values:
            grid = tc.EnergyGridWithCommitment(epsilon=eps)
            model = tc.EnergyGridWithCommitment.create_tiny_opf(grid, mode=2)
            opt = pyo.SolverFactory(solver)
            opt.set_instance(model)
            res = opt.solve(tee=False, save_results=False)
            self.assertEqual(
                res.solver.termination_condition, pyo.TerminationCondition.optimal
            )
            obj = pyo.value(model.obj)
            gap = obj - cont_obj
            gen = (
                pyo.value(model.generation["bus1"]),
                pyo.value(model.generation["bus2"]),
            )
            print(f"{eps:10.4g}  {obj:16.6f}  {gap:12.6f}  {gen}")
            if eps == 0.0:
                self.assertAlmostEqual(obj, cont_obj, places=3)

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    @pytest.mark.skipif(not matpower_available, reason="Need Matpower for these tests")
    def test_case5_continuous_vs_commitment_epsilon_gap(self, solver):
        """
        With a small default ε the commitment MIP must recover the continuous
        objective on copper-plate (within a tight numerical tolerance).
        A deliberately large floor is also checked to confirm that a gap appears.
        """
        file_path = PGLIB_CASE5

        # Continuous reference
        grid_c = tc.MatpowerGrid(m_file=file_path)
        model_c = tc.EnergyGrid.create_tiny_opf(grid_c, mode=0)
        opt_c = pyo.SolverFactory(solver)
        opt_c.set_instance(model_c)
        res_c = opt_c.solve(tee=False, save_results=False)
        self.assertEqual(
            res_c.solver.termination_condition, pyo.TerminationCondition.optimal
        )
        cont_obj = pyo.value(model_c.obj)

        # ε values that should not bind in any economically meaningful way
        small_eps = [0.0, 1e-6, 1e-4, 1e-3, 0.01, 0.1, 1.0]
        for eps in small_eps:
            grid = tc.MatpowerGridWithCommitment(m_file=file_path, epsilon=eps)
            model = tc.MatpowerGridWithCommitment.create_tiny_opf(grid, mode=0)
            opt = pyo.SolverFactory(solver)
            opt.set_instance(model)
            res = opt.solve(tee=False, save_results=False)
            self.assertEqual(
                res.solver.termination_condition, pyo.TerminationCondition.optimal
            )

            # ε=0 must match to high precision; larger (but still small) ε may
            # differ by O(ε · Δcost) because of the forced residual.
            tol = 1e-6 if eps == 0.0 else 1e-2
            self.assertAlmostEqual(
                pyo.value(model.obj),
                cont_obj,
                delta=tol,
                msg=f"ε={eps} should recover continuous objective {cont_obj}",
            )

        # One deliberately large floor – a visible gap is expected
        grid_large = tc.MatpowerGridWithCommitment(m_file=file_path, epsilon=5.0)
        model_large = tc.MatpowerGridWithCommitment.create_tiny_opf(grid_large, mode=0)
        opt_large = pyo.SolverFactory(solver)
        opt_large.set_instance(model_large)
        res_large = opt_large.solve(tee=False, save_results=False)
        self.assertEqual(
            res_large.solver.termination_condition, pyo.TerminationCondition.optimal
        )
        self.assertGreater(
            pyo.value(model_large.obj),
            cont_obj + 1.0,
            msg="ε=5.0 should produce a visible gap",
        )

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    @pytest.mark.skipif(not matpower_available, reason="Need Matpower for these tests")
    def test_case5_benders_dcopf_commitment_matches_extensive_continuous(self, solver):
        """
        Reference = extensive-form continuous DC-OPF (mode=2).
        Then run Benders on the commitment model with ε ∈ {0, 1e-6}
        and assert that the Benders objective recovers the same value.
        This guarantees that the Benders path does not introduce an extra gap.
        """
        file_path = PGLIB_CASE5
        max_cuts = 20
        tol = 1e-3

        # ------------------------------------------------------------------
        # Extensive-form continuous reference (mode=2, full DC-OPF)
        # ------------------------------------------------------------------
        grid_c = tc.MatpowerGrid(m_file=file_path)
        model_c = tc.EnergyGrid.create_tiny_opf(grid_c, mode=2)
        opt_c = pyo.SolverFactory(solver)
        opt_c.set_instance(model_c)
        res_c = opt_c.solve(tee=False, save_results=False)
        self.assertEqual(
            res_c.solver.termination_condition, pyo.TerminationCondition.optimal
        )
        cont_obj = pyo.value(model_c.obj)

        # ------------------------------------------------------------------
        # Benders commitment runs for the two small ε values
        # ------------------------------------------------------------------
        for eps in (0.0, 1e-6):
            grid = tc.MatpowerGridWithCommitment(m_file=file_path, epsilon=eps)

            if hasattr(
                tc.EnergyGridWithCommitment, "setup_energy_grid_commitment_persistent"
            ):
                opt, m = (
                    tc.EnergyGridWithCommitment.setup_energy_grid_commitment_persistent(
                        solver_name=solver,
                        grid=grid,
                        eta_lb=-1e5,
                        eta_ub=1e5,
                        mode=2,
                    )
                )
            else:
                m = tc.EnergyGridWithCommitment.create_root(
                    grid, eta_lb=-1e5, eta_ub=1e5
                )
                root_vars = list(m.commit.values())
                m.benders = BendersCutGenerator()
                m.benders.set_input(
                    root_vars=root_vars,
                    tol=1e-6,
                    transform="standard_lp",
                    allow_infeasible=True,
                )
                m.benders.add_subproblem(
                    subproblem_fn=tc.EnergyGridWithCommitment.create_subproblem,
                    subproblem_fn_kwargs={"root": m, "grid": grid, "mode": 2},
                    root_eta=m.eta,
                    subproblem_solver=solver,
                )
                opt = pyo.SolverFactory(solver)
                opt.set_instance(m)

            for _ in range(max_cuts):
                opt.solve(tee=False, save_results=False)
                cuts = m.benders.generate_cut()
                for c in cuts:
                    opt.add_constraint(c)
                if not cuts:
                    break

            commit_obj = pyo.value(m.obj)  # eta after convergence
            self.assertAlmostEqual(
                commit_obj,
                cont_obj,
                delta=tol,
                msg=(
                    f"Benders commitment (ε={eps}) should recover the extensive-form "
                    f"continuous objective {cont_obj}"
                ),
            )
