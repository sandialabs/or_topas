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
from or_topas.benders.benders_serial import (
    BendersGenerator_Serial as BendersCutGenerator,
)
import logging
import or_topas.benders.tests.test_cases as tc
from or_topas.util.mymunch import MyMunch
from pyomo.repn.standard_repn import generate_standard_repn

parameterized, param_available = attempt_import("parameterized")
if not param_available:
    raise unittest.SkipTest("Parameterized is not available.")
parameterized = parameterized.parameterized

non_persistent_mip_solvers = list(
    pyomo.opt.check_available_solvers("glpk", "highs", "gurobi_direct")
)
infeasibility_test_solvers = list(
    pyomo.opt.check_available_solvers("glpk", "gurobi_direct")
)

infeasibility_persistent_test_solvers = list(
    pyomo.opt.check_available_solvers(
        # "appsi_gurobi", #TODO: update to allow appsi_gurobi to work
        "gurobi_persistent",
    )
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


class TestBendersUtils(unittest.TestCase):
    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    def test_farmer(self, mip_solver):

        t = True
        assert t, "Trivial Test"

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_abs(self, solver):
        transform = "standard_lp"
        m = tc.absolute_value.create_root()
        root_vars = [m.x]
        m.benders = BendersCutGenerator()
        m.benders.set_input(root_vars=root_vars, tol=1e-8, transform=transform)
        m.benders.add_subproblem(
            subproblem_fn=tc.absolute_value.create_subproblem,
            subproblem_fn_kwargs={"root": m},
            root_eta=m.eta,
            subproblem_solver=solver,
        )
        opt = pyo.SolverFactory(solver)

        for i in range(30):
            res = opt.solve(m, tee=False)
            cuts_added = m.benders.generate_cut()
            if len(cuts_added) == 0:
                break
        self.assertAlmostEqual(m.x.value, 0.0, 4)
        self.assertAlmostEqual(pyo.value(m.obj), 0.0, 4)
        self.assertAlmostEqual(m.eta.value, 0.0, 4)

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_modified_absolute_value(self, solver):
        transform = "standard_lp"
        a_set = [-1, 3, -2.2, 4.99]
        for a in a_set:
            m = tc.modified_absolute_value.create_root()
            root_vars = [m.x]
            data = MyMunch(a=a, L=1, R=1, LB=None, UB=None)
            m.benders = BendersCutGenerator()
            m.benders.set_input(root_vars=root_vars, tol=1e-8, transform=transform)
            m.benders.add_subproblem(
                subproblem_fn=tc.modified_absolute_value.create_subproblem,
                subproblem_fn_kwargs={"root_x": m.x, "data": data},
                root_eta=m.eta,
                subproblem_solver=solver,
            )
            opt = pyo.SolverFactory(solver)

            for i in range(30):
                res = opt.solve(m, tee=False)
                cuts_added = m.benders.generate_cut()
                if len(cuts_added) == 0:
                    break
            self.assertAlmostEqual(m.x.value, a, 4)
            self.assertAlmostEqual(pyo.value(m.obj), 0.0, 4)
            self.assertAlmostEqual(m.eta.value, 0.0, 4)

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_farmer_multiple_scenario_evaluate_single_scen_model(self, mip_solver):

        transform = "standard_lp"
        outer_farmer = tc.Farmer()
        expected_crop_answers = {
            "BelowAverageScenario": {"WHEAT": 100, "CORN": 25, "SUGAR_BEETS": 375},
            "AverageScenario": {"WHEAT": 120, "CORN": 80, "SUGAR_BEETS": 300},
            "AboveAverageScenario": {
                "WHEAT": 550.0 / 3.0,
                "CORN": 200.0 / 3.0,
                "SUGAR_BEETS": 250,
            },
        }
        expected_obj_answers = {
            "BelowAverageScenario": -59_950,
            "AverageScenario": -118_600,
            "AboveAverageScenario": (-167_667 + 1.0 / 3.0),
        }
        scenarios = expected_obj_answers.keys()
        for scen in scenarios:
            local_farmer = tc.Farmer()
            local_farmer.scenario_probabilities = {scen: 1.0}
            local_farmer.scenarios = [scen]
            t0 = time.time()
            opt, m = tc.Farmer.setup_farmer(
                local_farmer, solver_name=mip_solver, transform=transform
            )

            m.devoted_acreage["CORN"] = 0
            m.devoted_acreage["SUGAR_BEETS"] = 0
            m.devoted_acreage["WHEAT"] = 0
            for s in local_farmer.scenarios:
                m.eta[s] = 0

            self.assertAlmostEqual(pyo.value(m.obj), 0, 3)
            results_list = m.benders.evaluate_all_subproblems()
            assert len(results_list) == 1, "Expect only one result object"
            assert results_list[0].subproblem_needs_cut == True, "Should need a cut"
            assert (
                results_list[0].subproblem_eta == 98000.0
            ), "Did not get expected magic value, double check accuracy"

            cuts_added = m.benders.generate_cut()
            assert len(cuts_added) == 1, "Expect a cut to be added"

            for s in local_farmer.scenarios:
                for crop, val in expected_crop_answers[s].items():
                    m.devoted_acreage[crop] = val

            results_list = m.benders.evaluate_all_subproblems()
            assert len(results_list) == 1, "Expect only one result object"
            assert results_list[0].subproblem_needs_cut == True, "Should need a cut"
            for i, s in enumerate(local_farmer.scenarios):
                m.eta[s] = results_list[i].subproblem_eta
            self.assertAlmostEqual(pyo.value(m.obj), expected_obj_answers[scen], 3)

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_farmer_single_scenario_evaluate_single_scen_model(self, mip_solver):

        transform = "standard_lp"
        outer_farmer = tc.Farmer()
        expected_crop_answers = {
            "BelowAverageScenario": {"WHEAT": 100, "CORN": 25, "SUGAR_BEETS": 375},
            "AverageScenario": {"WHEAT": 120, "CORN": 80, "SUGAR_BEETS": 300},
            "AboveAverageScenario": {
                "WHEAT": 550.0 / 3.0,
                "CORN": 200.0 / 3.0,
                "SUGAR_BEETS": 250,
            },
        }
        expected_obj_answers = {
            "BelowAverageScenario": -59_950,
            "AverageScenario": -118_600,
            "AboveAverageScenario": (-167_667 + 1.0 / 3.0),
        }
        scenarios = expected_obj_answers.keys()
        for scen in scenarios:
            local_farmer = tc.Farmer()
            local_farmer.scenario_probabilities = {scen: 1.0}
            local_farmer.scenarios = [scen]
            t0 = time.time()
            opt, m = tc.Farmer.setup_farmer(
                local_farmer, solver_name=mip_solver, transform=transform
            )

            m.devoted_acreage["CORN"] = 0
            m.devoted_acreage["SUGAR_BEETS"] = 0
            m.devoted_acreage["WHEAT"] = 0
            for s in local_farmer.scenarios:
                m.eta[s] = 0

            self.assertAlmostEqual(pyo.value(m.obj), 0, 3)
            results_list = m.benders.evaluate_all_subproblems()
            assert len(results_list) == 1, "Expect only one result object"
            assert results_list[0].subproblem_needs_cut == True, "Should need a cut"
            assert (
                results_list[0].subproblem_eta == 98000.0
            ), "Did not get expected magic value, double check accuracy"

            cuts_added = m.benders.generate_cut()
            assert len(cuts_added) == 1, "Expect a cut to be added"

            for s in local_farmer.scenarios:
                for crop, val in expected_crop_answers[s].items():
                    m.devoted_acreage[crop] = val

            results_munch = m.benders.evaluate_single_subproblem(index=0)
            assert isinstance(results_munch, MyMunch), "Expect only one MyMunch object"
            assert results_munch.subproblem_needs_cut == True, "Should need a cut"
            for i, s in enumerate(local_farmer.scenarios):
                m.eta[s] = results_munch.subproblem_eta
            self.assertAlmostEqual(pyo.value(m.obj), expected_obj_answers[scen], 3)

    @parameterized.expand(input=persistent_mip_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_farmer_persistent_single_scenario_evaluate_single_scen_model(
        self, mip_solver
    ):

        transform = "standard_lp"
        outer_farmer = tc.Farmer()
        expected_crop_answers = {
            "BelowAverageScenario": {"WHEAT": 100, "CORN": 25, "SUGAR_BEETS": 375},
            "AverageScenario": {"WHEAT": 120, "CORN": 80, "SUGAR_BEETS": 300},
            "AboveAverageScenario": {
                "WHEAT": 550.0 / 3.0,
                "CORN": 200.0 / 3.0,
                "SUGAR_BEETS": 250,
            },
        }
        expected_obj_answers = {
            "BelowAverageScenario": -59_950,
            "AverageScenario": -118_600,
            "AboveAverageScenario": (-167_667 + 1.0 / 3.0),
        }
        scenarios = expected_obj_answers.keys()
        for scen in scenarios:
            local_farmer = tc.Farmer()
            local_farmer.scenario_probabilities = {scen: 1.0}
            local_farmer.scenarios = [scen]
            t0 = time.time()
            opt, m = tc.Farmer.setup_farmer_persistent(
                local_farmer, solver_name=mip_solver, transform=transform
            )

            m.devoted_acreage["CORN"] = 0
            m.devoted_acreage["SUGAR_BEETS"] = 0
            m.devoted_acreage["WHEAT"] = 0
            for s in local_farmer.scenarios:
                m.eta[s] = 0

            self.assertAlmostEqual(pyo.value(m.obj), 0, 3)
            results_list = m.benders.evaluate_all_subproblems()
            assert len(results_list) == 1, "Expect only one result object"
            assert results_list[0].subproblem_needs_cut == True, "Should need a cut"
            assert (
                results_list[0].subproblem_eta == 98000.0
            ), "Did not get expected magic value, double check accuracy"

            cuts_added = m.benders.generate_cut()
            assert len(cuts_added) == 1, "Expect a cut to be added"

            for s in local_farmer.scenarios:
                for crop, val in expected_crop_answers[s].items():
                    m.devoted_acreage[crop] = val

            results_munch = m.benders.evaluate_single_subproblem(index=0)
            assert isinstance(results_munch, MyMunch), "Expect only one MyMunch object"
            assert results_munch.subproblem_needs_cut == True, "Should need a cut"
            for i, s in enumerate(local_farmer.scenarios):
                m.eta[s] = results_munch.subproblem_eta
            self.assertAlmostEqual(pyo.value(m.obj), expected_obj_answers[scen], 3)

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_farmer_multiple_scenario_evaluate_multiple_scen_model(self, mip_solver):

        transform = "standard_lp"
        expected_crop_answers = {"WHEAT": 170, "CORN": 80, "SUGAR_BEETS": 250}
        expected_obj_answer = -108390
        local_farmer = tc.Farmer()
        t0 = time.time()
        opt, m = tc.Farmer.setup_farmer(
            local_farmer, solver_name=mip_solver, transform=transform
        )

        m.devoted_acreage["CORN"] = 0
        m.devoted_acreage["SUGAR_BEETS"] = 0
        m.devoted_acreage["WHEAT"] = 0
        for s in local_farmer.scenarios:
            m.eta[s] = 0

        for crop, val in expected_crop_answers.items():
            m.devoted_acreage[crop] = val

        results_list = m.benders.evaluate_all_subproblems()
        assert len(results_list) == 3, "Expect only one result per subproblem"
        assert results_list[0].subproblem_needs_cut == True, "Should need a cut"
        for i, s in enumerate(local_farmer.scenarios):
            m.eta[s] = results_list[i].subproblem_eta
        # N.B. some of the checks here have low precision because the published "answers" to the reference problems round
        self.assertAlmostEqual(pyo.value(m.obj), expected_obj_answer, 0)

    @parameterized.expand(input=persistent_mip_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_farmer_persistent_multiple_scenario_evaluate_multiple_scen_model(
        self, mip_solver
    ):

        transform = "standard_lp"
        expected_crop_answers = {"WHEAT": 170, "CORN": 80, "SUGAR_BEETS": 250}
        expected_obj_answer = -108390
        local_farmer = tc.Farmer()
        t0 = time.time()
        opt, m = tc.Farmer.setup_farmer_persistent(
            local_farmer, solver_name=mip_solver, transform=transform
        )

        m.devoted_acreage["CORN"] = 0
        m.devoted_acreage["SUGAR_BEETS"] = 0
        m.devoted_acreage["WHEAT"] = 0
        for s in local_farmer.scenarios:
            m.eta[s] = 0

        for crop, val in expected_crop_answers.items():
            m.devoted_acreage[crop] = val

        results_list = m.benders.evaluate_all_subproblems()
        assert len(results_list) == 3, "Expect only one result per subproblem"
        assert results_list[0].subproblem_needs_cut == True, "Should need a cut"
        for i, s in enumerate(local_farmer.scenarios):
            m.eta[s] = results_list[i].subproblem_eta
        # N.B. some of the checks here have low precision because the published "answers" to the reference problems round
        self.assertAlmostEqual(pyo.value(m.obj), expected_obj_answer, 0)

    @parameterized.expand(input=infeasibility_test_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_infeasible_problem_evaluate_single_problem(self, solver):
        transform = "standard_lp"
        x_set = [-10, 10]
        for x_val in x_set:
            m = tc.modified_absolute_value.create_root()
            root_vars = [m.x]
            data = data = MyMunch(a=0, L=1, R=1, LB=-6, UB=4)
            m.benders = BendersCutGenerator()
            m.benders.set_input(
                root_vars=root_vars,
                tol=1e-8,
                transform=transform,
                allow_infeasible=True,
            )
            m.benders.add_subproblem(
                subproblem_fn=tc.modified_absolute_value.create_subproblem,
                subproblem_fn_kwargs={"root_x": m.x, "data": data},
                root_eta=m.eta,
                subproblem_solver=solver,
            )
            m.x = x_val
            results_munch = m.benders.evaluate_single_subproblem(
                index=0, build_cut=False
            )
            assert isinstance(results_munch, MyMunch), "Expect only one MyMunch object"
            assert results_munch.subproblem_needs_cut == True, "Should need a cut"
            assert results_munch.subproblem_infeasible == True, "Should be infeasible"
            assert (
                results_munch.subproblem_constant is None
            ), "Should be None as problem is infeasible"
            assert (
                results_munch.subproblem_eta is None
            ), "Should be None as problem is infeasible"
            assert (
                results_munch.subproblem_coeff is None
            ), "Should be None as problem is infeasible"
            assert (
                results_munch.subproblem_eta_gap is None
            ), "Should be None as problem is infeasible"

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_infeasible_persistent_problem_evaluate_single_problem(self, solver):
        transform = "standard_lp"
        x_set = [-10, 10]
        for x_val in x_set:
            m = tc.modified_absolute_value.create_root()
            root_vars = [m.x]
            data = data = MyMunch(a=0, L=1, R=1, LB=-6, UB=4)
            m.benders = BendersCutGenerator()
            m.benders.set_input(
                root_vars=root_vars,
                tol=1e-8,
                transform=transform,
                allow_infeasible=True,
            )
            m.benders.add_subproblem(
                subproblem_fn=tc.modified_absolute_value.create_subproblem,
                subproblem_fn_kwargs={"root_x": m.x, "data": data},
                root_eta=m.eta,
                subproblem_solver=solver,
            )
            m.x = x_val
            results_munch = m.benders.evaluate_single_subproblem(
                index=0, build_cut=False
            )
            assert isinstance(results_munch, MyMunch), "Expect only one MyMunch object"
            assert results_munch.subproblem_needs_cut == True, "Should need a cut"
            assert results_munch.subproblem_infeasible == True, "Should be infeasible"
            assert (
                results_munch.subproblem_constant is None
            ), "Should be None as problem is infeasible"
            assert (
                results_munch.subproblem_eta is None
            ), "Should be None as problem is infeasible"
            assert (
                results_munch.subproblem_coeff is None
            ), "Should be None as problem is infeasible"
            assert (
                results_munch.subproblem_eta_gap is None
            ), "Should be None as problem is infeasible"

    @parameterized.expand(input=infeasibility_test_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_infeasible_problem_evaluate_all_subproblem(self, solver):
        transform = "standard_lp"
        x_set = [-10, 10]
        for x_val in x_set:
            m = tc.modified_absolute_value.create_root()
            root_vars = [m.x]
            data = data = MyMunch(a=0, L=1, R=1, LB=-6, UB=4)
            m.benders = BendersCutGenerator()
            m.benders.set_input(
                root_vars=root_vars,
                tol=1e-8,
                transform=transform,
                allow_infeasible=True,
            )
            m.benders.add_subproblem(
                subproblem_fn=tc.modified_absolute_value.create_subproblem,
                subproblem_fn_kwargs={"root_x": m.x, "data": data},
                root_eta=m.eta,
                subproblem_solver=solver,
            )
            m.x = x_val
            results_list = m.benders.evaluate_all_subproblems(build_cut=False)
            results_munch = results_list[0]
            assert isinstance(results_munch, MyMunch), "Expect only one MyMunch object"
            assert results_munch.subproblem_needs_cut == True, "Should need a cut"
            assert results_munch.subproblem_infeasible == True, "Should be infeasible"
            assert (
                results_munch.subproblem_constant is None
            ), "Should be None as problem is infeasible"
            assert (
                results_munch.subproblem_eta is None
            ), "Should be None as problem is infeasible"
            assert (
                results_munch.subproblem_coeff is None
            ), "Should be None as problem is infeasible"
            assert (
                results_munch.subproblem_eta_gap is None
            ), "Should be None as problem is infeasible"

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_infeasible_persistent_problem_evaluate_all_subproblem_skip_cut_build(
        self, solver
    ):
        transform = "standard_lp"
        x_set = [-10, 10]
        for x_val in x_set:
            m = tc.modified_absolute_value.create_root()
            root_vars = [m.x]
            data = data = MyMunch(a=0, L=1, R=1, LB=-6, UB=4)
            m.benders = BendersCutGenerator()
            m.benders.set_input(
                root_vars=root_vars,
                tol=1e-8,
                transform=transform,
                allow_infeasible=True,
            )
            m.benders.add_subproblem(
                subproblem_fn=tc.modified_absolute_value.create_subproblem,
                subproblem_fn_kwargs={"root_x": m.x, "data": data},
                root_eta=m.eta,
                subproblem_solver=solver,
            )
            m.x = x_val
            results_list = m.benders.evaluate_all_subproblems(build_cut=False)
            results_munch = results_list[0]
            assert isinstance(results_munch, MyMunch), "Expect only one MyMunch object"
            assert results_munch.subproblem_needs_cut == True, "Should need a cut"
            assert results_munch.subproblem_infeasible == True, "Should be infeasible"
            assert (
                results_munch.subproblem_constant is None
            ), "Should be None as problem is infeasible"
            assert (
                results_munch.subproblem_eta is None
            ), "Should be None as problem is infeasible"
            assert (
                results_munch.subproblem_coeff is None
            ), "Should be None as problem is infeasible"
            assert (
                results_munch.subproblem_eta_gap is None
            ), "Should be None as problem is infeasible"

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_infeasible_persistent_problem_evaluate_all_subproblem_cut_build(
        self, solver
    ):
        transform = "standard_lp"
        x_set = [-10, 10]
        constants = [-6, -4]
        coeffs = [[1], [-1]]
        for index, x_val in enumerate(x_set):
            m = tc.modified_absolute_value.create_root()
            root_vars = [m.x]
            data = data = MyMunch(a=0, L=1, R=1, LB=-6, UB=4)
            m.benders = BendersCutGenerator()
            m.benders.set_input(
                root_vars=root_vars,
                tol=1e-8,
                transform=transform,
                allow_infeasible=True,
            )
            m.benders.add_subproblem(
                subproblem_fn=tc.modified_absolute_value.create_subproblem,
                subproblem_fn_kwargs={"root_x": m.x, "data": data},
                root_eta=m.eta,
                subproblem_solver=solver,
            )
            m.x = x_val
            results_list = m.benders.evaluate_all_subproblems(build_cut=True)
            results_munch = results_list[0]
            assert isinstance(results_munch, MyMunch), "Expect only one MyMunch object"
            assert results_munch.subproblem_needs_cut == True, "Should need a cut"
            assert results_munch.subproblem_infeasible == True, "Should be infeasible"
            assert (
                results_munch.subproblem_eta is None
            ), "Should be None as problem is infeasible"
            assert (
                results_munch.subproblem_eta_gap is None
            ), "Should be None as problem is infeasible"
            self.assertAlmostEqual(
                results_munch.subproblem_constant, constants[index], 7
            )
            numpy.testing.assert_allclose(
                results_munch.subproblem_coeff,
                coeffs[index],
                rtol=1e-7,
                atol=1e-8,
                equal_nan=True,
            )

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_infeasible_persistent_problem_evaluate_single_problem_subproblem_cut_build(
        self, solver
    ):
        transform = "standard_lp"
        x_set = [-10, 10]
        constants = [-6, -4]
        coeffs = [[1], [-1]]
        for index, x_val in enumerate(x_set):
            m = tc.modified_absolute_value.create_root()
            root_vars = [m.x]
            data = data = MyMunch(a=0, L=1, R=1, LB=-6, UB=4)
            m.benders = BendersCutGenerator()
            m.benders.set_input(
                root_vars=root_vars,
                tol=1e-8,
                transform=transform,
                allow_infeasible=True,
            )
            m.benders.add_subproblem(
                subproblem_fn=tc.modified_absolute_value.create_subproblem,
                subproblem_fn_kwargs={"root_x": m.x, "data": data},
                root_eta=m.eta,
                subproblem_solver=solver,
            )
            m.x = x_val
            results_munch = m.benders.evaluate_single_subproblem(
                index=0, build_cut=True
            )
            assert isinstance(results_munch, MyMunch), "Expect only one MyMunch object"
            assert results_munch.subproblem_needs_cut == True, "Should need a cut"
            assert results_munch.subproblem_infeasible == True, "Should be infeasible"
            assert (
                results_munch.subproblem_eta is None
            ), "Should be None as problem is infeasible"
            assert (
                results_munch.subproblem_eta_gap is None
            ), "Should be None as problem is infeasible"
            self.assertAlmostEqual(
                results_munch.subproblem_constant, constants[index], 7
            )
            numpy.testing.assert_allclose(
                results_munch.subproblem_coeff,
                coeffs[index],
                rtol=1e-7,
                atol=1e-8,
                equal_nan=True,
            )

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_infeasible_persistent_generate_cut(self, solver):
        transform = "standard_lp"
        x_set = [-10, 10]
        constants = [-6, -4]
        # note the flipped sign here from the evaluate above
        # this is a detail of how the cuts are formed
        coeffs = [[-1], [1]]
        for index, x_val in enumerate(x_set):
            m = tc.modified_absolute_value.create_root()
            root_vars = [m.x]
            data = data = MyMunch(a=0, L=1, R=1, LB=-6, UB=4)
            m.benders = BendersCutGenerator()
            m.benders.set_input(
                root_vars=root_vars,
                tol=1e-8,
                transform=transform,
                allow_infeasible=True,
            )
            m.benders.add_subproblem(
                subproblem_fn=tc.modified_absolute_value.create_subproblem,
                subproblem_fn_kwargs={"root_x": m.x, "data": data},
                root_eta=m.eta,
                subproblem_solver=solver,
            )
            m.x = x_val

            cuts_list = m.benders.generate_all_subproblem_cut()
            assert len(cuts_list) == 1
            repn = generate_standard_repn(cuts_list[0].body, compute_values=False)
            self.assertAlmostEqual(repn.constant, constants[index], 7)
            numpy.testing.assert_allclose(
                repn.linear_coefs,
                coeffs[index],
                rtol=1e-7,
                atol=1e-8,
                equal_nan=True,
            )

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_dcopf_simple_evaluate_1(self, solver):
        transform = "standard_lp"
        set_points = [0, 50, 100]
        expected_obj = 5000
        # for index, gen_start in enumerate(set_points):
        grid = tc.EnergyGrid()
        m = tc.EnergyGrid.create_root(grid=grid)

        # TODO: this is finicky, need to resolve down to a list, not a list with a single wrapper object inside it
        # root_vars =  list(m.generation.values()) or [m.generation[s] for s in m.generation.index_set()] achieve this
        # root_vars = [m.generation.values()] does not
        root_vars = list(m.generation.values())
        print(root_vars)
        # root_vars = list(m.generation.values())
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
            subproblem_fn_kwargs={"root": m, "grid": grid},
            root_eta=m.eta,
            subproblem_solver=solver,
        )
        for b in grid.buses:
            m.generation[b] = 0

        results_munch = m.benders.evaluate_single_subproblem(index=0, build_cut=True)
        assert isinstance(results_munch, MyMunch), "Expect only one MyMunch object"
        assert results_munch.subproblem_needs_cut == True, "Should need a cut"
        assert results_munch.subproblem_infeasible == True, "Should be infeasible"
        assert (
            results_munch.subproblem_eta is None
        ), "Should be None as problem is infeasible"
        assert (
            results_munch.subproblem_eta_gap is None
        ), "Should be None as problem is infeasible"

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_dcopf_simple_evaluate_2(self, solver):
        transform = "standard_lp"
        set_points = [0, 50, 100]
        expected_obj = 5000
        # for index, gen_start in enumerate(set_points):
        grid = tc.EnergyGrid()
        m = tc.EnergyGrid.create_root(grid=grid)

        # TODO: this is finicky, need to resolve down to a list, not a list with a single wrapper object inside it
        # root_vars =  list(m.generation.values()) or [m.generation[s] for s in m.generation.index_set()] achieve this
        # root_vars = [m.generation.values()] does not
        root_vars = list(m.generation.values())
        print(root_vars)
        # root_vars = list(m.generation.values())
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
            subproblem_fn_kwargs={"root": m, "grid": grid},
            root_eta=m.eta,
            subproblem_solver=solver,
        )
        # m.eta.fix(0)
        print(f"{m.benders.feasibility_only=}")
        for b in grid.buses:
            m.generation[b] = 0
        m.generation["bus1"] = 50.0
        m.generation["bus2"] = 50.0

        results_munch = m.benders.evaluate_single_subproblem(index=0, build_cut=True)
        assert isinstance(results_munch, MyMunch), "Expect only one MyMunch object"
        assert results_munch.subproblem_infeasible == False, "Should be feasible"
        assert results_munch.subproblem_needs_cut == False, "Should not need a cut"
        assert (
            results_munch.subproblem_eta is not None
        ), "Should be not None as problem is feasible"

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_dcopf_simple_evaluate_3(self, solver):
        transform = "standard_lp"
        grid = tc.EnergyGrid()
        m = tc.EnergyGrid.create_root(grid=grid)

        root_vars = list(m.generation.values())
        print(root_vars)
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
            subproblem_fn_kwargs={"root": m, "grid": grid},
            root_eta=m.eta,
            subproblem_solver=solver,
        )
        print(f"{m.benders.feasibility_only=}")
        for b in grid.buses:
            m.generation[b] = 0
        m.generation["bus1"] = 100
        m.generation["bus2"] = 0

        results_munch = m.benders.evaluate_single_subproblem(index=0, build_cut=True)
        assert isinstance(results_munch, MyMunch), "Expect only one MyMunch object"
        assert results_munch.subproblem_infeasible == False
        assert results_munch.subproblem_needs_cut == False

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_dcopf_simple_evaluate_4(self, solver):
        transform = "standard_lp"
        # for index, gen_start in enumerate(set_points):
        grid = tc.EnergyGrid()
        m = tc.EnergyGrid.create_root(grid=grid)

        root_vars = list(m.generation.values())
        print(root_vars)
        # root_vars = list(m.generation.values())
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
            subproblem_fn_kwargs={"root": m, "grid": grid, "feasibility_only": True},
            root_eta=m.eta,
            subproblem_solver=solver,
        )
        # m.eta.fix(0)
        print(f"{m.benders.feasibility_only=}")
        for b in grid.buses:
            m.generation[b] = 0
        m.generation["bus1"] = 100
        m.generation["bus2"] = 0

        results_munch = m.benders.evaluate_single_subproblem(index=0, build_cut=True)
        assert isinstance(results_munch, MyMunch), "Expect only one MyMunch object"
        assert results_munch.subproblem_infeasible == False
        assert results_munch.subproblem_needs_cut == False

    #
    # Newsvendor tests
    #
    #
    # Newsvendor tests
    #

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_newsvendor_simple_evaluate_1(self, solver):
        transform = "standard_lp"
        # for index, gen_start in enumerate(set_points):
        m = tc.newsvendor.create_root()

        root_vars = [m.x]
        print(root_vars)
        # root_vars = list(m.generation.values())
        m.benders = BendersCutGenerator()
        m.benders.set_input(
            root_vars=root_vars,
            tol=1e-8,
            transform=transform,
            allow_infeasible=True,
            feasibility_only=True,
        )
        m.benders.add_subproblem(
            subproblem_fn=tc.newsvendor.create_subproblem,
            subproblem_fn_kwargs={"root_x": m.x},
            root_eta=m.eta,
            subproblem_solver=solver,
        )
        m.x = 0
        expected_obj_answer = 75

        results_munch = m.benders.evaluate_single_subproblem(index=0)
        assert isinstance(results_munch, MyMunch), "Expect only one MyMunch object"
        assert results_munch.subproblem_infeasible == False
        # assert results_munch.subproblem_needs_cut == True
        self.assertAlmostEqual(results_munch.subproblem_eta, expected_obj_answer, 3)

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_newsvendor_simple_evaluate_2(self, solver):
        transform = "standard_lp"
        # for index, gen_start in enumerate(set_points):
        m = tc.newsvendor.create_root()

        root_vars = [m.x]
        print(root_vars)
        # root_vars = list(m.generation.values())
        m.benders = BendersCutGenerator()
        m.benders.set_input(
            root_vars=root_vars,
            tol=1e-8,
            transform=transform,
            allow_infeasible=True,
            feasibility_only=True,
        )
        m.benders.add_subproblem(
            subproblem_fn=tc.newsvendor.create_subproblem,
            subproblem_fn_kwargs={"root_x": m.x},
            root_eta=m.eta,
            subproblem_solver=solver,
        )
        m.x = 25  # 0
        expected_obj_answer = 62.5  # 75

        results_munch = m.benders.evaluate_single_subproblem(index=0)
        assert isinstance(results_munch, MyMunch), "Expect only one MyMunch object"
        assert results_munch.subproblem_infeasible == False
        # assert results_munch.subproblem_needs_cut == True
        self.assertAlmostEqual(results_munch.subproblem_eta, expected_obj_answer, 3)

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_newsvendor_simple_evaluate_2(self, solver):
        transform = "standard_lp"
        # for index, gen_start in enumerate(set_points):
        m = tc.newsvendor.create_root()

        root_vars = [m.x]
        print(root_vars)
        # root_vars = list(m.generation.values())
        m.benders = BendersCutGenerator()
        m.benders.set_input(
            root_vars=root_vars,
            tol=1e-8,
            transform=transform,
            allow_infeasible=True,
            feasibility_only=True,
        )
        m.benders.add_subproblem(
            subproblem_fn=tc.newsvendor.create_subproblem,
            subproblem_fn_kwargs={"root_x": m.x},
            root_eta=m.eta,
            subproblem_solver=solver,
        )
        m.x = 25  # 0
        expected_obj_answer = 62.5  # 75

        results_munch = m.benders.evaluate_single_subproblem(index=0)
        assert isinstance(results_munch, MyMunch), "Expect only one MyMunch object"
        assert results_munch.subproblem_infeasible == False
        # assert results_munch.subproblem_needs_cut == True
        self.assertAlmostEqual(results_munch.subproblem_eta, expected_obj_answer, 3)

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_newsvendor_simple_evaluate_3(self, solver):
        transform = "standard_lp"
        # for index, gen_start in enumerate(set_points):
        m = tc.newsvendor.create_root()

        root_vars = [m.x]
        print(root_vars)
        # root_vars = list(m.generation.values())
        m.benders = BendersCutGenerator()
        m.benders.set_input(
            root_vars=root_vars,
            tol=1e-8,
            transform=transform,
            allow_infeasible=True,
            feasibility_only=True,
        )
        m.benders.add_subproblem(
            subproblem_fn=tc.newsvendor.create_subproblem,
            subproblem_fn_kwargs={"root_x": m.x},
            root_eta=m.eta,
            subproblem_solver=solver,
        )
        x_vals = [0, 25, 50, 75, 100]
        obj_vals = [75, 62.5, 50, 77.5, 105]
        for i, v in enumerate(x_vals):
            m.x = v
            expected_obj_answer = obj_vals[i]

            results_munch = m.benders.evaluate_single_subproblem(index=0)
            assert isinstance(results_munch, MyMunch), "Expect only one MyMunch object"
            assert results_munch.subproblem_infeasible == False
            # assert results_munch.subproblem_needs_cut == True
            self.assertAlmostEqual(results_munch.subproblem_eta, expected_obj_answer, 3)


def _try_import_parallel_generator():
    try:
        from or_topas.benders.benders_parallel import (
            BendersGenerator_Parallel,
        )
    except Exception:
        return None
    return BendersGenerator_Parallel


class TestBendersLastEvalResults(unittest.TestCase):
    """
    Tests for last_eval_results / last_cuts_added and the Serial helpers.

    last_eval_results[i] is aligned with all_root_etas[i] (add_subproblem order).
    On standard_lp, last_subproblem_etas()[i] is Q_s(x), or None if that
    subproblem was infeasible.
    """

    def _build_abs(self, solver, transform="standard_lp", allow_infeasible=False):
        m = tc.absolute_value.create_root()
        m.benders = BendersCutGenerator()
        m.benders.set_input(
            root_vars=[m.x],
            tol=1e-8,
            transform=transform,
            allow_infeasible=allow_infeasible,
        )
        m.benders.add_subproblem(
            subproblem_fn=tc.absolute_value.create_subproblem,
            subproblem_fn_kwargs={"root": m},
            root_eta=m.eta,
            subproblem_solver=solver,
        )
        return m

    def _build_modified_abs(
        self, solver, data, allow_infeasible=False, transform="standard_lp"
    ):
        m = tc.modified_absolute_value.create_root()
        m.benders = BendersCutGenerator()
        m.benders.set_input(
            root_vars=[m.x],
            tol=1e-8,
            transform=transform,
            allow_infeasible=allow_infeasible,
        )
        m.benders.add_subproblem(
            subproblem_fn=tc.modified_absolute_value.create_subproblem,
            subproblem_fn_kwargs={"root_x": m.x, "data": data},
            root_eta=m.eta,
            subproblem_solver=solver,
        )
        return m

    def _build_two_modified_abs(self, solver, data0, data1, allow_infeasible=False):
        # Hand-rolled: do not use eta_count helper (it reuses one data object).
        m = tc.modified_absolute_value.create_root(eta_count=2)
        m.benders = BendersCutGenerator()
        m.benders.set_input(
            root_vars=[m.x],
            tol=1e-8,
            transform="standard_lp",
            allow_infeasible=allow_infeasible,
        )
        m.benders.add_subproblem(
            subproblem_fn=tc.modified_absolute_value.create_subproblem,
            subproblem_fn_kwargs={"root_x": m.x, "data": data0},
            root_eta=m.eta[0],
            subproblem_solver=solver,
        )
        m.benders.add_subproblem(
            subproblem_fn=tc.modified_absolute_value.create_subproblem,
            subproblem_fn_kwargs={"root_x": m.x, "data": data1},
            root_eta=m.eta[1],
            subproblem_solver=solver,
        )
        return m

    def _assert_pre_eval_helpers(self, benders):
        self.assertIsNone(benders.last_eval_results)
        self.assertIsNone(benders.last_iterate_had_infeasible_subproblem())
        self.assertFalse(benders.last_iterate_is_feasible())
        self.assertIsNone(benders.last_subproblem_etas())

    #
    # Interface
    #

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_helpers_before_any_eval_serial(self, solver):
        m = self._build_abs(solver)
        self._assert_pre_eval_helpers(m.benders)

    def test_parallel_helpers_raise(self):
        Parallel = _try_import_parallel_generator()
        if Parallel is None:
            self.skipTest("BendersGenerator_Parallel could not be imported")
        try:
            m = pyo.ConcreteModel()
            m.benders = Parallel()
        except ImportError as e:
            self.skipTest(str(e))

        expected = "use Benders_Serial"
        for method_name in (
            "last_iterate_had_infeasible_subproblem",
            "last_iterate_is_feasible",
            "last_subproblem_etas",
        ):
            with self.assertRaises(NotImplementedError) as cm:
                getattr(m.benders, method_name)()
            self.assertIn(expected, str(cm.exception))

    #
    # Single absolute_value  (Q(x) = |x|)
    #

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_generate_cut_abs_feasible_populates(self, solver):
        m = self._build_abs(solver)
        m.x = 2
        m.eta = 0
        cuts_added = m.benders.generate_cut()

        self.assertTrue(m.benders.last_iterate_is_feasible())
        self.assertFalse(m.benders.last_iterate_had_infeasible_subproblem())
        self.assertEqual(len(m.benders.last_eval_results), 1)
        etas = m.benders.last_subproblem_etas()
        self.assertEqual(len(etas), 1)
        self.assertAlmostEqual(etas[0], 2.0, 6)
        self.assertEqual(len(m.benders.last_cuts_added), len(cuts_added))
        self.assertEqual(list(m.benders.last_cuts_added), list(cuts_added))
        self.assertGreaterEqual(len(cuts_added), 1)

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_evaluate_all_abs_feasible_populates(self, solver):
        m = self._build_abs(solver)
        m.x = 2
        m.eta = 0
        results = m.benders.evaluate_all_subproblems()

        self.assertEqual(len(results), 1)
        self.assertIs(m.benders.last_eval_results, results)
        self.assertTrue(m.benders.last_iterate_is_feasible())
        self.assertAlmostEqual(m.benders.last_subproblem_etas()[0], 2.0, 6)
        self.assertFalse(results[0].subproblem_infeasible)

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_generate_cut_overwrites_evaluate_all(self, solver):
        m = self._build_abs(solver)
        m.x = 2
        m.eta = 0
        m.benders.evaluate_all_subproblems()
        self.assertAlmostEqual(m.benders.last_subproblem_etas()[0], 2.0, 6)

        m.x = 5
        m.benders.generate_cut()
        self.assertTrue(m.benders.last_iterate_is_feasible())
        self.assertAlmostEqual(m.benders.last_subproblem_etas()[0], 5.0, 6)

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_set_input_resets_stash(self, solver):
        m = self._build_abs(solver)
        m.x = 2
        m.eta = 0
        m.benders.generate_cut()
        self.assertIsNotNone(m.benders.last_eval_results)

        m.benders.set_input(
            root_vars=[m.x],
            tol=1e-8,
            transform="standard_lp",
        )
        self.assertEqual(m.benders.last_cuts_added, [])
        self._assert_pre_eval_helpers(m.benders)

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_feasibility_transform_abs_treated_as_feasible(self, solver):
        m = self._build_abs(solver, transform="feasibility")
        m.x = 2
        m.eta = 0
        m.benders.generate_cut()

        self.assertTrue(m.benders.last_iterate_is_feasible())
        self.assertFalse(m.benders.last_iterate_had_infeasible_subproblem())
        etas = m.benders.last_subproblem_etas()
        self.assertEqual(len(etas), 1)
        self.assertIsNotNone(etas[0])
        # feasibility-transform eta is a dual coefficient, not Q(x)=|x|
        self.assertEqual(len(m.benders.last_eval_results), 1)
        self.assertFalse(
            hasattr(m.benders.last_eval_results[0], "subproblem_infeasible")
            and m.benders.last_eval_results[0].subproblem_infeasible
        )

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_abs_solver_loop_terminates_feasible(self, solver):
        m = self._build_abs(solver)
        opt = pyo.SolverFactory(solver)
        for _ in range(30):
            opt.solve(m, tee=False)
            cuts_added = m.benders.generate_cut()
            if len(cuts_added) == 0:
                break
        self.assertAlmostEqual(m.x.value, 0.0, 4)
        self.assertTrue(m.benders.last_iterate_is_feasible())
        self.assertAlmostEqual(m.benders.last_subproblem_etas()[0], 0.0, 4)

    #
    # Single modified_absolute_value  (infeasible + feasible)
    #

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_modified_abs_infeasible_generate_cut(self, solver):
        data = MyMunch(a=0, L=1, R=1, LB=-6, UB=4)
        for x_val in (-10, 10):
            m = self._build_modified_abs(solver, data, allow_infeasible=True)
            m.x = x_val
            m.eta = 0
            cuts_added = m.benders.generate_cut()

            self.assertTrue(m.benders.last_iterate_had_infeasible_subproblem())
            self.assertFalse(m.benders.last_iterate_is_feasible())
            etas = m.benders.last_subproblem_etas()
            self.assertEqual(len(etas), 1)
            self.assertIsNone(etas[0])
            self.assertGreaterEqual(len(cuts_added), 1)
            self.assertEqual(len(m.benders.last_cuts_added), len(cuts_added))

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_modified_abs_infeasible_evaluate_all(self, solver):
        data = MyMunch(a=0, L=1, R=1, LB=-6, UB=4)
        for x_val in (-10, 10):
            for build_cut in (False, True):
                m = self._build_modified_abs(solver, data, allow_infeasible=True)
                m.x = x_val
                m.eta = 0
                results = m.benders.evaluate_all_subproblems(build_cut=build_cut)
                self.assertEqual(len(results), 1)
                self.assertTrue(results[0].subproblem_infeasible)
                self.assertTrue(m.benders.last_iterate_had_infeasible_subproblem())
                self.assertFalse(m.benders.last_iterate_is_feasible())
                self.assertIsNone(m.benders.last_subproblem_etas()[0])

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_modified_abs_feasible_inside_bounds(self, solver):
        data = MyMunch(a=0, L=1, R=1, LB=-6, UB=4)
        m = self._build_modified_abs(solver, data)
        m.x = 1
        m.eta = 0
        m.benders.evaluate_all_subproblems()
        self.assertTrue(m.benders.last_iterate_is_feasible())
        self.assertAlmostEqual(m.benders.last_subproblem_etas()[0], 1.0, 6)

    #
    # Two modified-abs with different data
    #

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_two_modified_abs_alignment(self, solver):
        data0 = MyMunch(a=0, L=1, R=1, LB=None, UB=None)
        data1 = MyMunch(a=3, L=1, R=1, LB=None, UB=None)
        m = self._build_two_modified_abs(solver, data0, data1)
        m.x = 1
        m.eta[0] = 0
        m.eta[1] = 0
        m.benders.evaluate_all_subproblems()

        etas = m.benders.last_subproblem_etas()
        self.assertEqual(len(etas), 2)
        self.assertEqual(len(m.benders.last_eval_results), 2)
        self.assertEqual(len(m.benders.all_root_etas), 2)
        # add order: Q_0(1)=|1-0|=1, Q_1(1)=|1-3|=2
        self.assertAlmostEqual(etas[0], 1.0, 6)
        self.assertAlmostEqual(etas[1], 2.0, 6)
        self.assertIs(m.benders.all_root_etas[0], m.eta[0])
        self.assertIs(m.benders.all_root_etas[1], m.eta[1])
        self.assertTrue(m.benders.last_iterate_is_feasible())

        # swapped values would fail the reconstruction
        m.eta[0] = etas[0]
        m.eta[1] = etas[1]
        self.assertAlmostEqual(pyo.value(m.obj), 3.0, 6)

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_evaluate_single_does_not_clobber(self, solver):
        data0 = MyMunch(a=0, L=1, R=1, LB=None, UB=None)
        data1 = MyMunch(a=3, L=1, R=1, LB=None, UB=None)
        m = self._build_two_modified_abs(solver, data0, data1)
        m.x = 1
        m.eta[0] = 0
        m.eta[1] = 0
        m.benders.evaluate_all_subproblems()
        snapshot = list(m.benders.last_subproblem_etas())
        self.assertEqual(len(snapshot), 2)

        single = m.benders.evaluate_single_subproblem(index=0)
        self.assertAlmostEqual(single.subproblem_eta, 1.0, 6)
        # full-vector stash must be unchanged
        self.assertEqual(len(m.benders.last_eval_results), 2)
        etas = m.benders.last_subproblem_etas()
        self.assertAlmostEqual(etas[0], snapshot[0], 6)
        self.assertAlmostEqual(etas[1], snapshot[1], 6)

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_two_modified_abs_mixed_feasibility(self, solver):
        data0 = MyMunch(a=0, L=1, R=1, LB=-6, UB=4)
        data1 = MyMunch(a=3, L=1, R=1, LB=None, UB=None)
        m = self._build_two_modified_abs(solver, data0, data1, allow_infeasible=True)
        m.x = -10
        m.eta[0] = 0
        m.eta[1] = 0
        m.benders.generate_cut()

        self.assertTrue(m.benders.last_iterate_had_infeasible_subproblem())
        self.assertFalse(m.benders.last_iterate_is_feasible())
        etas = m.benders.last_subproblem_etas()
        self.assertEqual(len(etas), 2)
        self.assertIsNone(etas[0])
        self.assertAlmostEqual(etas[1], 13.0, 6)  # |-10-3|

    #
    # Farmer  (three distinct Q_s, add-order = farmer.scenarios)
    #

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_farmer_zero_acres_evaluate_all_alignment(self, mip_solver):
        local_farmer = tc.Farmer()
        opt, m = tc.Farmer.setup_farmer(
            local_farmer, solver_name=mip_solver, transform="standard_lp"
        )
        for crop in local_farmer.crops:
            m.devoted_acreage[crop] = 0
        for s in local_farmer.scenarios:
            m.eta[s] = 0

        results = m.benders.evaluate_all_subproblems()
        self.assertEqual(len(results), 3)
        self.assertEqual(len(m.benders.last_eval_results), 3)
        self.assertTrue(m.benders.last_iterate_is_feasible())

        etas = m.benders.last_subproblem_etas()
        self.assertEqual(len(etas), 3)
        self.assertTrue(all(e is not None for e in etas))
        # add_subproblem order is farmer.scenarios
        # (do not assert the three Q_s are distinct — at zero acres they
        #  are p_s * 98000 and can match if probabilities match)
        for i, s in enumerate(local_farmer.scenarios):
            self.assertIs(m.benders.all_root_etas[i], m.eta[s])
            m.eta[s] = etas[i]
        # zero acres: first-stage cost 0; sum_s p_s * 98000 = 98000
        self.assertAlmostEqual(pyo.value(m.obj), 98000.0, 0)

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_farmer_known_acreage_alignment(self, mip_solver):
        local_farmer = tc.Farmer()
        opt, m = tc.Farmer.setup_farmer(
            local_farmer, solver_name=mip_solver, transform="standard_lp"
        )
        expected_crop = {"WHEAT": 170, "CORN": 80, "SUGAR_BEETS": 250}
        for crop, val in expected_crop.items():
            m.devoted_acreage[crop] = val
        for s in local_farmer.scenarios:
            m.eta[s] = 0

        m.benders.evaluate_all_subproblems()
        self.assertTrue(m.benders.last_iterate_is_feasible())
        etas = m.benders.last_subproblem_etas()
        self.assertEqual(len(etas), 3)
        for i, s in enumerate(local_farmer.scenarios):
            m.eta[s] = etas[i]
        # published farmer objective at this acreage
        self.assertAlmostEqual(pyo.value(m.obj), -108390, 0)

        m.benders.generate_cut()
        self.assertTrue(m.benders.last_iterate_is_feasible())
        etas_after_cut = m.benders.last_subproblem_etas()
        self.assertEqual(len(etas_after_cut), 3)
        for i, s in enumerate(local_farmer.scenarios):
            m.eta[s] = etas_after_cut[i]
        self.assertAlmostEqual(pyo.value(m.obj), -108390, 0)
