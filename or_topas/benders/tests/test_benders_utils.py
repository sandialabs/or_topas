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
