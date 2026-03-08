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

parameterized, param_available = attempt_import("parameterized")
if not param_available:
    raise unittest.SkipTest("Parameterized is not available.")
parameterized = parameterized.parameterized

non_persistnet_mip_solvers = list(
    pyomo.opt.check_available_solvers("glpk", "highs", "gurobi_direct")
)

qp_solvers = list(pyomo.opt.check_available_solvers("ipopt", "gurobi_direct", "highs"))
non_linear_solvers = list(pyomo.opt.check_available_solvers("ipopt"))


ipopt_available = pyo.SolverFactory("ipopt").available(exception_flag=False)
gurobi_available = pyo.SolverFactory("gurobi_persistent").available(
    exception_flag=False
)


class TestBendersUtils(unittest.TestCase):
    @parameterized.expand(input=non_persistnet_mip_solvers, skip_on_empty=True)
    def test_farmer(self, mip_solver):

        t = True
        assert t, "Trivial Test"

    @parameterized.expand(input=non_persistnet_mip_solvers, skip_on_empty=True)
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

    @parameterized.expand(input=non_persistnet_mip_solvers, skip_on_empty=True)
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
            assert results_list[0].subproblem_needs_cut == True, "Should need a cut"
            for i, s in enumerate(local_farmer.scenarios):
                m.eta[s] = results_munch.subproblem_eta
            self.assertAlmostEqual(pyo.value(m.obj), expected_obj_answers[scen], 3)

    @parameterized.expand(input=non_persistnet_mip_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_farmer_multiple_scenario_evaluate_multiple_scen_model(self, mip_solver):

        transform = "standard_lp"
        expected_crop_answers = {"WHEAT": 170, "CORN": 80, "SUGAR_BEETS": 250}
        expected_obj_answer = -108390
        # self.assertAlmostEqual(m.devoted_acreage["CORN"].value, 80, 7)
        # self.assertAlmostEqual(m.devoted_acreage["SUGAR_BEETS"].value, 250, 7)
        # self.assertAlmostEqual(m.devoted_acreage["WHEAT"].value, 170, 7)
        # self.assertAlmostEqual(pyo.value(m.obj), -108390, 0)
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
