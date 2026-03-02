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

# from or_topas.benders.benders_serial import (
#     BendersGenerator_Serial as BendersCutGenerator,
# )

from or_topas.benders import (
    BendersGenerator_Serial as BendersCutGenerator,
)

import or_topas.benders.tests.test_cases as tc

parameterized, param_available = attempt_import("parameterized")
if not param_available:
    raise unittest.SkipTest("Parameterized is not available.")
parameterized = parameterized.parameterized

non_persistent_mip_solvers = list(
    pyomo.opt.check_available_solvers("glpk", "highs", "gurobi_direct")
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


class TestBendersSolver(unittest.TestCase):

    #
    # Farmer Tests
    #

    @unittest.skipIf(not numpy_available, "numpy is not available.")
    @parameterized.expand(
        input=iter_product(non_persistent_mip_solvers, transforms),
        name_func=lambda func, num, params: f"{func.__name__}_solver_{params.args[0]}_transform_{params.args[1]}",
        skip_on_empty=True,
    )
    def test_farmer_single_scenarios(self, mip_solver, transform):

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
            "AboveAverageScenario": -167_667,
        }
        for scen, prob in outer_farmer.scenario_probabilities.items():
            local_farmer = tc.Farmer()
            local_farmer.scenario_probabilities = {scen: 1.0}
            local_farmer.scenarios = [scen]
            t0 = time.time()
            opt, m = tc.Farmer.setup_farmer(
                local_farmer, solver_name=mip_solver, transform=transform
            )

            print("\n")
            print(
                "{0:<15}{1:<15}{2:<15}{3:<15}{4:<15}".format(
                    "# Cuts", "Corn", "Sugar Beets", "Wheat", "Total_Time"
                )
            )
            for i in range(30):
                res = opt.solve(m, tee=False)
                cuts_added = m.benders.generate_cut()
                # for c in cuts_added:
                #     opt.add_constraint(c)
                print(
                    "{0:<15}{1:<15.2f}{2:<15.2f}{3:<15.2f}{4:<15.2f}".format(
                        len(cuts_added),
                        m.devoted_acreage["CORN"].value,
                        m.devoted_acreage["SUGAR_BEETS"].value,
                        m.devoted_acreage["WHEAT"].value,
                        time.time() - t0,
                    )
                )
                if len(cuts_added) == 0:
                    break

            expected_result = expected_crop_answers[scen]
            self.assertAlmostEqual(
                m.devoted_acreage["CORN"].value, expected_result["CORN"], 7
            )
            self.assertAlmostEqual(
                m.devoted_acreage["SUGAR_BEETS"].value,
                expected_result["SUGAR_BEETS"],
                7,
            )
            self.assertAlmostEqual(
                m.devoted_acreage["WHEAT"].value, expected_result["WHEAT"], 7
            )
            self.assertAlmostEqual(pyo.value(m.obj), expected_obj_answers[scen], 0)

    @unittest.skipIf(not numpy_available, "numpy is not available.")
    @parameterized.expand(
        input=iter_product(non_persistent_mip_solvers, transforms),
        name_func=lambda func, num, params: f"{func.__name__}_solver_{params.args[0]}_transform_{params.args[1]}",
        skip_on_empty=True,
    )
    def test_farmer_single_scenario_average_yield(self, mip_solver, transform):

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
            "AboveAverageScenario": -167_667,
        }
        scenarios = {"AverageScenario": 1.0}
        # scenarios = {"BelowAverageScenario": 1.0}
        for scen, prob in scenarios.items():
            local_farmer = tc.Farmer()
            local_farmer.scenario_probabilities = {scen: 1.0}
            local_farmer.scenarios = [scen]
            t0 = time.time()
            # opt, m = tc.Farmer.setup_farmer(local_farmer, solver_name=mip_solver, transform="standard_lp")
            opt, m = tc.Farmer.setup_farmer(
                local_farmer, solver_name=mip_solver, transform=transform
            )

            print("\n")
            print(
                "{0:<15}{1:<15}{2:<15}{3:<15}{4:<15}".format(
                    "# Cuts", "Corn", "Sugar Beets", "Wheat", "Total_Time"
                )
            )
            for i in range(30):
                res = opt.solve(m, tee=False)
                cuts_added = m.benders.generate_cut()
                # for c in cuts_added:
                #     opt.add_constraint(c)
                print(
                    "{0:<15}{1:<15.2f}{2:<15.2f}{3:<15.2f}{4:<15.2f}".format(
                        len(cuts_added),
                        m.devoted_acreage["CORN"].value,
                        m.devoted_acreage["SUGAR_BEETS"].value,
                        m.devoted_acreage["WHEAT"].value,
                        time.time() - t0,
                    )
                )
                if len(cuts_added) == 0:
                    break

            expected_result = expected_crop_answers[scen]
            self.assertAlmostEqual(
                m.devoted_acreage["CORN"].value, expected_result["CORN"], 7
            )
            self.assertAlmostEqual(
                m.devoted_acreage["SUGAR_BEETS"].value,
                expected_result["SUGAR_BEETS"],
                7,
            )
            self.assertAlmostEqual(
                m.devoted_acreage["WHEAT"].value, expected_result["WHEAT"], 7
            )
            self.assertAlmostEqual(pyo.value(m.obj), expected_obj_answers[scen], 0)

    @unittest.skipIf(not numpy_available, "numpy is not available.")
    @unittest.skipIf(not gurobi_available, "Gurobi is not available.")
    @parameterized.expand(
        transforms,
        skip_on_empty=True,
    )
    def test_farmer_gurobi_persistent(self, transform):
        solver_name = "gurobi_persistent"
        t0 = time.time()
        opt, m = tc.Farmer.setup_farmer_gurobi_persistent(
            tc.Farmer(),
            transform=transform,
        )
        print(
            "{0:<15}{1:<15}{2:<15}{3:<15}{4:<15}".format(
                "# Cuts", "Corn", "Sugar Beets", "Wheat", "Total_Time"
            )
        )
        for i in range(30):
            res = opt.solve(tee=False, save_results=False)
            cuts_added = m.benders.generate_cut()
            for c in cuts_added:
                opt.add_constraint(c)
            print(
                "{0:<15}{1:<15.2f}{2:<15.2f}{3:<15.2f}{4:<15.2f}".format(
                    len(cuts_added),
                    m.devoted_acreage["CORN"].value,
                    m.devoted_acreage["SUGAR_BEETS"].value,
                    m.devoted_acreage["WHEAT"].value,
                    time.time() - t0,
                )
            )
            if len(cuts_added) == 0:
                break

        self.assertAlmostEqual(m.devoted_acreage["CORN"].value, 80, 7)
        self.assertAlmostEqual(m.devoted_acreage["SUGAR_BEETS"].value, 250, 7)
        self.assertAlmostEqual(m.devoted_acreage["WHEAT"].value, 170, 7)
        self.assertAlmostEqual(pyo.value(m.obj), -108390, 0)

    @unittest.skipIf(not numpy_available, "numpy is not available.")
    @parameterized.expand(
        input=iter_product(non_persistent_mip_solvers, transforms),
        name_func=lambda func, num, params: f"{func.__name__}_solver_{params.args[0]}_transform_{params.args[1]}",
        skip_on_empty=True,
    )
    def test_farmer(self, mip_solver, transform):

        t0 = time.time()
        opt, m = tc.Farmer.setup_farmer(
            tc.Farmer(),
            solver_name=mip_solver,
            # transform=default_transform,
            transform=transform,
        )

        print(
            "{0:<15}{1:<15}{2:<15}{3:<15}{4:<15}".format(
                "# Cuts", "Corn", "Sugar Beets", "Wheat", "Total_Time"
            )
        )
        for i in range(30):
            res = opt.solve(m, tee=False)
            cuts_added = m.benders.generate_cut()
            # for c in cuts_added:
            #     opt.add_constraint(c)
            print(
                "{0:<15}{1:<15.2f}{2:<15.2f}{3:<15.2f}{4:<15.2f}".format(
                    len(cuts_added),
                    m.devoted_acreage["CORN"].value,
                    m.devoted_acreage["SUGAR_BEETS"].value,
                    m.devoted_acreage["WHEAT"].value,
                    time.time() - t0,
                )
            )
            if len(cuts_added) == 0:
                break

        self.assertAlmostEqual(m.devoted_acreage["CORN"].value, 80, 7)
        self.assertAlmostEqual(m.devoted_acreage["SUGAR_BEETS"].value, 250, 7)
        self.assertAlmostEqual(m.devoted_acreage["WHEAT"].value, 170, 7)

    #
    # Grothey Tests
    #

    @unittest.skipIf(not numpy_available, "numpy is not available.")
    @unittest.skipIf(len(qp_solvers) == 0, "No Solver with Quadratic Support Available")
    @unittest.skipIf(
        len(non_linear_solvers) == 0,
        "No Solver with general Non-linear Support Available",
    )
    @parameterized.expand(
        input=iter_product(qp_solvers, non_linear_solvers),
        name_func=lambda func, num, params: f"{func.__name__}_master_sol_{params.args[0]}_sub_sol_{params.args[1]}",
        skip_on_empty=True,
    )
    def test_grothey(self, qp_solver, nl_solver):
        print(f"Master solver {qp_solvers=}, Subproblem Solver {nl_solver=}")
        master_problem_solver = qp_solver
        subproblem_solver = nl_solver
        m = tc.Grothey.create_root()
        root_vars = [m.y]
        m.benders = BendersCutGenerator()
        m.benders.set_input(root_vars=root_vars, tol=1e-8)
        m.benders.add_subproblem(
            subproblem_fn=tc.Grothey.create_subproblem,
            subproblem_fn_kwargs={"root": m},
            root_eta=m.eta,
            subproblem_solver=subproblem_solver,
        )
        opt = pyo.SolverFactory(master_problem_solver)

        for i in range(30):
            res = opt.solve(m, tee=False)
            cuts_added = m.benders.generate_cut()
            if len(cuts_added) == 0:
                break
        self.assertAlmostEqual(m.y.value, 2.721381, 4)
        self.assertAlmostEqual(m.eta.value, -0.0337568, 4)

    #
    # abs Tests
    #

    # @unittest.skipIf(True, "Temporary")
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    def test_abs(self, solver):

        m = tc.absolute_value.create_root()
        root_vars = [m.x]
        m.benders = BendersCutGenerator()
        m.benders.set_input(root_vars=root_vars, tol=1e-8, transform=default_transform)
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
