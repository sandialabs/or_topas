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
from or_topas.benders.benders_serial import (
    BendersGenerator_Serial as BendersCutGenerator,
)

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


class TestBenders(unittest.TestCase):
    # TODO: generalize and add single scenario farmer test
    class Farmer:
        def __init__(self):
            self.crops = ["WHEAT", "CORN", "SUGAR_BEETS"]
            self.total_acreage = 500
            self.PriceQuota = {
                "WHEAT": 100000.0,
                "CORN": 100000.0,
                "SUGAR_BEETS": 6000.0,
            }
            self.SubQuotaSellingPrice = {
                "WHEAT": 170.0,
                "CORN": 150.0,
                "SUGAR_BEETS": 36.0,
            }
            self.SuperQuotaSellingPrice = {
                "WHEAT": 0.0,
                "CORN": 0.0,
                "SUGAR_BEETS": 10.0,
            }
            self.CattleFeedRequirement = {
                "WHEAT": 200.0,
                "CORN": 240.0,
                "SUGAR_BEETS": 0.0,
            }
            self.PurchasePrice = {
                "WHEAT": 238.0,
                "CORN": 210.0,
                "SUGAR_BEETS": 100000.0,
            }
            self.PlantingCostPerAcre = {
                "WHEAT": 150.0,
                "CORN": 230.0,
                "SUGAR_BEETS": 260.0,
            }
            self.scenarios = [
                "BelowAverageScenario",
                "AverageScenario",
                "AboveAverageScenario",
            ]
            self.crop_yield = dict()
            self.crop_yield["BelowAverageScenario"] = {
                "WHEAT": 2.0,
                "CORN": 2.4,
                "SUGAR_BEETS": 16.0,
            }
            self.crop_yield["AverageScenario"] = {
                "WHEAT": 2.5,
                "CORN": 3.0,
                "SUGAR_BEETS": 20.0,
            }
            self.crop_yield["AboveAverageScenario"] = {
                "WHEAT": 3.0,
                "CORN": 3.6,
                "SUGAR_BEETS": 24.0,
            }
            self.scenario_probabilities = dict()
            self.scenario_probabilities["BelowAverageScenario"] = 0.3333
            self.scenario_probabilities["AverageScenario"] = 0.3334
            self.scenario_probabilities["AboveAverageScenario"] = 0.3333

        @staticmethod
        def create_root(farmer):
            m = pyo.ConcreteModel()

            m.crops = pyo.Set(initialize=farmer.crops, ordered=True)
            m.scenarios = pyo.Set(initialize=farmer.scenarios, ordered=True)

            m.devoted_acreage = pyo.Var(m.crops, bounds=(0, farmer.total_acreage))
            m.eta = pyo.Var(m.scenarios)
            for s in m.scenarios:
                m.eta[s].setlb(-432000 * farmer.scenario_probabilities[s])

            m.total_acreage_con = pyo.Constraint(
                expr=sum(m.devoted_acreage.values()) <= farmer.total_acreage
            )

            m.obj = pyo.Objective(
                expr=sum(
                    farmer.PlantingCostPerAcre[crop] * m.devoted_acreage[crop]
                    for crop in m.crops
                )
                + sum(m.eta.values())
            )
            return m

        @staticmethod
        def create_subproblem(root, farmer, scenario):
            m = pyo.ConcreteModel()

            m.crops = pyo.Set(initialize=farmer.crops, ordered=True)

            m.devoted_acreage = pyo.Var(m.crops)
            m.QuantitySubQuotaSold = pyo.Var(m.crops, bounds=(0.0, None))
            m.QuantitySuperQuotaSold = pyo.Var(m.crops, bounds=(0.0, None))
            m.QuantityPurchased = pyo.Var(m.crops, bounds=(0.0, None))

            def EnforceCattleFeedRequirement_rule(m, i):
                return (
                    farmer.CattleFeedRequirement[i]
                    <= (farmer.crop_yield[scenario][i] * m.devoted_acreage[i])
                    + m.QuantityPurchased[i]
                    - m.QuantitySubQuotaSold[i]
                    - m.QuantitySuperQuotaSold[i]
                )

            m.EnforceCattleFeedRequirement = pyo.Constraint(
                m.crops, rule=EnforceCattleFeedRequirement_rule
            )

            def LimitAmountSold_rule(m, i):
                return (
                    m.QuantitySubQuotaSold[i]
                    + m.QuantitySuperQuotaSold[i]
                    - (farmer.crop_yield[scenario][i] * m.devoted_acreage[i])
                    <= 0.0
                )

            m.LimitAmountSold = pyo.Constraint(m.crops, rule=LimitAmountSold_rule)

            def EnforceQuotas_rule(m, i):
                return (0.0, m.QuantitySubQuotaSold[i], farmer.PriceQuota[i])

            m.EnforceQuotas = pyo.Constraint(m.crops, rule=EnforceQuotas_rule)

            obj_expr = sum(
                farmer.PurchasePrice[crop] * m.QuantityPurchased[crop]
                for crop in m.crops
            )
            obj_expr -= sum(
                farmer.SubQuotaSellingPrice[crop] * m.QuantitySubQuotaSold[crop]
                for crop in m.crops
            )
            obj_expr -= sum(
                farmer.SuperQuotaSellingPrice[crop] * m.QuantitySuperQuotaSold[crop]
                for crop in m.crops
            )
            m.obj = pyo.Objective(
                expr=farmer.scenario_probabilities[scenario] * obj_expr
            )

            complicating_vars_map = pyo.ComponentMap()
            for crop in m.crops:
                complicating_vars_map[root.devoted_acreage[crop]] = m.devoted_acreage[
                    crop
                ]

            return m, complicating_vars_map

        @staticmethod
        def setup_farmer_gurobi_persistent(Farmer_Data):
            # designed for gurobi_persistent
            solver_name = "gurobi_persistent"
            farmer = Farmer_Data
            m = TestBenders.Farmer.create_root(farmer=farmer)
            root_vars = list(m.devoted_acreage.values())
            m.benders = BendersCutGenerator()
            m.benders.set_input(root_vars=root_vars, tol=1e-8)
            for s in farmer.scenarios:
                subproblem_fn_kwargs = dict()
                subproblem_fn_kwargs["root"] = m
                subproblem_fn_kwargs["farmer"] = farmer
                subproblem_fn_kwargs["scenario"] = s
                m.benders.add_subproblem(
                    subproblem_fn=TestBenders.Farmer.create_subproblem,
                    subproblem_fn_kwargs=subproblem_fn_kwargs,
                    root_eta=m.eta[s],
                    subproblem_solver=solver_name,
                )
            opt = pyo.SolverFactory(solver_name)
            opt.set_instance(m)
            return opt, m

        @staticmethod
        def setup_farmer(Farmer_Data, solver_name):
            farmer = Farmer_Data
            m = TestBenders.Farmer.create_root(farmer=farmer)
            root_vars = list(m.devoted_acreage.values())
            m.benders = BendersCutGenerator()
            m.benders.set_input(root_vars=root_vars, tol=1e-8)
            for s in farmer.scenarios:
                subproblem_fn_kwargs = dict()
                subproblem_fn_kwargs["root"] = m
                subproblem_fn_kwargs["farmer"] = farmer
                subproblem_fn_kwargs["scenario"] = s
                m.benders.add_subproblem(
                    subproblem_fn=TestBenders.Farmer.create_subproblem,
                    subproblem_fn_kwargs=subproblem_fn_kwargs,
                    root_eta=m.eta[s],
                    subproblem_solver=solver_name,
                )
            opt = pyo.SolverFactory(solver_name)
            return opt, m

    @unittest.skipIf(not numpy_available, "numpy is not available.")
    @unittest.skipIf(not gurobi_available, "Gurobi is not available.")
    def test_farmer_gurobi_persistent(self):
        solver_name = "gurobi_persistent"
        t0 = time.time()
        opt, m = TestBenders.Farmer.setup_farmer_gurobi_persistent(
            self.Farmer(),
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

    @unittest.skipIf(not numpy_available, "numpy is not available.")
    @parameterized.expand(input=non_persistnet_mip_solvers, skip_on_empty=True)
    def test_farmer(self, mip_solver):

        t0 = time.time()
        opt, m = TestBenders.Farmer.setup_farmer(self.Farmer(), solver_name=mip_solver)

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

    @unittest.skipIf(not numpy_available, "numpy is not available.")
    @unittest.skipIf(len(qp_solvers) == 0, "No Solver with Quadratic Support Available")
    @unittest.skipIf(len(non_linear_solvers) == 0, "No Solver with general Non-linear Support Available")
    @parameterized.expand(input=iter_product(qp_solvers, non_linear_solvers),
                          name_func=lambda func, num, params: f"{func.__name__}_master_sol_{params.args[0]}_sub_sol_{params.args[1]}",
                          skip_on_empty=True)
    def test_grothey(self, qp_solver, nl_solver):
        def create_root():
            m = pyo.ConcreteModel()
            m.y = pyo.Var(bounds=(1, None))
            m.eta = pyo.Var(bounds=(-10, None))
            m.obj = pyo.Objective(expr=m.y**2 + m.eta)
            return m

        def create_subproblem(root):
            m = pyo.ConcreteModel()
            m.x1 = pyo.Var()
            m.x2 = pyo.Var()
            m.y = pyo.Var()
            m.obj = pyo.Objective(expr=-m.x2)
            m.c1 = pyo.Constraint(expr=(m.x1 - 1) ** 2 + m.x2**2 <= pyo.log(m.y))
            m.c2 = pyo.Constraint(expr=(m.x1 + 1) ** 2 + m.x2**2 <= pyo.log(m.y))

            complicating_vars_map = pyo.ComponentMap()
            complicating_vars_map[root.y] = m.y

            return m, complicating_vars_map

        print(f"Master solver {qp_solvers=}, Subproblem Solver {nl_solver=}")
        master_problem_solver = qp_solver
        subproblem_solver = nl_solver
        m = create_root()
        root_vars = [m.y]
        m.benders = BendersCutGenerator()
        m.benders.set_input(root_vars=root_vars, tol=1e-8)
        m.benders.add_subproblem(
            subproblem_fn=create_subproblem,
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
