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


class TestBendersSolverNestedSubproblems(unittest.TestCase):

    #
    # Grothey Tests
    #

    @parameterized.expand(
        input=iter_product(qp_solvers, non_linear_solvers),
        name_func=lambda func, num, params: f"{func.__name__}_master_sol_{params.args[0]}_sub_sol_{params.args[1]}",
        skip_on_empty=True,
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    @unittest.skipIf(len(qp_solvers) == 0, "No Solver with Quadratic Support Available")
    @unittest.skipIf(
        len(non_linear_solvers) == 0,
        "No Solver with general Non-linear Support Available",
    )
    def test_grothey_nested_1(self, qp_solver, nl_solver):
        print(f"Master solver {qp_solvers=}, Subproblem Solver {nl_solver=}")
        master_problem_solver = qp_solver
        subproblem_solver = nl_solver
        m = tc.Grothey.create_root()
        root_vars = [m.y]
        m.benders = BendersCutGenerator()
        m.benders.set_input(root_vars=root_vars, tol=1e-8)
        m.benders.add_subproblem(
            subproblem_fn=tc.Grothey.create_nested_subproblem_1,
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

    @parameterized.expand(
        input=iter_product(qp_solvers, non_linear_solvers),
        name_func=lambda func, num, params: f"{func.__name__}_master_sol_{params.args[0]}_sub_sol_{params.args[1]}",
        skip_on_empty=True,
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    @unittest.skipIf(len(qp_solvers) == 0, "No Solver with Quadratic Support Available")
    @unittest.skipIf(
        len(non_linear_solvers) == 0,
        "No Solver with general Non-linear Support Available",
    )
    def test_grothey_nested_2(self, qp_solver, nl_solver):
        print(f"Master solver {qp_solvers=}, Subproblem Solver {nl_solver=}")
        master_problem_solver = qp_solver
        subproblem_solver = nl_solver
        m = tc.Grothey.create_root()
        root_vars = [m.y]
        m.benders = BendersCutGenerator()
        m.benders.set_input(root_vars=root_vars, tol=1e-8)
        m.benders.add_subproblem(
            subproblem_fn=tc.Grothey.create_nested_subproblem_2,
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
    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_abs_nested_subproblem_1(self, solver):

        m = tc.absolute_value.create_root()
        root_vars = [m.x]
        m.benders = BendersCutGenerator()
        m.benders.set_input(root_vars=root_vars, tol=1e-8, transform=default_transform)
        m.benders.add_subproblem(
            subproblem_fn=tc.absolute_value.create_nested_subproblem_1,
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

    # #
    # # modified abs Tests
    # #

    @parameterized.expand(
        input=infeasibility_persistent_test_solvers, skip_on_empty=True
    )
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_modified_abs_with_feas_cuts(self, solver):
        transform = "standard_lp"
        x_set = [-10, -7, 0, 7, 10]
        a_val = [-1, 0, 1]
        for current_a in a_val:
            for index, x_val in enumerate(x_set):
                m = tc.modified_absolute_value.create_root()
                root_vars = [m.x]
                data = MyMunch(a=current_a, L=1, R=1, LB=-6, UB=4)
                m.benders = BendersCutGenerator()
                m.benders.set_input(
                    root_vars=root_vars,
                    tol=1e-8,
                    transform=transform,
                    allow_infeasible=True,
                )
                m.benders.add_subproblem(
                    subproblem_fn=tc.modified_absolute_value.create_nested_subproblem,
                    subproblem_fn_kwargs={"root_x": m.x, "data": data},
                    root_eta=m.eta,
                    subproblem_solver=solver,
                )
                m.x = x_val
                opt = pyo.SolverFactory(solver)
                opt.set_instance(m)
                for i in range(30):
                    res = opt.solve(tee=False, save_results=False)
                    cuts_added = m.benders.generate_cut()
                    for c in cuts_added:
                        opt.add_constraint(c)
                    if len(cuts_added) == 0:
                        break

                self.assertAlmostEqual(m.x.value, current_a, 4)
                self.assertAlmostEqual(pyo.value(m.obj), 0.0, 4)
                self.assertAlmostEqual(m.eta.value, 0.0, 4)
