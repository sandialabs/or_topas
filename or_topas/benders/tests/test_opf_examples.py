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
        file_path = "PGLIB_Data/pglib_opf_case5_pjm.m"
        grid = tc.MatpowerGrid(m_file=file_path)
        m = tc.EnergyGrid.create_root(grid=grid)
        root_vars = list(m.generation.values())
        m.benders = BendersCutGenerator()
        m.benders.set_input(root_vars=root_vars, tol=1e-6, transform="standard_lp", allow_infeasible=True)
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
        file_path = "PGLIB_Data/pglib_opf_case5_pjm.m"
        grid = tc.MatpowerGrid(m_file=file_path)
        self.assertEqual(len(grid.buses), 5)
        model = tc.EnergyGrid.create_tiny_opf(grid, mode=2)
        if True:
            model.write(
                filename='model.lp',
                format='lp',
                io_options={
                    'symbolic_solver_labels': True,      # human-readable variable/constraint names
                    'file_determinism': 2,               # SORT_INDICES (or 3 for full symbol sort)
                    'skip_trivial_constraints': False,   # usually keep them for diagnosis
                }
            )
        opt = pyo.SolverFactory("gurobi_persistent")
        opt.set_instance(model)
        res = opt.solve(tee=False)
        self.assertEqual(res.solver.termination_condition, pyo.TerminationCondition.optimal)

    def test_matpower_case14_smoke(self):
        file_path = "PGLIB_Data/pglib_opf_case14_ieee.m"
        grid = tc.MatpowerGrid(m_file=file_path)
        self.assertEqual(len(grid.buses), 14)