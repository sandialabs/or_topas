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

from pyomo.common.dependencies import numpy as numpy, numpy_available

import pyomo.environ as pyo
import pyomo.opt
from pyomo.common import unittest
from pyomo.common.dependencies import attempt_import

parameterized, param_available = attempt_import("parameterized")
if not param_available:
    raise unittest.SkipTest("Parameterized is not available.")
parameterized = parameterized.parameterized

import or_topas.aos.tests.test_cases as tc
from or_topas.aos import lp_enum

#
# Find available solvers. Just use GLPK if it's available.
#
solvers = list(pyomo.opt.check_available_solvers("glpk", "gurobi"))

timelimit = {"gurobi": "TimeLimit", "appsi_gurobi": "TimeLimit", "glpk": "tmlim"}


class TestLPEnum(unittest.TestCase):

    @parameterized.expand(input=solvers)
    def test_bad_solver(self, mip_solver):
        """
        Confirm that an exception is thrown with a bad solver name.
        """
        m = tc.get_3d_polyhedron_problem()
        with self.assertRaises(pyomo.common.errors.ApplicationError):
            lp_enum.enumerate_linear_solutions(m, solver="unknown_solver")

    @parameterized.expand(input=solvers)
    def test_non_positive_num_solutions(self, mip_solver):
        """
        Confirm that an exception is thrown with a non-positive num solutions
        """
        m = tc.get_3d_polyhedron_problem()
        with self.assertRaises(ValueError):
            lp_enum.enumerate_linear_solutions(m, num_solutions=-1, solver=mip_solver)

    @unittest.skipIf(True, "Ignoring fragile test for solver timeout.")
    @parameterized.expand(input=solvers)
    def test_no_time(self, mip_solver):
        """
        Check that the correct bounds are found for a discrete problem where
        more restrictive bounds are implied by the constraints.
        """
        m = tc.get_3d_polyhedron_problem()
        with unittest.pytest.raises(Exception):
            lp_enum.enumerate_linear_solutions(
                m, solver=mip_solver, solver_options={timelimit[mip_solver]: 0}
            )

    @parameterized.expand(input=solvers)
    def test_3d_polyhedron(self, mip_solver):
        m = tc.get_3d_polyhedron_problem()
        m.o.deactivate()
        m.obj = pyo.Objective(expr=m.x[0] + m.x[1] + m.x[2])

        sols = lp_enum.enumerate_linear_solutions(m, solver=mip_solver)
        assert len(sols) == 2
        for s in sols:
            assert s.objective().value == unittest.pytest.approx(4)

    @parameterized.expand(input=solvers)
    def test_3d_polyhedron(self, mip_solver):
        m = tc.get_3d_polyhedron_problem()
        m.o.deactivate()
        m.obj = pyo.Objective(expr=m.x[0] + 2 * m.x[1] + 3 * m.x[2])

        sols = lp_enum.enumerate_linear_solutions(m, solver=mip_solver)
        assert len(sols) == 2
        for s in sols:
            assert s.objective().value == unittest.pytest.approx(
                9
            ) or s.objective().value == unittest.pytest.approx(10)

    @parameterized.expand(input=solvers)
    def test_2d_diamond_problem(self, mip_solver):
        m = tc.get_2d_diamond_problem()
        sols = lp_enum.enumerate_linear_solutions(m, solver=mip_solver, num_solutions=2)
        assert len(sols) == 2
        for s in sols:
            print(s)
        assert sols[0].objective().value == unittest.pytest.approx(6.789473684210527)
        assert sols[1].objective().value == unittest.pytest.approx(3.6923076923076916)

    @parameterized.expand(input=solvers)
    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_pentagonal_pyramid(self, mip_solver):
        n = tc.get_pentagonal_pyramid_mip()
        n.o.sense = pyo.minimize
        n.x.domain = pyo.Reals
        n.y.domain = pyo.Reals

        sols = lp_enum.enumerate_linear_solutions(n, solver=mip_solver, tee=False)
        for s in sols:
            print(s)
        assert len(sols) == 6

    @parameterized.expand(input=solvers)
    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_pentagon(self, mip_solver):
        n = tc.get_pentagonal_lp()

        sols = lp_enum.enumerate_linear_solutions(n, solver=mip_solver)
        for s in sols:
            print(s)
        assert len(sols) == 6

    @parameterized.expand(input=solvers)
    def test_triangle_lp(self, mip_solver):
        """
        Test that AOS method can be called multiple times in a row.
        Uses adaptive test from test cases.
        Feasible region is a right triangle with vertices (x,y) = (5,0), (0,5), (0,0)
        Objective is x+y
        Runs repeatedly changing the absolute gap tol at 0,1,2,3,4,5
        Checks that vertices found are the expected ones.
        Details in test_case.py
        """
        for level in range(0, 6):
            abs_tol = 5 - level
            m = tc.get_triangle_lp(level=level)
            sols = lp_enum.enumerate_linear_solutions(
                m, solver=mip_solver, abs_opt_gap=abs_tol
            )
            assert len(sols) == sum(m.num_ranked_solns)
            sol_set = set()
            for s in sols:
                s_x = s.variable("x")
                s_y = s.variable("y")
                sol_set.add(
                    ((int(s_x.value), int(s_y.value)), int(s.objective().value))
                )
            assert set(m.feasible_sols) == sol_set

    @parameterized.expand(input=solvers)
    def test_triangle_milp_fix_integer(self, mip_solver):
        """
        Test that AOS method can be called multiple times in a row and handle all integers fixed
        All integer fixed converts the MILP to effectively an LP
        Uses adaptive test from test cases.
        Feasible region is a right triangle with vertices (x,y) = (5,0), (0,5), (0,0)
        Objective is x+y
        Runs repeatedly changing the absolute gap tol at 0,1,2,3,4,5
        Checks that vertices found are the expected ones.
        Details in test_case.py
        """
        for level in range(0, 6):
            abs_tol = 5 - level
            m = tc.get_triangle_lp(level=level)
            sols = lp_enum.enumerate_linear_solutions(
                m, solver=mip_solver, abs_opt_gap=abs_tol
            )
            assert len(sols) == sum(m.num_ranked_solns)
            sol_set = set()
            for s in sols:
                s_x = s.variable("x")
                s_y = s.variable("y")
                sol_set.add(
                    ((int(s_x.value), int(s_y.value)), int(s.objective().value))
                )
            assert set(m.feasible_sols) == sol_set
