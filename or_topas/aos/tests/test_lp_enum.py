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
from or_topas.util import pyomo_utils
import warnings

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
    def test_3d_polyhedron_with_variable_count_check(self, mip_solver):
        """
        Test that model restored to same variables before and after AOS call
        Also checks for solution accuracy
        """
        m = tc.get_3d_polyhedron_problem()
        m.o.deactivate()
        m.obj = pyo.Objective(expr=m.x[0] + m.x[1] + m.x[2])

        all_variables_before_solve = pyomo_utils.get_model_variables(m)
        sols = lp_enum.enumerate_linear_solutions(m, solver=mip_solver)
        all_variables_after_solve = pyomo_utils.get_model_variables(m)
        all_variables_before_solve_names = [
            var.name for var in all_variables_before_solve
        ]
        all_variables_after_solve_names = [
            var.name for var in all_variables_after_solve
        ]
        assert len(sols) == 2
        for s in sols:
            assert s.objective().value == unittest.pytest.approx(4)
        assert len(all_variables_before_solve) == len(all_variables_after_solve)
        assert set(all_variables_before_solve_names) == set(
            all_variables_after_solve_names
        )

    @parameterized.expand(input=solvers)
    def test_3d_polyhedron_called_twice(self, mip_solver):
        """
        Test that AOS method can be called twice in a row with no issues
        Also checks that objective results are the same across solves
        """
        m = tc.get_3d_polyhedron_problem()
        m.o.deactivate()
        m.obj = pyo.Objective(expr=m.x[0] + m.x[1] + m.x[2])

        all_variables_before_solve = pyomo_utils.get_model_variables(m)
        sols = lp_enum.enumerate_linear_solutions(m, solver=mip_solver)
        all_variables_after_solve = pyomo_utils.get_model_variables(m)
        assert len(sols) == 2
        for s in sols:
            assert s.objective().value == unittest.pytest.approx(4)
        assert len(all_variables_before_solve) == len(all_variables_after_solve)

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

    @parameterized.expand(input=solvers)
    def test_trivial_2d_box_lp_minimize(self, mip_solver):
        """
        Simple AOS test on 2D box example.
        Details in test_case.py for get_trivial_2d_box.
        Minimization case
        """

        m = tc.get_trivial_2d_box_lp(sense=pyo.minimize)
        sols = lp_enum.enumerate_linear_solutions(m, solver=mip_solver)
        assert len(sols) == sum(m.num_ranked_solns)
        sol_list = list()
        for s in sols:
            s_x = s.variable("x")
            s_y = s.variable("y")
            sol_list.append(
                ((int(s_x.value), int(s_y.value)), int(s.objective().value))
            )
        assert set(m.feasible_sols) == set(sol_list)
        assert m.feasible_sols[0] == sol_list[0]

    @parameterized.expand(input=solvers)
    def test_trivial_2d_box_lp_maximize(self, mip_solver):
        """
        Simple AOS test on 2D box example.
        Details in test_case.py for get_trivial_2d_box.
        Minimization case
        """

        m = tc.get_trivial_2d_box_lp(sense=pyo.maximize)
        sols = lp_enum.enumerate_linear_solutions(m, solver=mip_solver)
        assert len(sols) == sum(m.num_ranked_solns)
        sol_list = list()
        for s in sols:
            s_x = s.variable("x")
            s_y = s.variable("y")
            sol_list.append(
                ((int(s_x.value), int(s_y.value)), int(s.objective().value))
            )
        assert set(m.feasible_sols) == set(sol_list)
        assert m.feasible_sols[0] == sol_list[0]

    @parameterized.expand(input=solvers)
    def test_lp_enum_upper_objective_bound(self, mip_solver):
        """
        Simple AOS test on 2D box example using upper objective bound
        Details in test_case.py for get_trivial_2d_box.
        """

        m = tc.get_trivial_2d_box_lp(sense=pyo.minimize)
        # this case should keep all the same solutions
        sols = lp_enum.enumerate_linear_solutions(
            m, solver=mip_solver, upper_objective_threshold=2
        )
        assert len(sols) == sum(m.num_ranked_solns)
        sol_list = list()
        for s in sols:
            s_x = s.variable("x")
            s_y = s.variable("y")
            sol_list.append(
                ((int(s_x.value), int(s_y.value)), int(s.objective().value))
            )
        assert set(m.feasible_sols) == set(sol_list)
        assert m.feasible_sols[0] == sol_list[0]

        # this case should get sol_list 0-2 as solutions
        sols = lp_enum.enumerate_linear_solutions(
            m, solver=mip_solver, upper_objective_threshold=1
        )
        assert len(sols) == sum(m.num_ranked_solns[0:2])
        sol_list = list()
        for s in sols:
            s_x = s.variable("x")
            s_y = s.variable("y")
            sol_list.append(
                ((int(s_x.value), int(s_y.value)), int(s.objective().value))
            )
        assert set(m.feasible_sols[0:3]) == set(sol_list)
        assert m.feasible_sols[0] == sol_list[0]

        # this case should get sol_list 0 as solutions
        sols = lp_enum.enumerate_linear_solutions(
            m, solver=mip_solver, upper_objective_threshold=0
        )
        assert len(sols) == sum(m.num_ranked_solns[0:1])
        sol_list = list()
        for s in sols:
            s_x = s.variable("x")
            s_y = s.variable("y")
            sol_list.append(
                ((int(s_x.value), int(s_y.value)), int(s.objective().value))
            )
        assert m.feasible_sols[0] == sol_list[0]

        # this case should get 0 solutions as none are feasible with this bound
        # should raise warning
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            sols = lp_enum.enumerate_linear_solutions(
                m, solver=mip_solver, upper_objective_threshold=-1
            )
        assert len(wlist) == 1
        assert len(sols) == 0
        self.assertIs(wlist[0].category, RuntimeWarning)
        self.assertIn(
            str(wlist[0].message),
            "upper_objective_threshold violated at optimum, no valid solutions",
        )

    def test_lp_enum_lower_objective_bound_gurobi(self):
        """
        Simple AOS test on 2D box example using lower objective bound
        Details in test_case.py for get_trivial_2d_box.
        """
        mip_solver = "gurobi"
        m = tc.get_trivial_2d_box_lp(sense=pyo.maximize)
        # this case should keep all the same solutions
        sols = lp_enum.enumerate_linear_solutions(
            m, solver=mip_solver, lower_objective_threshold=0
        )
        assert len(sols) == sum(m.num_ranked_solns)
        sol_list = list()
        for s in sols:
            s_x = s.variable("x")
            s_y = s.variable("y")
            print(((int(s_x.value), int(s_y.value)), int(s.objective().value)))
            sol_list.append(
                ((int(s_x.value), int(s_y.value)), int(s.objective().value))
            )
        assert set(m.feasible_sols) == set(sol_list)
        assert m.feasible_sols[0] == sol_list[0]

        # this case should get sol_list 0-2 as solutions
        sols = lp_enum.enumerate_linear_solutions(
            m, solver=mip_solver, lower_objective_threshold=1
        )
        assert len(sols) == sum(m.num_ranked_solns[0:2])
        sol_list = list()
        for s in sols:
            s_x = s.variable("x")
            s_y = s.variable("y")
            sol_list.append(
                ((int(s_x.value), int(s_y.value)), int(s.objective().value))
            )
        assert set(m.feasible_sols[0:3]) == set(sol_list)
        assert m.feasible_sols[0] == sol_list[0]

        # this case should get sol_list 0 as solutions
        sols = lp_enum.enumerate_linear_solutions(
            m, solver=mip_solver, lower_objective_threshold=2
        )
        assert len(sols) == sum(m.num_ranked_solns[0:1])
        sol_list = list()
        for s in sols:
            s_x = s.variable("x")
            s_y = s.variable("y")
            sol_list.append(
                ((int(s_x.value), int(s_y.value)), int(s.objective().value))
            )
        assert m.feasible_sols[0] == sol_list[0]

        # this case should get 0 solutions as none are feasible with this bound
        # should raise warning
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            sols = lp_enum.enumerate_linear_solutions(
                m, solver=mip_solver, lower_objective_threshold=3
            )
        assert len(wlist) == 1
        assert len(sols) == 0
        self.assertIs(wlist[0].category, RuntimeWarning)
        self.assertIn(
            str(wlist[0].message),
            "lower_objective_threshold violated at optimum, no valid solutions",
        )

    # MPV: TODO for WEH, look at what is going on in these GLPK rounding issues
    # there isn't an issue in gurobi on the same tests
    # this issue is not fixed by changing within from NonNegativeReals to Reals
    # issue only appears to occur in maximization case
    @unittest.skipIf(True, "GPLK rounding issues")
    def test_lp_enum_lower_objective_bound_glpk(self):
        """
        Simple AOS test on 2D box example using lower objective bound
        Details in test_case.py for get_trivial_2d_box.
        """
        mip_solver = "glpk"
        m = tc.get_trivial_2d_box_lp(sense=pyo.maximize)
        # this case should keep all the same solutions
        sols = lp_enum.enumerate_linear_solutions(
            m, solver=mip_solver, lower_objective_threshold=0
        )
        assert len(sols) == sum(m.num_ranked_solns)
        sol_list = list()
        for s in sols:
            s_x = s.variable("x")
            s_y = s.variable("y")
            print(((int(s_x.value), int(s_y.value)), int(s.objective().value)))
            sol_list.append(
                ((int(s_x.value), int(s_y.value)), int(s.objective().value))
            )
        assert set(m.feasible_sols) == set(sol_list)
        assert m.feasible_sols[0] == sol_list[0]

        # this case should get sol_list 0-2 as solutions
        sols = lp_enum.enumerate_linear_solutions(
            m, solver=mip_solver, lower_objective_threshold=1
        )
        assert len(sols) == sum(m.num_ranked_solns[0:2])
        sol_list = list()
        for s in sols:
            s_x = s.variable("x")
            s_y = s.variable("y")
            sol_list.append(
                ((int(s_x.value), int(s_y.value)), int(s.objective().value))
            )
        assert set(m.feasible_sols[0:3]) == set(sol_list)
        assert m.feasible_sols[0] == sol_list[0]

        # this case should get sol_list 0 as solutions
        sols = lp_enum.enumerate_linear_solutions(
            m, solver=mip_solver, lower_objective_threshold=2
        )
        assert len(sols) == sum(m.num_ranked_solns[0:1])
        sol_list = list()
        for s in sols:
            s_x = s.variable("x")
            s_y = s.variable("y")
            sol_list.append(
                ((int(s_x.value), int(s_y.value)), int(s.objective().value))
            )
        assert m.feasible_sols[0] == sol_list[0]

        # this case should get 0 solutions as none are feasible with this bound
        # should raise warning
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            sols = lp_enum.enumerate_linear_solutions(
                m, solver=mip_solver, lower_objective_threshold=3
            )
        assert len(wlist) == 1
        assert len(sols) == 0
        self.assertIs(wlist[0].category, RuntimeWarning)
        self.assertIn(
            str(wlist[0].message),
            "lower_objective_threshold violated at optimum, no valid solutions",
        )

    @unittest.skipIf(True, "GPLK rounding issues")
    def test_lp_enum_lower_objective_bound_glpk_variant(self):
        """
        Simple AOS test on 2D box example using lower objective bound
        Details in test_case.py for get_trivial_2d_box.
        """
        mip_solver = "glpk"
        m = tc.get_trivial_2d_box_lp_variant(sense=pyo.maximize)
        # this case should keep all the same solutions
        sols = lp_enum.enumerate_linear_solutions(
            m, solver=mip_solver, lower_objective_threshold=0
        )
        assert len(sols) == sum(m.num_ranked_solns)
        sol_list = list()
        for s in sols:
            s_x = s.variable("x")
            s_y = s.variable("y")
            print(((int(s_x.value), int(s_y.value)), int(s.objective().value)))
            sol_list.append(
                ((int(s_x.value), int(s_y.value)), int(s.objective().value))
            )
        assert set(m.feasible_sols) == set(sol_list)
        assert m.feasible_sols[0] == sol_list[0]

        # this case should get sol_list 0-2 as solutions
        sols = lp_enum.enumerate_linear_solutions(
            m, solver=mip_solver, lower_objective_threshold=1
        )
        assert len(sols) == sum(m.num_ranked_solns[0:2])
        sol_list = list()
        for s in sols:
            s_x = s.variable("x")
            s_y = s.variable("y")
            sol_list.append(
                ((int(s_x.value), int(s_y.value)), int(s.objective().value))
            )
        assert set(m.feasible_sols[0:3]) == set(sol_list)
        assert m.feasible_sols[0] == sol_list[0]

        # this case should get sol_list 0 as solutions
        sols = lp_enum.enumerate_linear_solutions(
            m, solver=mip_solver, lower_objective_threshold=2
        )
        assert len(sols) == sum(m.num_ranked_solns[0:1])
        sol_list = list()
        for s in sols:
            s_x = s.variable("x")
            s_y = s.variable("y")
            sol_list.append(
                ((int(s_x.value), int(s_y.value)), int(s.objective().value))
            )
        assert m.feasible_sols[0] == sol_list[0]

        # this case should get 0 solutions as none are feasible with this bound
        # should raise warning
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            sols = lp_enum.enumerate_linear_solutions(
                m, solver=mip_solver, lower_objective_threshold=3
            )
        assert len(wlist) == 1
        assert len(sols) == 0
        self.assertIs(wlist[0].category, RuntimeWarning)
        self.assertIn(
            str(wlist[0].message),
            "lower_objective_threshold violated at optimum, no valid solutions",
        )
