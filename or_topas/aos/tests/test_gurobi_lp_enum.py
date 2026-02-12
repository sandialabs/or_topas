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

from pyomo.common.dependencies import numpy_available
from pyomo.common import unittest

import pyomo.common.errors
import or_topas.aos.tests.test_cases as tc
from or_topas.aos import gurobi_enumerate_linear_solutions
from or_topas.util import pyomo_utils
from pyomo.opt import check_available_solvers

import pyomo.environ as pyo
import warnings

gurobi_available = len(check_available_solvers("gurobi")) == 1

#
# TODO: Setup detailed tests here
#


@unittest.skipUnless(gurobi_available, "Gurobi MIP solver not available")
@unittest.skipUnless(numpy_available, "NumPy not found")
class TestLPEnumSolnpool(unittest.TestCase):

    def test_non_positive_num_solutions(self):
        """
        Confirm that an exception is thrown with a non-positive num solutions
        """
        n = tc.get_pentagonal_pyramid_mip()
        with self.assertRaises(ValueError):
            gurobi_enumerate_linear_solutions(n, num_solutions=-1)

    def test_generation(self):
        n = tc.get_pentagonal_pyramid_mip()
        n.x.domain = pyo.Reals
        n.y.domain = pyo.Reals

        sols = gurobi_enumerate_linear_solutions(n, tee=True)

        assert len(sols) == 7

    def test_generation_with_variable_count_check(self):
        # checks that the model is restored to same variable count/names
        # as before the gurobi_enumerate_linear_solutions call
        n = tc.get_pentagonal_pyramid_mip()
        n.x.domain = pyo.Reals
        n.y.domain = pyo.Reals

        all_variables_before_solve = pyomo_utils.get_model_variables(n)
        sols = gurobi_enumerate_linear_solutions(n, tee=True)
        all_variables_after_solve = pyomo_utils.get_model_variables(n)
        all_variables_before_solve_names = [
            var.name for var in all_variables_before_solve
        ]
        all_variables_after_solve_names = [
            var.name for var in all_variables_after_solve
        ]

        assert len(sols) == 7
        assert len(all_variables_before_solve) == len(all_variables_after_solve)
        assert set(all_variables_before_solve_names) == set(
            all_variables_after_solve_names
        )

    def test_generation_twice(self):
        # tests that the correct number of solutions are generated in repeated solves
        # also implicitly tests that no error is raised by second aos call
        n = tc.get_pentagonal_pyramid_mip()
        n.x.domain = pyo.Reals
        n.y.domain = pyo.Reals

        sols = gurobi_enumerate_linear_solutions(n, tee=True)
        assert len(sols) == 7
        sols_2 = gurobi_enumerate_linear_solutions(n, tee=True)
        assert len(sols_2) == 7

    def test_triangle_lp(self):
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
            sols = gurobi_enumerate_linear_solutions(m, abs_opt_gap=abs_tol)
            assert len(sols) == sum(m.num_ranked_solns)
            sol_set = set()
            for s in sols:
                s_x = s.variable("x")
                s_y = s.variable("y")
                sol_set.add(
                    ((int(s_x.value), int(s_y.value)), int(s.objective().value))
                )
            assert set(m.feasible_sols) == sol_set

    def test_triangle_milp_fix_integer(self):
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
            sols = gurobi_enumerate_linear_solutions(m, abs_opt_gap=abs_tol)
            assert len(sols) == sum(m.num_ranked_solns)
            sol_set = set()
            for s in sols:
                s_x = s.variable("x")
                s_y = s.variable("y")
                sol_set.add(
                    ((int(s_x.value), int(s_y.value)), int(s.objective().value))
                )
            assert set(m.feasible_sols) == sol_set

    def test_trivial_2d_box_lp_minimize(self):
        """
        Simple AOS test on 2D box example.
        Details in test_case.py for get_trivial_2d_box.
        Minimization case
        """

        m = tc.get_trivial_2d_box_lp(sense=pyo.minimize)
        sols = gurobi_enumerate_linear_solutions(m)
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

    def test_trivial_2d_box_lp_maximize(self):
        """
        Simple AOS test on 2D box example.
        Details in test_case.py for get_trivial_2d_box.
        Minimization case
        """

        m = tc.get_trivial_2d_box_lp(sense=pyo.maximize)
        sols = gurobi_enumerate_linear_solutions(m)
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

    def test_lp_enum_upper_objective_bound(self):
        """
        Simple AOS test on 2D box example using upper objective bound
        Details in test_case.py for get_trivial_2d_box.
        """

        m = tc.get_trivial_2d_box_lp(sense=pyo.minimize)
        # this case should keep all the same solutions
        sols = gurobi_enumerate_linear_solutions(m, upper_objective_threshold=2)
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
        sols = gurobi_enumerate_linear_solutions(m, upper_objective_threshold=1)
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
        sols = gurobi_enumerate_linear_solutions(m, upper_objective_threshold=0)
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
            sols = gurobi_enumerate_linear_solutions(m, upper_objective_threshold=-1)
        assert len(wlist) == 1
        assert len(sols) == 0
        self.assertIs(wlist[0].category, RuntimeWarning)
        self.assertIn(
            str(wlist[0].message),
            "upper_objective_threshold violated at optimum, no valid solutions",
        )

    def test_lp_enum_lower_objective_bound(self):
        """
        Simple AOS test on 2D box example using lower objective bound
        Details in test_case.py for get_trivial_2d_box.
        """
        m = tc.get_trivial_2d_box_lp(sense=pyo.maximize)
        # this case should keep all the same solutions
        sols = gurobi_enumerate_linear_solutions(m, lower_objective_threshold=0)
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
        sols = gurobi_enumerate_linear_solutions(m, lower_objective_threshold=1)
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
        sols = gurobi_enumerate_linear_solutions(m, lower_objective_threshold=2)
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
            sols = gurobi_enumerate_linear_solutions(m, lower_objective_threshold=3)
        assert len(wlist) == 1
        assert len(sols) == 0
        self.assertIs(wlist[0].category, RuntimeWarning)
        self.assertIn(
            str(wlist[0].message),
            "lower_objective_threshold violated at optimum, no valid solutions",
        )
