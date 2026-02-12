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

from collections import Counter

from pyomo.common import unittest
from pyomo.common.dependencies import numpy as np, numpy_available
import pyomo.environ as pyo
from or_topas.aos import gurobi_generate_solutions
from pyomo.contrib.appsi.solvers import Gurobi
from pyomo.common.errors import ApplicationError
import or_topas.aos.tests.test_cases as tc

gurobipy_available = Gurobi().available()


@unittest.skipIf(not gurobipy_available, "Gurobi MIP solver not available")
class TestGurobiSolnPoolUnit(unittest.TestCase):
    """
    Cases to cover:

        LP feasibility (for an LP just one solution should be returned since gurobi cannot enumerate over continuous vars)

        Pass at least one solver option to make sure that work, e.g. time limit

        We need a utility to check that a two sets of solutions are the same.
        Maybe this should be an AOS utility since it may be a thing we will want to do often.
    """

    def test_non_positive_num_solutions(self):
        """
        Confirm that an exception is thrown with a non-positive num solutions
        """
        m = tc.get_triangle_ip()
        with self.assertRaises(ValueError):
            gurobi_generate_solutions(m, num_solutions=-1)

    def test_search_mode(self):
        """
        Confirm that an exception is thrown with pool_search_mode not in [1,2]
        """
        m = tc.get_triangle_ip()
        with self.assertRaises(ValueError):
            gurobi_generate_solutions(m, pool_search_mode=0)

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_ip_feasibility(self):
        """
        Enumerate all solutions for an ip: triangle_ip.

        Check that the correct number of alternate solutions are found.
        """
        m = tc.get_triangle_ip()
        results = gurobi_generate_solutions(m, num_solutions=100)
        objectives = [round(soln.objective().value, 2) for soln in results]
        actual_solns_by_obj = m.num_ranked_solns
        unique_solns_by_obj = [val for val in Counter(objectives).values()]
        np.testing.assert_array_almost_equal(unique_solns_by_obj, actual_solns_by_obj)

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_ip_num_solutions(self):
        """
        Enumerate 8 solutions for an ip: triangle_ip.

        Check that the correct number of alternate solutions are found.
        """
        m = tc.get_triangle_ip()
        results = gurobi_generate_solutions(m, num_solutions=8)
        assert len(results) == 8
        objectives = [round(soln.objective().value, 2) for soln in results]
        actual_solns_by_obj = [6, 2]
        unique_solns_by_obj = [val for val in Counter(objectives).values()]
        np.testing.assert_array_almost_equal(unique_solns_by_obj, actual_solns_by_obj)

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_ip_solutions_1d(self):
        """
        Enumerate 8 solutions for an ip: triangle_ip.

        Check that the correct number of alternate solutions are found.
        """
        m = tc.get_1d_problem(discrete_x=True)
        # range is -10 to 10 so max of 21 integer solutions
        desired_sols = 5
        results = gurobi_generate_solutions(m, num_solutions=desired_sols)
        assert len(results) == desired_sols
        objectives = [round(soln.objective().value, 2) for soln in results]
        actual_solns_by_obj = [10, 9, 8, 7, 6]
        np.testing.assert_array_almost_equal(objectives, actual_solns_by_obj)
        for index, solution in enumerate(results.solutions):
            assert len(solution._variables) == 1
            assert solution._variables[0].value == 10 - index

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_mip_feasibility(self):
        """
        Enumerate all solutions for a mip: indexed_pentagonal_pyramid_mip.

        Check that the correct number of alternate solutions are found.
        """
        m = tc.get_indexed_pentagonal_pyramid_mip()
        results = gurobi_generate_solutions(m, num_solutions=100)
        objectives = [round(soln.objective().value, 2) for soln in results]
        actual_solns_by_obj = m.num_ranked_solns
        unique_solns_by_obj = [val for val in Counter(objectives).values()]
        np.testing.assert_array_almost_equal(unique_solns_by_obj, actual_solns_by_obj)

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_mip_rel_feasibility(self):
        """
        Enumerate solutions for a mip: indexed_pentagonal_pyramid_mip.

        Check that only solutions within a relative tolerance of 0.2 are
        found.
        """
        m = tc.get_pentagonal_pyramid_mip()
        results = gurobi_generate_solutions(m, num_solutions=100, rel_opt_gap=0.2)
        objectives = [round(soln.objective().value, 2) for soln in results]
        actual_solns_by_obj = m.num_ranked_solns[0:2]
        unique_solns_by_obj = [val for val in Counter(objectives).values()]
        np.testing.assert_array_almost_equal(unique_solns_by_obj, actual_solns_by_obj)

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_mip_rel_feasibility_options(self):
        """
        Enumerate solutions for a mip: indexed_pentagonal_pyramid_mip.

        Check that only solutions within a relative tolerance of 0.2 are
        found.
        """
        m = tc.get_pentagonal_pyramid_mip()
        results = gurobi_generate_solutions(
            m, num_solutions=100, solver_options={"PoolGap": 0.2}
        )
        objectives = [round(soln.objective().value, 2) for soln in results]
        actual_solns_by_obj = m.num_ranked_solns[0:2]
        unique_solns_by_obj = [val for val in Counter(objectives).values()]
        np.testing.assert_array_almost_equal(unique_solns_by_obj, actual_solns_by_obj)

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_mip_abs_feasibility(self):
        """
        Enumerate solutions for a mip: indexed_pentagonal_pyramid_mip.

        Check that only solutions within an absolute tolerance of 1.99 are
        found.
        """
        m = tc.get_pentagonal_pyramid_mip()
        results = gurobi_generate_solutions(m, num_solutions=100, abs_opt_gap=1.99)
        objectives = [round(soln.objective().value, 2) for soln in results]
        actual_solns_by_obj = m.num_ranked_solns[0:3]
        unique_solns_by_obj = [val for val in Counter(objectives).values()]
        np.testing.assert_array_almost_equal(unique_solns_by_obj, actual_solns_by_obj)

    @unittest.skipIf(True, "Ignoring fragile test for solver timeout.")
    def test_mip_no_time(self):
        """
        Enumerate solutions for a mip: indexed_pentagonal_pyramid_mip.

        Check that no solutions are returned with a timelimit of 0.
        """
        m = tc.get_pentagonal_pyramid_mip()
        # Use quiet=False to test error message
        results = gurobi_generate_solutions(
            m, num_solutions=100, solver_options={"TimeLimit": 0.0}, quiet=False
        )
        assert len(results) == 0

    def test_infeasible_minimization_ip(self):
        """
        Simple AOS test on 2D box example made to be infeasible
        Details in test_case.py for get_trivial_2d_box.
        Minimization case
        """

        m = tc.get_trivial_2d_box_ip(sense=pyo.minimize)
        m.c = pyo.Constraint(expr=m.x + m.y <= -1)
        with self.assertRaises(ApplicationError) as cm:
            sols = gurobi_generate_solutions(m, lower_objective_threshold=3)
        self.assertIn("Model cannot be solved, ", str(cm.exception))

    def test_infeasible_maximization_ip(self):
        """
        Simple AOS test on 2D box example made to be infeasible
        Details in test_case.py for get_trivial_2d_box.
        Maximization case
        """

        m = tc.get_trivial_2d_box_ip(sense=pyo.maximize)
        m.c = pyo.Constraint(expr=m.x + m.y >= 3)
        with self.assertRaises(ApplicationError) as cm:
            sols = gurobi_generate_solutions(m, lower_objective_threshold=3)
        self.assertIn("Model cannot be solved, ", str(cm.exception))

    def test_trivial_2d_box_ip_minimize(self):
        """
        Simple AOS test on 2D box example.
        Details in test_case.py for get_trivial_2d_box.
        Minimization case
        """

        m = tc.get_trivial_2d_box_ip(sense=pyo.minimize)
        sols = gurobi_generate_solutions(m)
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

    def test_trivial_2d_box_ip_maximize(self):
        """
        Simple AOS test on 2D box example.
        Details in test_case.py for get_trivial_2d_box.
        Minimization case
        """

        m = tc.get_trivial_2d_box_ip(sense=pyo.maximize)
        sols = gurobi_generate_solutions(m)
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

    def test_gurobi_generate_solutions_upper_objective_bound(self):
        """
        Simple AOS test on 2D box example using upper objective bound
        Details in test_case.py for get_trivial_2d_box.
        """

        m = tc.get_trivial_2d_box_ip(sense=pyo.minimize)
        # this case should keep all the same solutions
        sols = gurobi_generate_solutions(m, upper_objective_threshold=2)
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
        sols = gurobi_generate_solutions(m, upper_objective_threshold=1)
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
        sols = gurobi_generate_solutions(m, upper_objective_threshold=0)
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
        # should raise ApplicationError
        with self.assertRaises(ApplicationError) as cm:
            sols = gurobi_generate_solutions(m, upper_objective_threshold=-1)
        self.assertIn("Model cannot be solved, ", str(cm.exception))

    def test_gurobi_generate_solutions_lower_objective_bound(self):
        """
        Simple AOS test on 2D box example using lower objective bound
        Details in test_case.py for get_trivial_2d_box.
        """
        m = tc.get_trivial_2d_box_ip(sense=pyo.maximize)
        # this case should keep all the same solutions
        sols = gurobi_generate_solutions(m, lower_objective_threshold=0)
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
        sols = gurobi_generate_solutions(m, lower_objective_threshold=1)
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
        sols = gurobi_generate_solutions(m, lower_objective_threshold=2)
        assert len(sols) == sum(m.num_ranked_solns[0:1])
        sol_list = list()
        for s in sols:
            s_x = s.variable("x")
            s_y = s.variable("y")
            sol_list.append(
                ((int(s_x.value), int(s_y.value)), int(s.objective().value))
            )
        assert m.feasible_sols[0] == sol_list[0]

        with self.assertRaises(ApplicationError) as cm:
            sols = gurobi_generate_solutions(m, lower_objective_threshold=3)
        self.assertIn("Model cannot be solved, ", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
