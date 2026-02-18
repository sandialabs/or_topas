
from pyomo.common.dependencies import numpy as numpy, numpy_available

import pyomo.environ as pyo
import pyomo.opt
from pyomo.common import unittest
from pyomo.common.dependencies import attempt_import, numpy as np, numpy_available

parameterized, param_available = attempt_import("parameterized")
if not param_available:
    raise unittest.SkipTest("Parameterized is not available.")
parameterized = parameterized.parameterized

import or_topas.aos.tests.test_cases as tc
from or_topas.aos import enumerate_mixed_integer_linear_solutions, enumerate_linear_solutions
from or_topas.util import pyomo_utils
import warnings
from collections import Counter
#
# Find available solvers. Just use GLPK if it's available.
#
solvers = list(pyomo.opt.check_available_solvers("glpk", "gurobi"))

timelimit = {"gurobi": "TimeLimit", "appsi_gurobi": "TimeLimit", "glpk": "tmlim"}


class TestMILPEnum(unittest.TestCase):

    #test LP
    #test BP
    #test IP
    #test mixed binary linear program
    #test milp

    #test call twice

    # @parameterized.expand(input=solvers)
    # def test_3d_polyhedron(self, mip_solver):
    #     m = tc.get_3d_polyhedron_problem()
    #     m.o.deactivate()
    #     m.obj = pyo.Objective(expr=m.x[0] + 2 * m.x[1] + 3 * m.x[2])

    #     sols = enumerate_linear_solutions(m, solver=mip_solver)
    #     assert len(sols) == 2
    #     for s in sols:
    #         assert s.objective().value == unittest.pytest.approx(
    #             9
    #         ) or s.objective().value == unittest.pytest.approx(10)

    @parameterized.expand(input=solvers)
    def test_trivial(self, mip_solver):
        t = False
        assert t == False, 'This should work'
        print('Hi')

    # @parameterized.expand(input=solvers)
    # def test_triangle_lp(self, mip_solver):
    #     """
    #     Test that the MILP AOS method can handle  
    #     """
    #     for level in range(0, 6):
    #         abs_tol = 5 - level
    #         m = tc.get_triangle_lp(level=level)
    #         sols = lp_enum.enumerate_linear_solutions(
    #             m, solver=mip_solver, abs_opt_gap=abs_tol
    #         )
    #         assert len(sols) == sum(m.num_ranked_solns)
    #         sol_set = set()
    #         for s in sols:
    #             s_x = s.variable("x")
    #             s_y = s.variable("y")
    #             sol_set.add(
    #                 ((int(s_x.value), int(s_y.value)), int(s.objective().value))
    #             )
    #         assert set(m.feasible_sols) == sol_set
    # @unittest.skipIf(not numpy_available, "Numpy not installed")
    # def test_ip_num_solutions(self):
    #     """
    #     Enumerate 8 solutions for an ip: triangle_ip.

    #     Check that the correct number of alternate solutions are found.
    #     """
    #     m = tc.get_triangle_ip()
    #     results = gurobi_generate_solutions(m, num_solutions=8)
    #     assert len(results) == 8
    #     objectives = [round(soln.objective().value, 2) for soln in results]
    #     actual_solns_by_obj = [6, 2]
    #     unique_solns_by_obj = [val for val in Counter(objectives).values()]
    #     np.testing.assert_array_almost_equal(unique_solns_by_obj, actual_solns_by_obj)


    # @parameterized.expand(input=solvers)
    # def test_3d_polyhedron_called_twice(self, mip_solver):
    #     """
    #     Test that AOS method can be called twice in a row with no issues
    #     Also checks that objective results are the same across solves
    #     """
    #     m = tc.get_3d_polyhedron_problem()
    #     m.o.deactivate()
    #     m.obj = pyo.Objective(expr=m.x[0] + m.x[1] + m.x[2])

    #     all_variables_before_solve = pyomo_utils.get_model_variables(m)
    #     sols = lp_enum.enumerate_linear_solutions(m, solver=mip_solver)
    #     all_variables_after_solve = pyomo_utils.get_model_variables(m)
    #     assert len(sols) == 2
    #     for s in sols:
    #         assert s.objective().value == unittest.pytest.approx(4)
    #     assert len(all_variables_before_solve) == len(all_variables_after_solve)

    #     sols = lp_enum.enumerate_linear_solutions(m, solver=mip_solver)
    #     assert len(sols) == 2
    #     for s in sols:
    #         assert s.objective().value == unittest.pytest.approx(4)

    # @parameterized.expand(input=solvers)
    # def test_3d_polyhedron(self, mip_solver):
    #     m = tc.get_3d_polyhedron_problem()
    #     m.o.deactivate()
    #     m.obj = pyo.Objective(expr=m.x[0] + 2 * m.x[1] + 3 * m.x[2])

    #     sols = lp_enum.enumerate_linear_solutions(m, solver=mip_solver)
    #     assert len(sols) == 2
    #     for s in sols:
    #         assert s.objective().value == unittest.pytest.approx(
    #             9
    #         ) or s.objective().value == unittest.pytest.approx(10)

    # @parameterized.expand(input=solvers)
    # def test_triangle_lp(self, mip_solver):
    #     """
    #     Test that AOS method can be called multiple times in a row.
    #     Uses adaptive test from test cases.
    #     Feasible region is a right triangle with vertices (x,y) = (5,0), (0,5), (0,0)
    #     Objective is x+y
    #     Runs repeatedly changing the absolute gap tol at 0,1,2,3,4,5
    #     Checks that vertices found are the expected ones.
    #     Details in test_case.py
    #     """
    #     for level in range(0, 6):
    #         abs_tol = 5 - level
    #         m = tc.get_triangle_lp(level=level)
    #         sols = lp_enum.enumerate_linear_solutions(
    #             m, solver=mip_solver, abs_opt_gap=abs_tol
    #         )
    #         assert len(sols) == sum(m.num_ranked_solns)
    #         sol_set = set()
    #         for s in sols:
    #             s_x = s.variable("x")
    #             s_y = s.variable("y")
    #             sol_set.add(
    #                 ((int(s_x.value), int(s_y.value)), int(s.objective().value))
    #             )
    #         assert set(m.feasible_sols) == sol_set

    # @parameterized.expand(input=solvers)
    # def test_triangle_milp_fix_integer(self, mip_solver):
    #     """
    #     Test that AOS method can be called multiple times in a row and handle all integers fixed
    #     All integer fixed converts the MILP to effectively an LP
    #     Uses adaptive test from test cases.
    #     Feasible region is a right triangle with vertices (x,y) = (5,0), (0,5), (0,0)
    #     Objective is x+y
    #     Runs repeatedly changing the absolute gap tol at 0,1,2,3,4,5
    #     Checks that vertices found are the expected ones.
    #     Details in test_case.py
    #     """
    #     for level in range(0, 6):
    #         abs_tol = 5 - level
    #         m = tc.get_triangle_lp(level=level)
    #         sols = lp_enum.enumerate_linear_solutions(
    #             m, solver=mip_solver, abs_opt_gap=abs_tol
    #         )
    #         assert len(sols) == sum(m.num_ranked_solns)
    #         sol_set = set()
    #         for s in sols:
    #             s_x = s.variable("x")
    #             s_y = s.variable("y")
    #             sol_set.add(
    #                 ((int(s_x.value), int(s_y.value)), int(s.objective().value))
    #             )
    #         assert set(m.feasible_sols) == sol_set

    # @parameterized.expand(input=solvers)
    # def test_trivial_2d_box_lp_minimize(self, mip_solver):
    #     """
    #     Simple AOS test on 2D box example.
    #     Details in test_case.py for get_trivial_2d_box.
    #     Minimization case
    #     """

    #     m = tc.get_trivial_2d_box_lp(sense=pyo.minimize)
    #     sols = lp_enum.enumerate_linear_solutions(m, solver=mip_solver)
    #     assert len(sols) == sum(m.num_ranked_solns)
    #     sol_list = list()
    #     for s in sols:
    #         s_x = s.variable("x")
    #         s_y = s.variable("y")
    #         sol_list.append(
    #             ((int(s_x.value), int(s_y.value)), int(s.objective().value))
    #         )
    #     assert set(m.feasible_sols) == set(sol_list)
    #     assert m.feasible_sols[0] == sol_list[0]

if __name__ == "__main__":
    unittest.main()
