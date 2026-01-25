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

        # TODO - Confirm how solnpools deal with duplicate solutions
        if gurobi_available:
            assert len(sols) == 7
        else:
            assert len(sols) == 0

    def test_generation_with_variable_count_check(self):
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
        print(f"{all_variables_before_solve_names=}")
        print(f"{all_variables_after_solve_names=}")

        # TODO - Confirm how solnpools deal with duplicate solutions
        if gurobi_available:
            assert len(sols) == 7
        else:
            assert len(sols) == 0
        assert len(all_variables_before_solve) == len(all_variables_after_solve)

    def test_generation_twice(self):
        n = tc.get_pentagonal_pyramid_mip()
        n.x.domain = pyo.Reals
        n.y.domain = pyo.Reals

        sols = gurobi_enumerate_linear_solutions(n, tee=True)

        # TODO - Confirm how solnpools deal with duplicate solutions
        if gurobi_available:
            assert len(sols) == 7
        else:
            assert len(sols) == 0

        sols = gurobi_enumerate_linear_solutions(n, tee=True)

        # TODO - Confirm how solnpools deal with duplicate solutions
        if gurobi_available:
            assert len(sols) == 7
        else:
            assert len(sols) == 0
