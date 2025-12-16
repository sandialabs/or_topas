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

import pyomo.environ as pyo
import pyomo.opt
from pyomo.common import unittest
from pyomo.common.dependencies import numpy as numpy, numpy_available
from pyomo.common.dependencies import attempt_import

if numpy_available:
    from numpy.testing import assert_array_almost_equal

parameterized, param_available = attempt_import("parameterized")
if not param_available:
    raise unittest.SkipTest("Parameterized is not available.")
parameterized = parameterized.parameterized

import or_topas.aos.tests.test_cases as tc
from or_topas import shifted_lp

#
# Find available solvers
#
solvers = list(pyomo.opt.check_available_solvers("glpk", "gurobi"))


# TODO: add checks that confirm the shifted constraints make sense
class TestShiftedIP(unittest.TestCase):

    @parameterized.expand(input=solvers)
    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_mip_abs_objective(self, lp_solver):
        m = tc.get_indexed_pentagonal_pyramid_mip()
        m.x.domain = pyo.Reals

        opt = pyo.SolverFactory(lp_solver)
        old_results = opt.solve(m, tee=False)
        old_obj = pyo.value(m.o)

        new_model = shifted_lp.get_shifted_linear_model(m)
        new_results = opt.solve(new_model, tee=False)
        new_obj = pyo.value(new_model.objective)

        assert old_obj == unittest.pytest.approx(new_obj)

    @parameterized.expand(input=solvers)
    def test_polyhedron(self, lp_solver):
        m = tc.get_3d_polyhedron_problem()

        opt = pyo.SolverFactory(lp_solver)
        old_results = opt.solve(m, tee=False)
        old_obj = pyo.value(m.o)

        new_model = shifted_lp.get_shifted_linear_model(m)
        new_results = opt.solve(new_model, tee=False)
        new_obj = pyo.value(new_model.objective)

        assert old_obj == unittest.pytest.approx(new_obj)


if __name__ == "__main__":
    unittest.main()
