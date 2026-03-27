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
import or_topas.benders.tests.test_cases as tc
from or_topas.benders.aos_benders import aos_farmer_test as aos_benders_farmer_test
from or_topas.util.mymunch import MyMunch
from pyomo.repn.standard_repn import generate_standard_repn

parameterized, param_available = attempt_import("parameterized")
if not param_available:
    raise unittest.SkipTest("Parameterized is not available.")
parameterized = parameterized.parameterized

non_persistent_mip_solvers = list(
    pyomo.opt.check_available_solvers("glpk", "highs", "gurobi")
)
infeasibility_test_solvers = list(pyomo.opt.check_available_solvers("glpk", "gurobi"))

infeasibility_persistent_test_solvers = list(
    pyomo.opt.check_available_solvers(
        # "appsi_gurobi", #TODO: update to allow appsi_gurobi to work
        "gurobi_persistent",
    )
)

persistent_mip_solvers = list(
    pyomo.opt.check_available_solvers(
        # "appsi_highs",
        # "appsi_gurobi",
        "gurobi_persistent",
    )
)


ipopt_available = pyo.SolverFactory("ipopt").available(exception_flag=False)
gurobi_available = pyo.SolverFactory("gurobi_persistent").available(
    exception_flag=False
)


@unittest.skipIf(
    not gurobi_available, "gurobi is not available so feasibility cut tests skipped"
)
class TestAOS_Benders_Feasibility(unittest.TestCase):
    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    def test_farmer(self, mip_solver):

        t = True
        assert t, "Trivial Test"


class TestAOS_Benders_Optimality(unittest.TestCase):
    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    def test_farmer(self, mip_solver):

        t = True
        assert t, "Trivial Test"

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    def test_farmer(self, mip_solver):

        aos_benders_farmer_test(
            mip_solver=mip_solver,
            num_solutions=10,
            mode="s",
            tee=False,
            tee_final=False,
            rel_gap=0.01,
            use_skip_vars=False,
        )
