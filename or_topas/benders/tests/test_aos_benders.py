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
from or_topas.benders.aos_benders import (
    aos_benders_generate_candidates,
    aos_benders_filter,
)
from or_topas.util.mymunch import MyMunch
from or_topas.util.pyomo_utils import pprint_solution
from pyomo.repn.standard_repn import generate_standard_repn

parameterized, param_available = attempt_import("parameterized")
if not param_available:
    raise unittest.SkipTest("Parameterized is not available.")
parameterized = parameterized.parameterized

non_persistent_mip_solvers = list(
    pyomo.opt.check_available_solvers(
        "glpk",
        # "highs", highs not supported for now, disagrees with glpk and gurobi
        "gurobi",
    )
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
persistent_to_non_persistent_solver_map = {"gurobi_persistent": "gurobi"}


ipopt_available = pyo.SolverFactory("ipopt").available(exception_flag=False)
gurobi_available = pyo.SolverFactory("gurobi_persistent").available(
    exception_flag=False
)


@unittest.skipIf(
    not gurobi_available, "gurobi is not available so feasibility cut tests skipped"
)
class TestAOS_Benders_Persistent(unittest.TestCase):
    @parameterized.expand(input=persistent_mip_solvers, skip_on_empty=True)
    def test_farmer_aos_benders_filter_deterministic(self, mip_solver):
        num_solutions = 50
        mode = "d"
        tee = False
        tee_final = False
        rel_gaps = [0, 0.01, 0.5]
        expected_candidate_pool_size = [1, 6, 24]
        expected_true_pool_size = [1, 6, 24]
        mip_solver_non_persistent_version = persistent_to_non_persistent_solver_map[
            mip_solver
        ]
        for index, rel_gap in enumerate(rel_gaps):
            opt, m = tc.Farmer.run_farmer(
                mip_solver,
                mode=mode,
                transform="standard_lp",
                add_upper_bounds=True,
                is_persistent=True,
            )
            skip_vars = None
            candidate_pool, data = aos_benders_generate_candidates(
                m=m,
                rel_gap=rel_gap,
                num_solutions=num_solutions,
                mip_solver=mip_solver_non_persistent_version,
                skip_vars=skip_vars,
                ignore_opt_tol_in_basis=False,
            )
            assert (
                len(candidate_pool) == expected_candidate_pool_size[index]
            ), f"Expected {expected_candidate_pool_size[index]} candidate solutions, got {len(candidate_pool)}"
            true_pool = aos_benders_filter(
                candidate_pool, data, tee=tee, tee_final=tee_final
            )
            assert (
                len(true_pool) == expected_true_pool_size[index]
            ), f"Expected {expected_true_pool_size[index]} true solutions, got {len(true_pool)}"

    @parameterized.expand(input=persistent_mip_solvers, skip_on_empty=True)
    def test_farmer_aos_benders_filter_deterministic_ignore_opt_tol(self, mip_solver):
        num_solutions = 50
        mode = "d"
        tee = False
        tee_final = False
        rel_gaps = [0, 0.01, 0.5]
        expected_candidate_pool_size = [1, 1, 11]
        expected_true_pool_size = [1, 1, 11]
        mip_solver_non_persistent_version = persistent_to_non_persistent_solver_map[
            mip_solver
        ]
        for index, rel_gap in enumerate(rel_gaps):
            opt, m = tc.Farmer.run_farmer(
                mip_solver,
                mode=mode,
                transform="standard_lp",
                add_upper_bounds=True,
                is_persistent=True,
            )
            skip_vars = None
            candidate_pool, data = aos_benders_generate_candidates(
                m=m,
                rel_gap=rel_gap,
                num_solutions=num_solutions,
                mip_solver=mip_solver_non_persistent_version,
                skip_vars=skip_vars,
                ignore_opt_tol_in_basis=True,
            )
            assert (
                len(candidate_pool) == expected_candidate_pool_size[index]
            ), f"Expected {expected_candidate_pool_size[index]} candidate solutions, got {len(candidate_pool)}"
            true_pool = aos_benders_filter(
                candidate_pool, data, tee=tee, tee_final=tee_final
            )
            assert (
                len(true_pool) == expected_true_pool_size[index]
            ), f"Expected {expected_true_pool_size[index]} true solutions, got {len(true_pool)}"

    @parameterized.expand(input=persistent_mip_solvers, skip_on_empty=True)
    def test_farmer_aos_benders_generate_candidates_stochastic_ignore_opt_tol(
        self, mip_solver
    ):
        # TODO: come back to this with a single cut version of AOS Benders is implemented
        num_solutions = 50
        mode = "s"
        tee = False
        tee_final = False
        rel_gaps = [0, 0.01]
        expected_candidate_pool_size = [1, 3]
        mip_solver_non_persistent_version = persistent_to_non_persistent_solver_map[
            mip_solver
        ]
        for index, rel_gap in enumerate(rel_gaps):
            opt, m = tc.Farmer.run_farmer(
                mip_solver,
                mode=mode,
                transform="standard_lp",
                add_upper_bounds=True,
                is_persistent=True,
            )
            skip_vars = None
            candidate_pool, data = aos_benders_generate_candidates(
                m=m,
                rel_gap=rel_gap,
                num_solutions=num_solutions,
                mip_solver=mip_solver_non_persistent_version,
                skip_vars=skip_vars,
                ignore_opt_tol_in_basis=True,
            )
            for sol in candidate_pool:
                pprint_solution(sol)
            assert (
                len(candidate_pool) == expected_candidate_pool_size[index]
            ), f"Expected {expected_candidate_pool_size[index]} candidate solutions, got {len(candidate_pool)}"

    @parameterized.expand(input=persistent_mip_solvers, skip_on_empty=True)
    def test_abs_aos_benders_generate_candidates_single_scenario(self, mip_solver):
        num_solutions = 50
        rel_gaps = [0, 0.01, 0.5]
        solver_name = mip_solver
        expected_candidate_pool_size = [1, 1, 1]
        mip_solver_non_persistent_version = persistent_to_non_persistent_solver_map[
            mip_solver
        ]
        for index, rel_gap in enumerate(rel_gaps):
            opt, m = tc.absolute_value.run_absolute_value(
                mip_solver,
                mode="d",
                transform="standard_lp",
                add_upper_bounds=True,
                include_print=False,
                is_persistent=True,
            )
            m.x.setub(10)
            m.x.setlb(-10)
            skip_vars = None
            candidate_pool, data = aos_benders_generate_candidates(
                m=m,
                rel_gap=rel_gap,
                num_solutions=num_solutions,
                mip_solver=mip_solver_non_persistent_version,
                skip_vars=skip_vars,
                ignore_opt_tol_in_basis=True,
            )
            assert (
                len(candidate_pool) == expected_candidate_pool_size[index]
            ), f"Expected {expected_candidate_pool_size[index]} candidate solutions, got {len(candidate_pool)}"

    @parameterized.expand(input=persistent_mip_solvers, skip_on_empty=True)
    def test_abs_aos_benders_generate_candidates_multiple_scenarios(self, mip_solver):
        num_solutions = 50
        rel_gaps = [0, 0.01, 0.5]
        mip_solver_non_persistent_version = persistent_to_non_persistent_solver_map[
            mip_solver
        ]
        expected_candidate_pool_size = [1, 1, 1]
        for index, rel_gap in enumerate(rel_gaps):
            opt, m = tc.absolute_value.run_absolute_value(
                mip_solver,
                mode="s",
                transform="standard_lp",
                add_upper_bounds=True,
                include_print=False,
                is_persistent=True,
            )
            m.x.setub(10)
            m.x.setlb(-10)
            skip_vars = None
            candidate_pool, data = aos_benders_generate_candidates(
                m=m,
                rel_gap=rel_gap,
                num_solutions=num_solutions,
                mip_solver=mip_solver_non_persistent_version,
                skip_vars=skip_vars,
                ignore_opt_tol_in_basis=True,
            )
            assert (
                len(candidate_pool) == expected_candidate_pool_size[index]
            ), f"Expected {expected_candidate_pool_size[index]} candidate solutions, got {len(candidate_pool)}"

    @parameterized.expand(input=persistent_mip_solvers, skip_on_empty=True)
    def test_abs_aos_benders_filter_single_scenario(self, mip_solver):
        num_solutions = 50
        rel_gaps = [0, 0.01, 0.5]
        mip_solver_non_persistent_version = persistent_to_non_persistent_solver_map[
            mip_solver
        ]
        expected_candidate_pool_size = [1, 1, 1]
        expected_true_pool_size = [1, 1, 1]
        for index, rel_gap in enumerate(rel_gaps):
            opt, m = tc.absolute_value.run_absolute_value(
                mip_solver,
                mode="d",
                transform="standard_lp",
                add_upper_bounds=True,
                include_print=False,
                is_persistent=True,
            )
            m.x.setub(10)
            m.x.setlb(-10)
            skip_vars = None
            candidate_pool, data = aos_benders_generate_candidates(
                m=m,
                rel_gap=rel_gap,
                num_solutions=num_solutions,
                mip_solver=mip_solver_non_persistent_version,
                skip_vars=skip_vars,
                ignore_opt_tol_in_basis=True,
            )
            assert (
                len(candidate_pool) == expected_candidate_pool_size[index]
            ), f"Expected {expected_candidate_pool_size[index]} candidate solutions, got {len(candidate_pool)}"
            true_pool = aos_benders_filter(
                candidate_pool, data, tee=False, tee_final=False
            )
            assert (
                len(true_pool) == expected_true_pool_size[index]
            ), f"Expected {expected_true_pool_size[index]} true solutions, got {len(true_pool)}"

    @parameterized.expand(input=persistent_mip_solvers, skip_on_empty=True)
    def test_abs_aos_benders_filer_multiple_scenarios(self, mip_solver):
        num_solutions = 50
        rel_gaps = [0, 0.01, 0.5]
        mip_solver_non_persistent_version = persistent_to_non_persistent_solver_map[
            mip_solver
        ]
        expected_candidate_pool_size = [1, 1, 1]
        expected_true_pool_size = [1, 1, 1]
        for index, rel_gap in enumerate(rel_gaps):
            opt, m = tc.absolute_value.run_absolute_value(
                mip_solver,
                mode="s",
                transform="standard_lp",
                add_upper_bounds=True,
                include_print=False,
                is_persistent=True,
            )
            m.x.setub(10)
            m.x.setlb(-10)
            skip_vars = None
            candidate_pool, data = aos_benders_generate_candidates(
                m=m,
                rel_gap=rel_gap,
                num_solutions=num_solutions,
                mip_solver=mip_solver_non_persistent_version,
                skip_vars=skip_vars,
                ignore_opt_tol_in_basis=True,
            )
            assert (
                len(candidate_pool) == expected_candidate_pool_size[index]
            ), f"Expected {expected_candidate_pool_size[index]} candidate solutions, got {len(candidate_pool)}"
            true_pool = aos_benders_filter(
                candidate_pool, data, tee=False, tee_final=False
            )
            assert (
                len(true_pool) == expected_true_pool_size[index]
            ), f"Expected {expected_true_pool_size[index]} true solutions, got {len(true_pool)}"


class TestAOS_Benders_Non_Persistent(unittest.TestCase):

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    def test_farmer_end_to_end_deterministic(self, mip_solver):
        num_solutions = 50
        mode = "d"
        tee = False
        tee_final = False
        rel_gaps = [0, 0.01]
        expected_true_pool_size = [1, 6]
        for index, rel_gap in enumerate(rel_gaps):
            true_pool = aos_benders_farmer_test(
                mip_solver=mip_solver,
                num_solutions=num_solutions,
                mode="d",
                tee=False,
                tee_final=False,
                rel_gap=rel_gap,
                use_skip_vars=False,
            )
            assert (
                len(true_pool) == expected_true_pool_size[index]
            ), f"Expected {expected_true_pool_size[index]} true solutions, got {len(true_pool)}"

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    def test_farmer_aos_benders_generate_candidates_deterministic(self, mip_solver):
        num_solutions = 50
        mode = "d"
        tee = False
        tee_final = False
        rel_gaps = [0, 0.01, 0.5]
        expected_candidate_pool_size = [1, 6, 24]
        for index, rel_gap in enumerate(rel_gaps):
            opt, m = tc.Farmer.run_farmer(
                mip_solver, mode=mode, transform="standard_lp", add_upper_bounds=True
            )
            skip_vars = None
            candidate_pool, data = aos_benders_generate_candidates(
                m=m,
                rel_gap=rel_gap,
                num_solutions=num_solutions,
                mip_solver=mip_solver,
                skip_vars=skip_vars,
                ignore_opt_tol_in_basis=False,
            )
            assert (
                len(candidate_pool) == expected_candidate_pool_size[index]
            ), f"Expected {expected_candidate_pool_size[index]} candidate solutions, got {len(candidate_pool)}"

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    def test_farmer_aos_benders_generate_candidates_deterministic_ignore_opt_tol(
        self, mip_solver
    ):
        num_solutions = 50
        mode = "d"
        tee = False
        tee_final = False
        rel_gaps = [0, 0.01, 0.5]
        expected_candidate_pool_size = [1, 1, 11]
        for index, rel_gap in enumerate(rel_gaps):
            opt, m = tc.Farmer.run_farmer(
                mip_solver, mode=mode, transform="standard_lp", add_upper_bounds=True
            )
            skip_vars = None
            candidate_pool, data = aos_benders_generate_candidates(
                m=m,
                rel_gap=rel_gap,
                num_solutions=num_solutions,
                mip_solver=mip_solver,
                skip_vars=skip_vars,
                ignore_opt_tol_in_basis=True,
            )
            assert (
                len(candidate_pool) == expected_candidate_pool_size[index]
            ), f"Expected {expected_candidate_pool_size[index]} candidate solutions, got {len(candidate_pool)}"

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    def test_farmer_aos_benders_generate_candidates_stochastic_ignore_opt_tol(
        self, mip_solver
    ):
        # TODO: come back to this with a single cut version of AOS Benders is implemented
        num_solutions = 50
        mode = "s"
        tee = False
        tee_final = False
        rel_gaps = [0, 0.01]
        expected_candidate_pool_size = [1, 3]
        for index, rel_gap in enumerate(rel_gaps):
            opt, m = tc.Farmer.run_farmer(
                mip_solver, mode=mode, transform="standard_lp", add_upper_bounds=True
            )
            skip_vars = None
            candidate_pool, data = aos_benders_generate_candidates(
                m=m,
                rel_gap=rel_gap,
                num_solutions=num_solutions,
                mip_solver=mip_solver,
                skip_vars=skip_vars,
                ignore_opt_tol_in_basis=True,
            )
            for sol in candidate_pool:
                pprint_solution(sol)
            assert (
                len(candidate_pool) == expected_candidate_pool_size[index]
            ), f"Expected {expected_candidate_pool_size[index]} candidate solutions, got {len(candidate_pool)}"

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    def test_farmer_aos_benders_filter_deterministic(self, mip_solver):
        num_solutions = 50
        mode = "d"
        tee = False
        tee_final = False
        rel_gaps = [0, 0.01, 0.5]
        expected_candidate_pool_size = [1, 6, 24]
        expected_true_pool_size = [1, 6, 24]
        for index, rel_gap in enumerate(rel_gaps):
            opt, m = tc.Farmer.run_farmer(
                mip_solver, mode=mode, transform="standard_lp", add_upper_bounds=True
            )
            skip_vars = None
            candidate_pool, data = aos_benders_generate_candidates(
                m=m,
                rel_gap=rel_gap,
                num_solutions=num_solutions,
                mip_solver=mip_solver,
                skip_vars=skip_vars,
                ignore_opt_tol_in_basis=False,
            )
            assert (
                len(candidate_pool) == expected_candidate_pool_size[index]
            ), f"Expected {expected_candidate_pool_size[index]} candidate solutions, got {len(candidate_pool)}"
            true_pool = aos_benders_filter(
                candidate_pool, data, tee=tee, tee_final=tee_final
            )
            assert (
                len(true_pool) == expected_true_pool_size[index]
            ), f"Expected {expected_true_pool_size[index]} true solutions, got {len(true_pool)}"

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    def test_farmer_aos_benders_filter_deterministic_ignore_opt_tol(self, mip_solver):
        num_solutions = 50
        mode = "d"
        tee = False
        tee_final = False
        rel_gaps = [0, 0.01, 0.5]
        expected_candidate_pool_size = [1, 1, 11]
        expected_true_pool_size = [1, 1, 11]
        for index, rel_gap in enumerate(rel_gaps):
            opt, m = tc.Farmer.run_farmer(
                mip_solver, mode=mode, transform="standard_lp", add_upper_bounds=True
            )
            skip_vars = None
            candidate_pool, data = aos_benders_generate_candidates(
                m=m,
                rel_gap=rel_gap,
                num_solutions=num_solutions,
                mip_solver=mip_solver,
                skip_vars=skip_vars,
                ignore_opt_tol_in_basis=True,
            )
            assert (
                len(candidate_pool) == expected_candidate_pool_size[index]
            ), f"Expected {expected_candidate_pool_size[index]} candidate solutions, got {len(candidate_pool)}"
            true_pool = aos_benders_filter(
                candidate_pool, data, tee=tee, tee_final=tee_final
            )
            assert (
                len(true_pool) == expected_true_pool_size[index]
            ), f"Expected {expected_true_pool_size[index]} true solutions, got {len(true_pool)}"

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    def test_abs_aos_benders_generate_candidates_single_scenario(self, mip_solver):
        num_solutions = 50
        rel_gaps = [0, 0.01, 0.5]
        solver_name = mip_solver
        expected_candidate_pool_size = [1, 1, 1]
        for index, rel_gap in enumerate(rel_gaps):
            opt, m = tc.absolute_value.run_absolute_value(
                mip_solver,
                mode="d",
                transform="standard_lp",
                add_upper_bounds=True,
                include_print=False,
                is_persistent=False,
            )
            m.x.setub(10)
            m.x.setlb(-10)
            skip_vars = None
            candidate_pool, data = aos_benders_generate_candidates(
                m=m,
                rel_gap=rel_gap,
                num_solutions=num_solutions,
                mip_solver=mip_solver,
                skip_vars=skip_vars,
                ignore_opt_tol_in_basis=True,
            )
            assert (
                len(candidate_pool) == expected_candidate_pool_size[index]
            ), f"Expected {expected_candidate_pool_size[index]} candidate solutions, got {len(candidate_pool)}"

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    def test_abs_aos_benders_generate_candidates_multiple_scenarios(self, mip_solver):
        num_solutions = 50
        rel_gaps = [0, 0.01, 0.5]
        solver_name = mip_solver
        expected_candidate_pool_size = [1, 1, 1]
        for index, rel_gap in enumerate(rel_gaps):
            opt, m = tc.absolute_value.run_absolute_value(
                mip_solver,
                mode="s",
                transform="standard_lp",
                add_upper_bounds=True,
                include_print=False,
                is_persistent=False,
            )
            m.x.setub(10)
            m.x.setlb(-10)
            skip_vars = None
            candidate_pool, data = aos_benders_generate_candidates(
                m=m,
                rel_gap=rel_gap,
                num_solutions=num_solutions,
                mip_solver=mip_solver,
                skip_vars=skip_vars,
                ignore_opt_tol_in_basis=True,
            )
            assert (
                len(candidate_pool) == expected_candidate_pool_size[index]
            ), f"Expected {expected_candidate_pool_size[index]} candidate solutions, got {len(candidate_pool)}"

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    def test_abs_aos_benders_filter_single_scenario(self, mip_solver):
        num_solutions = 50
        rel_gaps = [0, 0.01, 0.5]
        solver_name = mip_solver
        expected_candidate_pool_size = [1, 1, 1]
        expected_true_pool_size = [1, 1, 1]
        for index, rel_gap in enumerate(rel_gaps):
            opt, m = tc.absolute_value.run_absolute_value(
                mip_solver,
                mode="d",
                transform="standard_lp",
                add_upper_bounds=True,
                include_print=False,
                is_persistent=False,
            )
            m.x.setub(10)
            m.x.setlb(-10)
            skip_vars = None
            candidate_pool, data = aos_benders_generate_candidates(
                m=m,
                rel_gap=rel_gap,
                num_solutions=num_solutions,
                mip_solver=mip_solver,
                skip_vars=skip_vars,
                ignore_opt_tol_in_basis=True,
            )
            assert (
                len(candidate_pool) == expected_candidate_pool_size[index]
            ), f"Expected {expected_candidate_pool_size[index]} candidate solutions, got {len(candidate_pool)}"
            true_pool = aos_benders_filter(
                candidate_pool, data, tee=False, tee_final=False
            )
            assert (
                len(true_pool) == expected_true_pool_size[index]
            ), f"Expected {expected_true_pool_size[index]} true solutions, got {len(true_pool)}"

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    def test_abs_aos_benders_filer_multiple_scenarios(self, mip_solver):
        num_solutions = 50
        rel_gaps = [0, 0.01, 0.5]
        solver_name = mip_solver
        expected_candidate_pool_size = [1, 1, 1]
        expected_true_pool_size = [1, 1, 1]
        for index, rel_gap in enumerate(rel_gaps):
            opt, m = tc.absolute_value.run_absolute_value(
                mip_solver,
                mode="s",
                transform="standard_lp",
                add_upper_bounds=True,
                include_print=False,
                is_persistent=False,
            )
            m.x.setub(10)
            m.x.setlb(-10)
            skip_vars = None
            candidate_pool, data = aos_benders_generate_candidates(
                m=m,
                rel_gap=rel_gap,
                num_solutions=num_solutions,
                mip_solver=mip_solver,
                skip_vars=skip_vars,
                ignore_opt_tol_in_basis=True,
            )
            assert (
                len(candidate_pool) == expected_candidate_pool_size[index]
            ), f"Expected {expected_candidate_pool_size[index]} candidate solutions, got {len(candidate_pool)}"
            true_pool = aos_benders_filter(
                candidate_pool, data, tee=False, tee_final=False
            )
            assert (
                len(true_pool) == expected_true_pool_size[index]
            ), f"Expected {expected_true_pool_size[index]} true solutions, got {len(true_pool)}"
