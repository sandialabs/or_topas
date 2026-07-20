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
        tee = True
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
        tee = True
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
                tee=True,
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
        tee = True
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

    @parameterized.expand(input=persistent_mip_solvers, skip_on_empty=True)
    def test_abs_aos_benders_filter_multiple_scenarios_with_obj_offset(
        self, mip_solver
    ):
        num_solutions = 50
        rel_gaps = [0, 0.01, 0.5]
        solver_name = mip_solver
        expected_candidate_pool_size = [1, 5, 5]
        expected_true_pool_size = [1, 5, 5]
        mip_solver_non_persistent_version = persistent_to_non_persistent_solver_map[
            mip_solver
        ]
        for index, rel_gap in enumerate(rel_gaps):
            opt, m = tc.absolute_value.run_absolute_value(
                mip_solver,
                mode="s",
                transform="standard_lp",
                add_upper_bounds=True,
                include_print=False,
                is_persistent=True,
                obj_offset=1,
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
                ignore_opt_tol_in_basis=False,
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
    def test_modified_abs_aos_benders_filter_multiple_scenarios_with_obj_offset(
        self, mip_solver
    ):
        num_solutions = 50
        rel_gaps = [0, 0.01, 0.5, 5, 10, 20]
        solver_name = mip_solver
        # TODO: note that some of these are weird because this is a keep all pool structure not a keep unique within tolerance
        expected_candidate_pool_size = [1, 5, 5, 5, 5, 7]
        expected_true_pool_size = [1, 5, 5, 5, 5, 6]
        mip_solver_non_persistent_version = persistent_to_non_persistent_solver_map[
            mip_solver
        ]
        data = MyMunch(a=0, L=1, R=1, LB=-5, UB=5)
        print()
        for index, rel_gap in enumerate(rel_gaps):
            print(f"{index=}, {rel_gap=}")
            opt, m = tc.modified_absolute_value.run_modified_absolute_value(
                mip_solver,
                mode="s",
                transform="standard_lp",
                add_upper_bounds=True,
                include_print=False,
                is_persistent=True,
                data=None,
                obj_offset=1,
            )
            m.x.setub(10)
            m.x.setlb(-10)
            unbounded_vars = [
                v.name
                for v in m.component_data_objects(
                    pyo.Var, descend_into=True, active=True
                )
                if (v.has_lb() == False or v.has_ub() == False)
            ]
            assert (
                len(unbounded_vars) == 0
            ), f"Needed all vars bounded, got unbounded {unbounded_vars=}"
            skip_vars = None
            candidate_pool, data = aos_benders_generate_candidates(
                m=m,
                rel_gap=rel_gap,
                num_solutions=num_solutions,
                mip_solver=mip_solver_non_persistent_version,
                skip_vars=skip_vars,
                ignore_opt_tol_in_basis=False,
            )
            print("Candidates")
            for sol in candidate_pool:
                pprint_solution(sol)
            assert (
                len(candidate_pool) == expected_candidate_pool_size[index]
            ), f"Expected {expected_candidate_pool_size[index]} candidate solutions, got {len(candidate_pool)}"
            true_pool = aos_benders_filter(
                candidate_pool, data, tee=False, tee_final=False
            )
            print("True")
            for sol in true_pool:
                pprint_solution(sol)
            assert (
                len(true_pool) == expected_true_pool_size[index]
            ), f"Expected {expected_true_pool_size[index]} true solutions, got {len(true_pool)}"

    @parameterized.expand(input=persistent_mip_solvers, skip_on_empty=True)
    def test_modified_abs_aos_benders_filter_multiple_scenarios_with_obj_offset_ignoring_opt_tol(
        self, mip_solver
    ):
        num_solutions = 50
        rel_gaps = [0, 0.01, 0.5, 5, 10, 20]
        solver_name = mip_solver
        # TODO: note that some of these are weird because this is a keep all pool structure not a keep unique within tolerance
        expected_candidate_pool_size = [1, 1, 1, 1, 2, 3]
        expected_true_pool_size = [1, 1, 1, 1, 2, 2]
        mip_solver_non_persistent_version = persistent_to_non_persistent_solver_map[
            mip_solver
        ]
        data = MyMunch(a=0, L=1, R=1, LB=-5, UB=5)
        print()
        for index, rel_gap in enumerate(rel_gaps):
            print(f"{index=}, {rel_gap=}")
            opt, m = tc.modified_absolute_value.run_modified_absolute_value(
                mip_solver,
                mode="s",
                transform="standard_lp",
                add_upper_bounds=True,
                include_print=False,
                is_persistent=True,
                data=None,
                obj_offset=1,
            )
            m.x.setub(10)
            m.x.setlb(-10)
            unbounded_vars = [
                v.name
                for v in m.component_data_objects(
                    pyo.Var, descend_into=True, active=True
                )
                if (v.has_lb() == False or v.has_ub() == False)
            ]
            assert (
                len(unbounded_vars) == 0
            ), f"Needed all vars bounded, got unbounded {unbounded_vars=}"
            skip_vars = None
            candidate_pool, data = aos_benders_generate_candidates(
                m=m,
                rel_gap=rel_gap,
                num_solutions=num_solutions,
                mip_solver=mip_solver_non_persistent_version,
                skip_vars=skip_vars,
                ignore_opt_tol_in_basis=True,
            )
            print("Candidates")
            for sol in candidate_pool:
                pprint_solution(sol)
            assert (
                len(candidate_pool) == expected_candidate_pool_size[index]
            ), f"Expected {expected_candidate_pool_size[index]} candidate solutions, got {len(candidate_pool)}"
            true_pool = aos_benders_filter(
                candidate_pool, data, tee=False, tee_final=False
            )
            print("True")
            for sol in true_pool:
                pprint_solution(sol)
            assert (
                len(true_pool) == expected_true_pool_size[index]
            ), f"Expected {expected_true_pool_size[index]} true solutions, got {len(true_pool)}"

    @parameterized.expand(input=persistent_mip_solvers, skip_on_empty=True)
    def test_modified_abs_aos_benders_filter(self, mip_solver):
        num_solutions = 50
        rel_gaps = [0, 2, 5, 10]
        solver_name = mip_solver
        expected_candidate_pool_size = [1, 3, 3, 4]
        expected_true_pool_size = [1, 3, 3, 3]
        mip_solver_non_persistent_version = persistent_to_non_persistent_solver_map[
            mip_solver
        ]
        data = MyMunch(a=0, L=1, R=1, LB=-5, UB=5)
        for index, rel_gap in enumerate(rel_gaps):
            print(f"{index=}, {rel_gap=}")
            opt, m = tc.modified_absolute_value.run_modified_absolute_value(
                mip_solver,
                mode="d",
                transform="standard_lp",
                add_upper_bounds=True,
                include_print=False,
                is_persistent=True,
                data=None,
                obj_offset=1,
            )
            m.x.setub(10)
            m.x.setlb(-10)
            unbounded_vars = [
                v.name
                for v in m.component_data_objects(
                    pyo.Var, descend_into=True, active=True
                )
                if (v.has_lb() == False or v.has_ub() == False)
            ]
            assert (
                len(unbounded_vars) == 0
            ), f"Needed all vars bounded, got unbounded {unbounded_vars=}"
            skip_vars = None
            candidate_pool, data = aos_benders_generate_candidates(
                m=m,
                rel_gap=rel_gap,
                num_solutions=num_solutions,
                mip_solver=mip_solver_non_persistent_version,
                skip_vars=skip_vars,
                ignore_opt_tol_in_basis=False,
            )

            print("Candidates")
            for sol in candidate_pool:
                pprint_solution(sol)
            assert (
                len(candidate_pool) == expected_candidate_pool_size[index]
            ), f"Expected {expected_candidate_pool_size[index]} candidate solutions, got {len(candidate_pool)}"
            true_pool = aos_benders_filter(
                candidate_pool, data, tee=False, tee_final=False
            )
            print("True")
            for sol in true_pool:
                pprint_solution(sol)
            assert (
                len(true_pool) == expected_true_pool_size[index]
            ), f"Expected {expected_true_pool_size[index]} true solutions, got {len(true_pool)}"

    @parameterized.expand(input=persistent_mip_solvers, skip_on_empty=True)
    def test_modified_abs_aos_benders_filter_ignore_opt_tol(self, mip_solver):
        num_solutions = 50
        rel_gaps = [0, 2, 5, 10]
        solver_name = mip_solver
        expected_candidate_pool_size = [1, 1, 2, 3]
        expected_true_pool_size = [1, 1, 2, 2]
        mip_solver_non_persistent_version = persistent_to_non_persistent_solver_map[
            mip_solver
        ]
        data = MyMunch(a=0, L=1, R=1, LB=-5, UB=5)
        for index, rel_gap in enumerate(rel_gaps):
            print(f"{index=}, {rel_gap=}")
            opt, m = tc.modified_absolute_value.run_modified_absolute_value(
                mip_solver,
                mode="d",
                transform="standard_lp",
                add_upper_bounds=True,
                include_print=False,
                is_persistent=True,
                data=None,
                obj_offset=1,
            )
            m.x.setub(10)
            m.x.setlb(-10)
            unbounded_vars = [
                v.name
                for v in m.component_data_objects(
                    pyo.Var, descend_into=True, active=True
                )
                if (v.has_lb() == False or v.has_ub() == False)
            ]
            assert (
                len(unbounded_vars) == 0
            ), f"Needed all vars bounded, got unbounded {unbounded_vars=}"
            skip_vars = None
            candidate_pool, data = aos_benders_generate_candidates(
                m=m,
                rel_gap=rel_gap,
                num_solutions=num_solutions,
                mip_solver=mip_solver_non_persistent_version,
                skip_vars=skip_vars,
                ignore_opt_tol_in_basis=True,
            )

            # print("Candidates")
            # for sol in candidate_pool:
            #     pprint_solution(sol)
            assert (
                len(candidate_pool) == expected_candidate_pool_size[index]
            ), f"Expected {expected_candidate_pool_size[index]} candidate solutions, got {len(candidate_pool)}"
            true_pool = aos_benders_filter(
                candidate_pool, data, tee=False, tee_final=False
            )
            # print("True")
            # for sol in true_pool:
            #     pprint_solution(sol)
            assert (
                len(true_pool) == expected_true_pool_size[index]
            ), f"Expected {expected_true_pool_size[index]} true solutions, got {len(true_pool)}"

    @parameterized.expand(input=persistent_mip_solvers, skip_on_empty=True)
    def test_grid_aos_benders_filter(self, mip_solver):
        num_solutions = 50
        rel_gaps = [0, 0.01, 0.1]
        solver_name = mip_solver
        expected_candidate_pool_size = [2, 4, 4]
        expected_true_pool_size = [2, 2, 2]
        mip_solver_non_persistent_version = persistent_to_non_persistent_solver_map[
            mip_solver
        ]
        for index, rel_gap in enumerate(rel_gaps):
            print(f"{index=}, {rel_gap=}")
            opt, m = tc.EnergyGrid.run_energy_grid(
                mip_solver,
                transform="standard_lp",
                add_upper_bounds=True,
                include_print=True,
                is_persistent=True,
                grid=None,
            )
            unbounded_vars = [
                v.name
                for v in m.component_data_objects(
                    pyo.Var, descend_into=True, active=True
                )
                if (v.has_lb() == False or v.has_ub() == False)
            ]
            assert (
                len(unbounded_vars) == 0
            ), f"Needed all vars bounded, got unbounded {unbounded_vars=}"
            skip_vars = None
            candidate_pool, data = aos_benders_generate_candidates(
                m=m,
                rel_gap=rel_gap,
                num_solutions=num_solutions,
                mip_solver=mip_solver_non_persistent_version,
                skip_vars=skip_vars,
                ignore_opt_tol_in_basis=False,
            )

            # print("Candidates")
            # for sol in candidate_pool:
            #     pprint_solution(sol)
            assert (
                len(candidate_pool) == expected_candidate_pool_size[index]
            ), f"Expected {expected_candidate_pool_size[index]} candidate solutions, got {len(candidate_pool)}"
            true_pool = aos_benders_filter(
                candidate_pool, data, tee=False, tee_final=False
            )
            # print("True")
            # for sol in true_pool:
            #     pprint_solution(sol)
            assert (
                len(true_pool) == expected_true_pool_size[index]
            ), f"Expected {expected_true_pool_size[index]} true solutions, got {len(true_pool)}"

    @parameterized.expand(input=persistent_mip_solvers, skip_on_empty=True)
    def test_grid_aos_benders_filter_feasibility_only(self, mip_solver):
        num_solutions = 50
        rel_gaps = [0, 0.01, 0.1]
        solver_name = mip_solver
        expected_candidate_pool_size = [4, 8, 8]
        expected_true_pool_size = [4, 4, 4]
        mip_solver_non_persistent_version = persistent_to_non_persistent_solver_map[
            mip_solver
        ]
        for index, rel_gap in enumerate(rel_gaps):
            print(f"{index=}, {rel_gap=}")
            opt, m = tc.EnergyGrid.run_energy_grid(
                mip_solver,
                transform="standard_lp",
                add_upper_bounds=True,
                include_print=True,
                is_persistent=True,
                grid=None,
                feasibility_only=True,
            )
            unbounded_vars = [
                v.name
                for v in m.component_data_objects(
                    pyo.Var, descend_into=True, active=True
                )
                if (v.has_lb() == False or v.has_ub() == False)
            ]
            assert (
                len(unbounded_vars) == 0
            ), f"Needed all vars bounded, got unbounded {unbounded_vars=}"
            skip_vars = None
            candidate_pool, data = aos_benders_generate_candidates(
                m=m,
                rel_gap=rel_gap,
                num_solutions=num_solutions,
                mip_solver=mip_solver_non_persistent_version,
                skip_vars=skip_vars,
                ignore_opt_tol_in_basis=False,
            )

            # print("Candidates")
            # for sol in candidate_pool:
            #     pprint_solution(sol)
            assert (
                len(candidate_pool) == expected_candidate_pool_size[index]
            ), f"Expected {expected_candidate_pool_size[index]} candidate solutions, got {len(candidate_pool)}"
            true_pool = aos_benders_filter(
                candidate_pool, data, tee=False, tee_final=False
            )
            # print("True")
            # for sol in true_pool:
            #     pprint_solution(sol)
            assert (
                len(true_pool) == expected_true_pool_size[index]
            ), f"Expected {expected_true_pool_size[index]} true solutions, got {len(true_pool)}"

    # ------------------------------------------------------------------
    # AOS-Benders tests for the 3-bus commitment OPF, Grok 4.20 generated
    # ------------------------------------------------------------------

    @parameterized.expand(input=persistent_mip_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_commitment_3bus_aos_benders_binary_relgap0(self, mip_solver):
        """AOS-Benders (binary enumeration) on the 3-bus commitment model.
        At rel_gap=0 the true pool must contain exactly the three optimal
        commitment patterns (1,0), (0,1), (1,1)."""
        eps = 1.0
        mode = 2
        rel_gap = 0.0
        num_solutions = 20
        expected_patterns = {(1, 0), (0, 1), (1, 1)}

        grid = tc.EnergyGridWithCommitment(epsilon=eps)

        # Build and solve the commitment Benders model to optimality
        if hasattr(
            tc.EnergyGridWithCommitment, "setup_energy_grid_commitment_persistent"
        ):
            opt, m = (
                tc.EnergyGridWithCommitment.setup_energy_grid_commitment_persistent(
                    solver_name=mip_solver,
                    grid=grid,
                    eta_lb=-1e5,
                    eta_ub=1e5,
                    mode=mode,
                )
            )
        else:
            # Hand-rolled fallback that matches the continuous tests
            m = tc.EnergyGridWithCommitment.create_root(grid, eta_lb=-1e5, eta_ub=1e5)
            root_vars = list(m.commit.values())
            m.benders = BendersCutGenerator()
            m.benders.set_input(
                root_vars=root_vars,
                tol=1e-8,
                transform="standard_lp",
                allow_infeasible=True,
            )
            m.benders.add_subproblem(
                subproblem_fn=tc.EnergyGridWithCommitment.create_subproblem,
                subproblem_fn_kwargs={"root": m, "grid": grid, "mode": mode},
                root_eta=m.eta,
                subproblem_solver=mip_solver,
            )
            opt = pyo.SolverFactory(mip_solver)
            opt.set_instance(m)

        # Short Benders loop to optimality
        for i in range(20):
            opt.solve(tee=False, save_results=False)
            cuts = m.benders.generate_cut()
            for c in cuts:
                opt.add_constraint(c)
            if not cuts:
                break
        self.assertLess(i, 20, "Benders should converge well before 20 iterations")

        # AOS pass – binary enumeration
        mip_solver_non_persistent = persistent_to_non_persistent_solver_map.get(
            mip_solver, "gurobi"
        )
        candidate_pool, data = aos_benders_generate_candidates(
            m=m,
            rel_gap=rel_gap,
            num_solutions=num_solutions,
            mip_solver=mip_solver_non_persistent,
            enumeration_method="binary",
            tee=False,
        )
        true_pool = aos_benders_filter(candidate_pool, data, tee=False, tee_final=False)

        # Extract the commitment vectors from every solution that survived filtering
        recovered = set()
        for sol in true_pool:
            sol.load_into_model(
                model=m,
                value_overrides=None,
                descend_into=True,
                skip_nan_inf=True,
                error_if_value_missing=False,
                track_missing=True,
                track_fixed=True,
                track_unfixed=True,
                track_nan_inf=True,
                unfix_by_default=False,
                fix_continuous=False,
                fix_binary=False,
                fix_integer=False,
                fix_if_sol_var_fixed=False,
                fix_var_names=None,
            )
            u1 = int(round(pyo.value(m.commit["bus1"])))
            u2 = int(round(pyo.value(m.commit["bus2"])))
            recovered.add((u1, u2))

        self.assertEqual(
            recovered,
            expected_patterns,
            f"Expected optimal commitment patterns {expected_patterns}, "
            f"got {recovered}",
        )
        self.assertEqual(
            len(true_pool),
            3,
            f"Expected true pool of size 3, got {len(true_pool)}",
        )

    @parameterized.expand(input=persistent_mip_solvers, skip_on_empty=True)
    @unittest.skipIf(not numpy_available, "numpy is not available.")
    def test_commitment_3bus_aos_benders_binary_relgap_positive(self, mip_solver):
        """With a modest positive relative gap the three optimal patterns
        must still be present in the true pool."""
        eps = 1.0
        mode = 2
        rel_gap = 0.01
        num_solutions = 20
        expected_patterns = {(1, 0), (0, 1), (1, 1)}

        grid = tc.EnergyGridWithCommitment(epsilon=eps)

        if hasattr(
            tc.EnergyGridWithCommitment, "setup_energy_grid_commitment_persistent"
        ):
            opt, m = (
                tc.EnergyGridWithCommitment.setup_energy_grid_commitment_persistent(
                    solver_name=mip_solver,
                    grid=grid,
                    eta_lb=-1e5,
                    eta_ub=1e5,
                    mode=mode,
                )
            )
        else:
            m = tc.EnergyGridWithCommitment.create_root(grid, eta_lb=-1e5, eta_ub=1e5)
            root_vars = list(m.commit.values())
            m.benders = BendersCutGenerator()
            m.benders.set_input(
                root_vars=root_vars,
                tol=1e-8,
                transform="standard_lp",
                allow_infeasible=True,
            )
            m.benders.add_subproblem(
                subproblem_fn=tc.EnergyGridWithCommitment.create_subproblem,
                subproblem_fn_kwargs={"root": m, "grid": grid, "mode": mode},
                root_eta=m.eta,
                subproblem_solver=mip_solver,
            )
            opt = pyo.SolverFactory(mip_solver)
            opt.set_instance(m)

        for i in range(20):
            opt.solve(tee=False, save_results=False)
            cuts = m.benders.generate_cut()
            for c in cuts:
                opt.add_constraint(c)
            if not cuts:
                break

        mip_solver_non_persistent = persistent_to_non_persistent_solver_map.get(
            mip_solver, "gurobi"
        )
        candidate_pool, data = aos_benders_generate_candidates(
            m=m,
            rel_gap=rel_gap,
            num_solutions=num_solutions,
            mip_solver=mip_solver_non_persistent,
            enumeration_method="binary",
            tee=False,
        )
        true_pool = aos_benders_filter(candidate_pool, data, tee=False, tee_final=False)

        recovered = set()
        for sol in true_pool:
            sol.load_into_model(
                model=m,
                value_overrides=None,
                descend_into=True,
                skip_nan_inf=True,
                error_if_value_missing=False,
                track_missing=True,
                track_fixed=True,
                track_unfixed=True,
                track_nan_inf=True,
                unfix_by_default=False,
                fix_continuous=False,
                fix_binary=False,
                fix_integer=False,
                fix_if_sol_var_fixed=False,
                fix_var_names=None,
            )
            u1 = int(round(pyo.value(m.commit["bus1"])))
            u2 = int(round(pyo.value(m.commit["bus2"])))
            recovered.add((u1, u2))

        # All three optimal patterns must still be present
        self.assertTrue(
            expected_patterns.issubset(recovered),
            f"True pool missing some optimal patterns. "
            f"Expected at least {expected_patterns}, recovered {recovered}",
        )

    # ------------------------------------------------------------------
    # AOS-Benders tests for the 3-bus commitment OPF – Gurobi Solution Pool
    # ------------------------------------------------------------------

    @unittest.skipIf(not numpy_available, "numpy is not available.")
    @unittest.skipIf(
        not gurobi_available, "Gurobi is required for enumeration_method='gurobi_pool'"
    )
    def test_commitment_3bus_aos_benders_gurobi_pool_relgap0(self):
        """AOS-Benders (Gurobi Solution Pool) on the 3-bus commitment model.
        At rel_gap=0 the true pool must contain exactly the three optimal
        commitment patterns (1,0), (0,1), (1,1)."""
        eps = 1.0
        mode = 2
        rel_gap = 0.0
        num_solutions = 20
        expected_patterns = {(1, 0), (0, 1), (1, 1)}
        mip_solver = "gurobi_persistent"

        grid = tc.EnergyGridWithCommitment(epsilon=eps)

        # Build and solve the commitment Benders model to optimality
        if hasattr(
            tc.EnergyGridWithCommitment, "setup_energy_grid_commitment_persistent"
        ):
            opt, m = (
                tc.EnergyGridWithCommitment.setup_energy_grid_commitment_persistent(
                    solver_name=mip_solver,
                    grid=grid,
                    eta_lb=-1e5,
                    eta_ub=1e5,
                    mode=mode,
                )
            )
        else:
            m = tc.EnergyGridWithCommitment.create_root(grid, eta_lb=-1e5, eta_ub=1e5)
            root_vars = list(m.commit.values())
            m.benders = BendersCutGenerator()
            m.benders.set_input(
                root_vars=root_vars,
                tol=1e-8,
                transform="standard_lp",
                allow_infeasible=True,
            )
            m.benders.add_subproblem(
                subproblem_fn=tc.EnergyGridWithCommitment.create_subproblem,
                subproblem_fn_kwargs={"root": m, "grid": grid, "mode": mode},
                root_eta=m.eta,
                subproblem_solver=mip_solver,
            )
            opt = pyo.SolverFactory(mip_solver)
            opt.set_instance(m)

        # Short Benders loop to optimality
        for i in range(20):
            opt.solve(tee=False, save_results=False)
            cuts = m.benders.generate_cut()
            for c in cuts:
                opt.add_constraint(c)
            if not cuts:
                break
        self.assertLess(i, 20, "Benders should converge well before 20 iterations")

        # AOS pass – Gurobi Solution Pool (PoolSearchMode=2)
        candidate_pool, data = aos_benders_generate_candidates(
            m=m,
            rel_gap=rel_gap,
            num_solutions=num_solutions,
            mip_solver=mip_solver,  # ignored by the generator
            enumeration_method="gurobi_pool",
            tee=False,
        )
        true_pool = aos_benders_filter(candidate_pool, data, tee=False, tee_final=False)

        # Extract the commitment vectors
        recovered = set()
        for sol in true_pool:
            sol.load_into_model(
                model=m,
                value_overrides=None,
                descend_into=True,
                skip_nan_inf=True,
                error_if_value_missing=False,
                track_missing=True,
                track_fixed=True,
                track_unfixed=True,
                track_nan_inf=True,
                unfix_by_default=False,
                fix_continuous=False,
                fix_binary=False,
                fix_integer=False,
                fix_if_sol_var_fixed=False,
                fix_var_names=None,
            )
            u1 = int(round(pyo.value(m.commit["bus1"])))
            u2 = int(round(pyo.value(m.commit["bus2"])))
            recovered.add((u1, u2))

        self.assertEqual(
            recovered,
            expected_patterns,
            f"Expected optimal commitment patterns {expected_patterns}, "
            f"got {recovered}",
        )
        self.assertEqual(
            len(true_pool),
            3,
            f"Expected true pool of size 3, got {len(true_pool)}",
        )

    @unittest.skipIf(not numpy_available, "numpy is not available.")
    @unittest.skipIf(
        not gurobi_available, "Gurobi is required for enumeration_method='gurobi_pool'"
    )
    def test_commitment_3bus_aos_benders_gurobi_pool_relgap_positive(self):
        """With a modest positive relative gap the three optimal patterns
        must still be present in the true pool (Gurobi Solution Pool path)."""
        eps = 1.0
        mode = 2
        rel_gap = 0.01
        num_solutions = 20
        expected_patterns = {(1, 0), (0, 1), (1, 1)}
        mip_solver = "gurobi_persistent"

        grid = tc.EnergyGridWithCommitment(epsilon=eps)

        if hasattr(
            tc.EnergyGridWithCommitment, "setup_energy_grid_commitment_persistent"
        ):
            opt, m = (
                tc.EnergyGridWithCommitment.setup_energy_grid_commitment_persistent(
                    solver_name=mip_solver,
                    grid=grid,
                    eta_lb=-1e5,
                    eta_ub=1e5,
                    mode=mode,
                )
            )
        else:
            m = tc.EnergyGridWithCommitment.create_root(grid, eta_lb=-1e5, eta_ub=1e5)
            root_vars = list(m.commit.values())
            m.benders = BendersCutGenerator()
            m.benders.set_input(
                root_vars=root_vars,
                tol=1e-8,
                transform="standard_lp",
                allow_infeasible=True,
            )
            m.benders.add_subproblem(
                subproblem_fn=tc.EnergyGridWithCommitment.create_subproblem,
                subproblem_fn_kwargs={"root": m, "grid": grid, "mode": mode},
                root_eta=m.eta,
                subproblem_solver=mip_solver,
            )
            opt = pyo.SolverFactory(mip_solver)
            opt.set_instance(m)

        for i in range(20):
            opt.solve(tee=False, save_results=False)
            cuts = m.benders.generate_cut()
            for c in cuts:
                opt.add_constraint(c)
            if not cuts:
                break

        candidate_pool, data = aos_benders_generate_candidates(
            m=m,
            rel_gap=rel_gap,
            num_solutions=num_solutions,
            mip_solver=mip_solver,
            enumeration_method="gurobi_pool",
            tee=False,
        )
        true_pool = aos_benders_filter(candidate_pool, data, tee=False, tee_final=False)

        recovered = set()
        for sol in true_pool:
            sol.load_into_model(
                model=m,
                value_overrides=None,
                descend_into=True,
                skip_nan_inf=True,
                error_if_value_missing=False,
                track_missing=True,
                track_fixed=True,
                track_unfixed=True,
                track_nan_inf=True,
                unfix_by_default=False,
                fix_continuous=False,
                fix_binary=False,
                fix_integer=False,
                fix_if_sol_var_fixed=False,
                fix_var_names=None,
            )
            u1 = int(round(pyo.value(m.commit["bus1"])))
            u2 = int(round(pyo.value(m.commit["bus2"])))
            recovered.add((u1, u2))

        self.assertTrue(
            expected_patterns.issubset(recovered),
            f"True pool missing some optimal patterns. "
            f"Expected at least {expected_patterns}, recovered {recovered}",
        )


class TestAOS_Benders_Non_Persistent(unittest.TestCase):

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    def test_farmer_end_to_end_deterministic(self, mip_solver):
        num_solutions = 50
        mode = "d"
        tee = True
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
        tee = True
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
        tee = True
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
        tee = True
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
        tee = True
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
        tee = True
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
                ignore_opt_tol_in_basis=False,
            )
            assert (
                len(candidate_pool) == expected_candidate_pool_size[index]
            ), f"Expected {expected_candidate_pool_size[index]} candidate solutions, got {len(candidate_pool)}"

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    def test_abs_aos_benders_generate_candidates_single_scenario_with_offset(
        self, mip_solver
    ):
        num_solutions = 50
        rel_gaps = [0, 0.01, 0.5]
        solver_name = mip_solver
        expected_candidate_pool_size = [1, 3, 3]
        for index, rel_gap in enumerate(rel_gaps):
            opt, m = tc.absolute_value.run_absolute_value(
                mip_solver,
                mode="d",
                transform="standard_lp",
                add_upper_bounds=True,
                include_print=True,
                is_persistent=False,
                obj_offset=1,
            )
            print(f"{pyo.value(m.obj)=}")
            m.x.setub(10)
            m.x.setlb(-10)
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
                ignore_opt_tol_in_basis=False,
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
                ignore_opt_tol_in_basis=False,
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
    def test_abs_aos_benders_filter_single_scenario_with_offset(self, mip_solver):
        num_solutions = 50
        rel_gaps = [0, 0.01, 0.5]
        solver_name = mip_solver
        expected_candidate_pool_size = [1, 3, 3]
        expected_true_pool_size = [1, 3, 3]
        for index, rel_gap in enumerate(rel_gaps):
            opt, m = tc.absolute_value.run_absolute_value(
                mip_solver,
                mode="d",
                transform="standard_lp",
                add_upper_bounds=True,
                include_print=True,
                is_persistent=False,
                obj_offset=1,
            )
            print(f"{pyo.value(m.obj)=}")
            m.x.setub(10)
            m.x.setlb(-10)
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
                candidate_pool, data, tee=False, tee_final=False
            )
            assert (
                len(true_pool) == expected_true_pool_size[index]
            ), f"Expected {expected_true_pool_size[index]} true solutions, got {len(true_pool)}"

    @parameterized.expand(input=non_persistent_mip_solvers, skip_on_empty=True)
    def test_abs_aos_benders_filter_multiple_scenarios(self, mip_solver):
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
                ignore_opt_tol_in_basis=False,
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
    def test_abs_aos_benders_filter_multiple_scenarios_with_obj_offset(
        self, mip_solver
    ):
        num_solutions = 50
        rel_gaps = [0, 0.01, 0.5]
        solver_name = mip_solver
        expected_candidate_pool_size = [1, 5, 5]
        expected_true_pool_size = [1, 5, 5]
        for index, rel_gap in enumerate(rel_gaps):
            opt, m = tc.absolute_value.run_absolute_value(
                mip_solver,
                mode="s",
                transform="standard_lp",
                add_upper_bounds=True,
                include_print=False,
                is_persistent=False,
                obj_offset=1,
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
                ignore_opt_tol_in_basis=False,
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
