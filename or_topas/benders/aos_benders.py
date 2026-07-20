import pyomo.environ as pyo
from or_topas.benders import (
    BendersGenerator_Serial,
)
from or_topas.solnpool.solnpool import _as_pyomo_solution
from or_topas.solnpool import PyomoPoolManager, PoolPolicy
from or_topas.benders.tests.test_cases import Farmer as TestCasesFarmer
import time
import or_topas
from or_topas.util import pyomo_utils
from or_topas.util.mymunch import MyMunch
from or_topas.util.pyomo_utils import pprint_solution
from pyomo.common.collections import ComponentSet


def aos_benders_generate_candidates(
    m,
    rel_gap=0,
    num_solutions=10,
    mip_solver="glpk",
    skip_vars=None,
    ignore_opt_tol_in_basis=False,
    bound_smoothing_tol=1e-6,
    tee=False,
    scenarios=None,
    enumeration_method="linear",
    binary_var_set=None,
):
    # At present, only LP AOS Supported

    # assume that we have a solved model
    # objectives = [pyo.value(o) for o in m.component_objects(pyo.Objective, descend_into=False, active=True)]
    if scenarios is None and hasattr(m, "scenarios"):
        scenarios = m.scenarios

    objectives = [
        o for o in m.component_objects(pyo.Objective, descend_into=False, active=True)
    ]
    assert len(objectives) == 1, "Should only have one active objective"
    # need to grap lower bound before objectives gets updated
    lower_bound = pyo.value(objectives[0])
    if tee:
        print(f"{lower_bound=}")

    # compute gap settings

    # find all Benders blocks, general
    # get benders blocks
    benders_blocks = [
        b
        for b in m.component_data_objects(pyo.Block)
        if "BendersGenerator" in str(type(b))
    ]
    assert len(benders_blocks) == 1, "There should only be one benders block"
    benders_block = benders_blocks[0]

    # hypothetically we want to deactivate Benders block before AOS pass
    # since we don't want the Benders Cut block stuff in solutions, but need m.benders.cuts
    # since benders cuts live in the benders_block, not the master, at present we can't deactivate
    # benders_block.deactivate()

    if enumeration_method == "linear":
        if binary_var_set is not None and tee:
            raise RuntimeWarning(
                f"In {enumeration_method=}, the value of {binary_var_set=} should be None and was not"
            )
        # check that all vars have bounds
        # LP AOS needs this
        unbounded_vars = [
            v
            for v in m.component_data_objects(pyo.Var, descend_into=True, active=True)
            if (v.has_lb() == False or v.has_ub() == False)
        ]
        assert (
            len(unbounded_vars) == 0
        ), f"Need to make sure all vars have bounds for LP AOS methods, there are {len(unbounded_vars)} unbounded variables"

        # do AOS pass
        # can we augment the AOS methods to return to us what the min/max supported values are?
        # it computes it, we should be able to augment to get it

        candidate_sol_pool = or_topas.aos.enumerate_linear_solutions(
            model=m,
            solver=mip_solver,
            rel_opt_gap=rel_gap,
            num_solutions=num_solutions,
            variables_to_skip=skip_vars,
            ignore_opt_tol_in_basis=ignore_opt_tol_in_basis,
        )
    elif enumeration_method == "binary":
        if (skip_vars is not None or ignore_opt_tol_in_basis) and tee:
            raise RuntimeWarning(
                f"In {enumeration_method=}, the value of {skip_vars=} should be None and {ignore_opt_tol_in_basis=} should be Falsy"
            )

        candidate_sol_pool = or_topas.aos.enumerate_binary_solutions(
            model=m,
            num_solutions=num_solutions,
            variables=None,
            rel_opt_gap=rel_gap,
            solver=mip_solver,
            tee=tee,
        )
    else:
        raise ValueError(
            f"enumeration_method must be 'linear' or 'binary', got {enumeration_method}"
        )

    # reactivate Benders blocks
    # add in if the deactivate ever gets used again
    # benders_block.activate()

    # filter solutions step
    upper_bound = lower_bound + rel_gap * abs(lower_bound) + bound_smoothing_tol
    if tee:
        print(f"{upper_bound=}")
    other_data_munch = MyMunch(
        objective_expr=m.obj,
        model=m,
        # TODO: update this to use the benders_block found above
        # benders_block=m.benders,
        benders_block=benders_block,
        scenarios=scenarios,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        objective=objectives[0],
    )
    return candidate_sol_pool, other_data_munch


def aos_benders_filter(
    candidate_pool, data, tee=False, tee_final=False, true_pool=None, smoothing_tol=1e-6
):

    # handle case when solution pool not given
    if true_pool is None:
        true_pool = PyomoPoolManager()
        true_pool.add_pool(name="aos_benders", policy=PoolPolicy.keep_all)
    if tee or tee_final:
        print(f"Candidate Solution pool has {len(candidate_pool)} sols")
        print(f"True Pool Inclusion Lower Bound: {data.lower_bound}")
        print(f"True Pool Inclusion Upper Bound: {data.upper_bound}")

    root_vars = data.benders_block.root_vars
    aos_filtering_model = data.model
    for index, sol in enumerate(candidate_pool):
        # iterate through each solution in the candidate pool

        # we assume the solutions just have master problem data
        if tee:
            print(f"Solution {index} check")
            # these are model specific print checks
            print(f"Before Loading Sol")
            pprint_solution(sol)

        # load the solution into model
        sol.load_into_model(
            model=aos_filtering_model,
            value_overrides=None,
            descend_into=True,
            skip_nan_inf=True,
            error_if_value_missing=False,
            track_missing=True,
            track_fixed=True,
            track_unfixed=True,
            track_nan_inf=True,
            unfix_by_default=False,  # unfix_by_default = True,
            fix_continuous=False,
            fix_binary=False,
            fix_integer=False,
            fix_if_sol_var_fixed=False,
            fix_var_names=None,
        )
        if tee:
            print(f"After but before evaluate Loading Sol {index}")
            # print(
            #     f"x={str([(c,pyo.value(aos_filtering_model.devoted_acreage[c])) for c in aos_filtering_model.devoted_acreage.index_set()])}"
            # )
            print(f"x={str([(rv,pyo.value(rv)) for rv in root_vars])}")
            print(
                f"eta={str([(c,pyo.value(aos_filtering_model.eta[c])) for c in aos_filtering_model.eta.index_set()])}"
            )
            print(f"obj: {pyo.value(data.objective_expr)}")

        # might be able to get this from getting the active objective and then calling pyo.value on that
        objective_based_on_loaded_etas = pyo.value(data.objective_expr)

        # evaluate all subproblems
        # we can probably turn off build cut here
        results_list = data.benders_block.evaluate_all_subproblems()

        # feasibility check goes here.
        # if allow infeasible and model infeasible, break and go to next solution
        if data.benders_block.allow_infeasible and any(
            result.subproblem_infeasible for result in results_list
        ):
            # if any of the scenarios result in infeasible, the infeasible point
            if tee:
                # helper comments for print outs
                print(f"Sol {index} not added to true pool, infeasible")
            # can skip to next solution
            continue

        if tee:
            print(f"After evaluate before eta load {index}")
            # Update to be root variables on line below, Benders block will track the root vars
            # print(
            #     f"x={str([(c,pyo.value(aos_filtering_model.devoted_acreage[c])) for c in aos_filtering_model.devoted_acreage.index_set()])}"
            # )
            print(f"x={str([(rv,pyo.value(rv)) for rv in root_vars])}")
            print(
                f"eta={str([(c,pyo.value(aos_filtering_model.eta[c])) for c in aos_filtering_model.eta.index_set()])}"
            )
            print(f"obj: {pyo.value(data.objective_expr)}")

        # pull out the etas from this results list
        # this is tightly assuming an eta per scenario
        # will need to be changed for anything but multicut benders
        if data.scenarios is None:
            aos_filtering_model.eta = results_list[0].subproblem_eta
        else:
            for i, s in enumerate(data.scenarios):
                aos_filtering_model.eta[s] = results_list[i].subproblem_eta
        if tee:
            print(f"After eta load {index}")
            # Again update to be root vars
            # print(
            #     f"x={str([(c,pyo.value(aos_filtering_model.devoted_acreage[c])) for c in aos_filtering_model.devoted_acreage.index_set()])}"
            # )
            print(f"x={str([(rv,pyo.value(rv)) for rv in root_vars])}")
            print(
                f"eta={str([(c,pyo.value(aos_filtering_model.eta[c])) for c in aos_filtering_model.eta.index_set()])}"
            )
            print(f"obj: {pyo.value(data.objective_expr)}")

        # check filter value
        local_variables = pyomo_utils.get_model_variables(
            aos_filtering_model, include_fixed=True
        )
        local_variable_values = [(v.name, pyo.value(v)) for v in local_variables]
        orig_objective = pyomo_utils.get_active_objective(aos_filtering_model)
        present_obj_value = pyo.value(orig_objective)

        added_to_pool = False
        lower_bound_check = (data.lower_bound - smoothing_tol) <= present_obj_value
        upper_bound_check = present_obj_value <= data.upper_bound + smoothing_tol
        if lower_bound_check and upper_bound_check:
            true_pool.add(variables=local_variables, objective=orig_objective)
            added_to_pool = True

        if tee or tee_final:
            if added_to_pool:
                print(f"Sol {index} added to true pool")
            else:
                print(f"Sol {index} not added to true pool, not in bounds")

            print(f"After get model variables on updated model")
            print(f"x = {local_variable_values=}")
            print(
                f"obj val based on pre-evaluate etas : {objective_based_on_loaded_etas}"
            )
            print(f"obj val based on orig_objective: {present_obj_value}")
            print(
                f"Diff in pre-eval and post-eval objectives: {(abs(objective_based_on_loaded_etas-present_obj_value) > smoothing_tol)}"
            )
            print(
                f"Post-eval objective < Pre-eval objective: {present_obj_value + 10*smoothing_tol < objective_based_on_loaded_etas}"
            )
            print(
                f"Sol likely a natural (non-level induced) vertex: {present_obj_value < data.upper_bound - smoothing_tol}"
            )

    return true_pool


def aos_farmer_test(
    mip_solver="glpk",
    num_solutions=10,
    mode="s",
    tee=False,
    tee_final=False,
    rel_gap=0.01,
    use_skip_vars=False,
):
    opt, m = TestCasesFarmer.run_farmer(
        mip_solver, mode=mode, transform="standard_lp", add_upper_bounds=True
    )
    if use_skip_vars:
        # skip_vars = ComponentSet(m.eta[s] for s in m.scenarios)
        # skip_vars = ComponentSet(m.eta[s] for s in m.scenarios)
        skip_vars = ComponentSet()
        skip_vars.update(
            [m.benders.cuts[index] for index in m.benders.cuts.index_set()]
        )
    else:
        skip_vars = None
    candidate_pool, data = aos_benders_generate_candidates(
        m=m,
        rel_gap=rel_gap,
        num_solutions=num_solutions,
        mip_solver=mip_solver,
        skip_vars=skip_vars,
    )
    true_pool = aos_benders_filter(candidate_pool, data, tee=tee, tee_final=tee_final)
    return true_pool


# TODO: example for extended absolute value
# should just get the vertices

# TODO: example for DCOPF simple
