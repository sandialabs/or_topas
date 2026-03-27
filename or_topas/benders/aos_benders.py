import pyomo.environ as pyo
from or_topas.benders import (
    BendersGenerator_Serial,
)
from or_topas.solnpool.solnpool import _as_pyomo_solution
from or_topas.solnpool import PyomoPoolManager, PoolPolicy
from or_topas.benders.tests import test_cases as tc
import time
import or_topas
from or_topas.util import pyomo_utils
from or_topas.util.mymunch import MyMunch
from pyomo.common.collections import ComponentSet


def aos_benders_generate_candidates(
    m,
    rel_gap=0,
    num_solutions=10,
    mip_solver="glpk",
    skip_vars=None,
    ignore_opt_tol_in_basis=False,
):
    # At present, only LP AOS Supported

    # assume that we have a solved model
    # objectives = [pyo.value(o) for o in m.component_objects(pyo.Objective, descend_into=False, active=True)]
    objectives = [
        o for o in m.component_objects(pyo.Objective, descend_into=False, active=True)
    ]
    assert len(objectives) == 1, "Should only have one active objective"

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

    # reactivate Benders blocks
    # add in if the deactivate ever gets used again
    # benders_block.activate()

    # filter solutions step
    upper_bound = objectives[0] + rel_gap * abs(objectives[0]) + 1e-7
    other_data_munch = MyMunch(
        objective_expr=m.obj,
        model=m,
        # TODO: update this to use the benders_block found above
        # benders_block=m.benders,
        benders_block=benders_block,
        scenarios=m.scenarios,
        upper_bound=upper_bound,
        lower_bound=pyo.value(objectives[0]),
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

    aos_filtering_model = data.model
    for index, sol in enumerate(candidate_pool):
        # iterate through each solution in the candidate pool

        # we assume the solutions just have master problem data
        if tee:
            print(f"Solution {index} check")
            # these are model specific print checks
            print(f"Before Loading Sol")
            print(
                f"x={str([(c,pyo.value(aos_filtering_model.devoted_acreage[c])) for c in aos_filtering_model.devoted_acreage.index_set()])}"
            )
            print(
                f"eta={str([(c,pyo.value(aos_filtering_model.eta[c])) for c in aos_filtering_model.eta.index_set()])}"
            )
            print(f"obj: {pyo.value(data.objective_expr)}")

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
            print(
                f"x={str([(c,pyo.value(aos_filtering_model.devoted_acreage[c])) for c in aos_filtering_model.devoted_acreage.index_set()])}"
            )
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
            print(
                f"x={str([(c,pyo.value(aos_filtering_model.devoted_acreage[c])) for c in aos_filtering_model.devoted_acreage.index_set()])}"
            )
            print(
                f"eta={str([(c,pyo.value(aos_filtering_model.eta[c])) for c in aos_filtering_model.eta.index_set()])}"
            )
            print(f"obj: {pyo.value(data.objective_expr)}")

        # pull out the etas from this results list
        # this is tightly assuming an eta per scenario
        # will need to be changed for anything but multicut benders
        for i, s in enumerate(data.scenarios):
            aos_filtering_model.eta[s] = results_list[i].subproblem_eta
        if tee:
            print(f"After eta load {index}")
            # Again update to be root vars
            print(
                f"x={str([(c,pyo.value(aos_filtering_model.devoted_acreage[c])) for c in aos_filtering_model.devoted_acreage.index_set()])}"
            )
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
        if ((data.lower_bound - smoothing_tol) <= present_obj_value) and (
            present_obj_value <= data.upper_bound + smoothing_tol
        ):
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


# adapted from the tests in or_topas.benders for serial farmer
def test_farmer(
    mip_solver,
    mode="s",
    transform="standard_lp",
    add_upper_bounds=False,
    include_print=False,
):

    t0 = time.time()
    local_farmer = tc.Farmer()
    if mode == "d":
        local_farmer = tc.Farmer()
        local_farmer.scenario_probabilities = {"AverageScenario": 1.0}
        local_farmer.scenarios = ["AverageScenario"]
    opt, m = tc.Farmer.setup_farmer(
        local_farmer,
        solver_name=mip_solver,
        transform=transform,
    )
    if add_upper_bounds:
        for s in m.scenarios:
            m.eta[s].setub(1_000_000)
    if include_print:
        print(
            "{0:<15}{1:<15}{2:<15}{3:<15}{4:<15}".format(
                "# Cuts", "Corn", "Sugar Beets", "Wheat", "Total_Time"
            )
        )
    for i in range(30):
        res = opt.solve(m, tee=False)
        cuts_added = m.benders.generate_cut()
        # for c in cuts_added:
        #     opt.add_constraint(c)
        if include_print:
            print(
                "{0:<15}{1:<15.2f}{2:<15.2f}{3:<15.2f}{4:<15.2f}".format(
                    len(cuts_added),
                    m.devoted_acreage["CORN"].value,
                    m.devoted_acreage["SUGAR_BEETS"].value,
                    m.devoted_acreage["WHEAT"].value,
                    time.time() - t0,
                )
            )
        if len(cuts_added) == 0:
            break

    if mode == "s":
        tol = 1e-7
        assert abs(m.devoted_acreage["CORN"].value - 80) < tol
        assert abs(m.devoted_acreage["SUGAR_BEETS"].value - 250) < tol
        assert abs(m.devoted_acreage["WHEAT"].value - 170) < tol
    return opt, m


# TODO: update this to be an example with farmers
# pull from benders.tests.tc for create farmer
def aos_farmer_test(
    mip_solver="glpk",
    num_solutions=10,
    mode="s",
    tee=False,
    tee_final=False,
    rel_gap=0.01,
    use_skip_vars=False,
):
    opt, m = test_farmer(
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
    candidate_pool, data = aos_generate_candidates_lp(
        m=m,
        rel_gap=rel_gap,
        num_solutions=num_solutions,
        mip_solver=mip_solver,
        skip_vars=skip_vars,
    )
    true_pool = aos_filter(candidate_pool, data, tee=tee, tee_final=tee_final)
    return true_pool


# TODO: example for extended absolute value
# should just get the vertices

# TODO: example for DCOPF simple
