import logging

logger = logging.getLogger(__name__)

import pyomo.environ as pyo
from or_topas.util import pyomo_utils
from or_topas.solnpool import PyomoPoolManager, PoolPolicy
from or_topas.aos import (
    gurobi_enumerate_linear_solutions,
    enumerate_linear_solutions,
    enumerate_binary_solutions,
    gurobi_generate_solutions,
)


def enumerate_mixed_integer_linear_solutions(
    model,
    *,
    rel_opt_gap=None,
    abs_opt_gap=None,
    lower_objective_threshold=None,
    upper_objective_threshold=None,
    ip_method="enumerate_binary_solutions",
    ip_options={},
    lp_method="enumerate_linear_solutions",
    lp_options={},
    tee=False,
):
    """
    Finds alternative optimal solutions a mixed-integer linear program.
    First finds integer alternative solutions using a choice of IP AOS methods.
    Second finds linear alternative solutions using a choice of LP AOS methods.
    The LP AOS method is run for each IP solution with IP variables fixed.

    Parameters
    ----------
    model : ConcreteModel
        A concrete Pyomo model
    rel_opt_gap : float or None
        The relative optimality gap for the original objective for which
        variable bounds will be found. None indicates that a relative gap
        constraint will not be added to the model.
    abs_opt_gap : float or None
        The absolute optimality gap for the original objective for which
        variable bounds will be found. None indicates that an absolute gap
        constraint will not be added to the model.
    lower_objective_threshold : float or None
        Sense dependent, used in maximization problems to add a constraint of
        form objective >= lower_objective_threshold. If not satisfied at
        the optimal objective, method returns pool manager with no solutions
        added. None indicates that a lower objective threshold will not
        be added to the model.
    upper_objective_threshold : float or None
        Sense dependent, used in minimization problems to add a constraint of
        form objective <= upper_objective_threshold. If not satisfied at
        the optimal objective, method returns pool manager with no solutions
        added. None indicates that a lower objective threshold will not
        be added to the model.
    ip_method : string
        Name of the IP AOS method to apply.
    ip_options : dict
        Solver option-value pairs to be passed to the IP AOS method.
    lp_method : string
        Name of the LP AOS method to apply.
    lp_options : dict
        Solver option-value pairs to be passed to the LP AOS method.
    tee : boolean
        Boolean indicating that the solver output should be displayed.

    Returns
    -------
    pool_manager
        A PyomoPoolManager object
    """
    logger.info("STARTING MILP ENUMERATION ANALYSIS")
    # TODO: add overall timelimit capability
    # TODO: add adaptive limit to number of solutions in LP AOS step

    #allow single point of control for solver output
    if not 'tee' in lp_options:
        lp_options['tee'] = tee
    if not 'tee' in ip_options:
        ip_options['tee'] = tee

    allowed_ip_methods = {
        "enumerate_binary_solutions": enumerate_binary_solutions,
        "gurobi_generate_solutions": gurobi_generate_solutions,
    }
    allowed_lp_methods = {
        "enumerate_linear_solutions": enumerate_linear_solutions,
        "gurobi_enumerate_linear_solutions": gurobi_enumerate_linear_solutions,
    }
    # check that IP AOS method is in permitted list
    assert (
        ip_method in allowed_ip_methods
    ), f"Attempted IP AOS Method {ip_method} not supported"
    ip_method_handle = allowed_ip_methods[ip_method]
    # check that LP AOS method is in the permitted list
    assert (
        lp_method in allowed_lp_methods
    ), f"Attempted LP AOS Method {lp_method} not supported"
    lp_method_handle = allowed_lp_methods[lp_method]
    # default to controlling the entire pool manager for first version
    pool_manager = PyomoPoolManager()
    pool_manager.add_pool(
        name="enumerate_milp_solutions_integer_var_pool", policy=PoolPolicy.keep_all
    )

    # get original objective
    original_objective = pyomo_utils.get_active_objective(model)
    original_obj_sense_is_min = original_objective.sense == pyo.minimize

    # get list of originally fixed variables
    all_variables = pyomo_utils.get_model_variables(model, include_fixed=True)
    original_fixed_variables_set = {var.name for var in all_variables if var.is_fixed()}

    logger.info(f"PERFORMING IP AOS GENERATION WITH METHOD:{ip_method}")
    # run IP AOS method in IP pool manager
    try:
        ip_method_handle(
            model=model,
            pool_manager=pool_manager,
            rel_opt_gap=rel_opt_gap,
            abs_opt_gap=abs_opt_gap,
            lower_objective_threshold=lower_objective_threshold,
            upper_objective_threshold=upper_objective_threshold,
            **ip_options,
        )
    except Exception as e:
        raise RuntimeError(f"Runtime Issue in IP AOS Method call") from e

    logger.info("COMPLETED IP AOS GENERATION")
    # check that at least one solution was found
    if len(pool_manager) == 0:
        raise RuntimeError(f"No solutions generated in IP AOS Step")

    # find 'best' solution in list
    # TODO: there should be a generalized way to do this for pools, either get best sol or get ith best sol
    # defaulting to using objectives here instead of objective due to PyomoSolution definition
    best_sol = pool_manager.last_solution
    for current_sol in pool_manager:
        better_obj = (
            current_sol.objectives[0] < best_sol.objectives[0]
            if original_obj_sense_is_min
            else current_sol.objectives[0] > best_sol.objectives[0]
        )
        if better_obj:
            best_sol = current_sol

    # compute the true lower/upper threshold values on basis of best solution
    best_sol_val = best_sol.objectives[0]
    lp_effective_lower_threshold = None
    lp_effective_upper_threshold = None
    if original_obj_sense_is_min:
        # compute rel_tol based bound
        rel_tol_bound = (
            best_sol_val + abs(best_sol_val * rel_opt_gap)
            if rel_opt_gap is not None
            else None
        )
        # compute abs_tol based bound
        abs_tol_bound = best_sol_val + abs_opt_gap if abs_opt_gap is not None else None

        # get least/tightest upper bound of rel_tol_bound, abs_tol_bound, upper_objective_threshold
        # return none if all are none
        bounds = [rel_tol_bound, abs_tol_bound, upper_objective_threshold]
        if any(x is not None for x in bounds):
            lp_effective_upper_threshold = min(x for x in bounds if x is not None)
    else:
        # compute rel_tol based bound
        rel_tol_bound = (
            best_sol_val - abs(best_sol_val * rel_opt_gap)
            if rel_opt_gap is not None
            else None
        )
        # compute abs_tol based bound
        abs_tol_bound = best_sol_val - abs_opt_gap if abs_opt_gap is not None else None

        # find greatest/tightest lower bound of rel_tol_bound, abs_tol_bound, lower_objective_threshold
        # return none if all are none
        bounds = [rel_tol_bound, abs_tol_bound, lower_objective_threshold]
        if any(x is not None for x in bounds):
            lp_effective_lower_threshold = max(x for x in bounds if x is not None)
    logger.info(f"PERFORMING LP AOS GENERATION WITH METHOD:{lp_method}")

    # need access to ip solution pool while allowing manager to change
    ip_sol_pool = pool_manager.active_pool
    # for each sol in IP sol pool
    for index, ip_sol in enumerate(ip_sol_pool):
        # load sol into model, fixing integer (including binary) variables
        ip_sol.load_into_model(
            model, fix_integer=True, fix_var_names=original_fixed_variables_set
        )

        # run LP AOS method
        try:
            lp_pool_manager = lp_method_handle(
                model,
                lower_objective_threshold=lp_effective_lower_threshold,
                upper_objective_threshold=lp_effective_upper_threshold,
                **lp_options,
            )
        except Exception as e:
            # restore model to original form
            # unfix the vars that do not need to be fixed
            for var in all_variables:
                if var.is_fixed() and (not var.name in original_fixed_variables_set):
                    var.unfix()
            raise RuntimeError(
                f"Runtime Issue in LP AOS Method call on ip_sol with index {index}"
            ) from e
        if len(lp_pool_manager) == 0:
            raise RuntimeError(
                f"No solutions generated in LP AOS Method call on ip_sol with index {index}"
            )

        # grab created pool and put into overall pool manager
        lp_pool = lp_pool_manager.active_pool

        # TODO: this is a workaround for the absence of a pool_manager append pool method
        # rename pool with consistent naming
        lp_pool.name = f"enumerate_milp_solutions_lp_pool_for_ip_sol_{index}"
        pool_manager._pools[lp_pool.name] = lp_pool
    logger.info("COMPLETED LP AOS GENERATION")
    logger.info("COMPLETED MILP ENUMERATION ANALYSIS")

    # restore model to original form
    # unfix the vars that do not need to be fixed
    for var in all_variables:
        if var.is_fixed() and (not var.name in original_fixed_variables_set):
            var.unfix()

    return pool_manager
