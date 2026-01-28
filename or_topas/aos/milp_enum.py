#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2026
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging

logger = logging.getLogger(__name__)

import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from or_topas.solnpool import PyomoPoolManager, PoolPolicy
from or_topas.util import pyomo_utils, numpy_utils
from or_topas.aos import (
    gurobi_generate_solutions,
    enumerate_binary_solutions,
    enumerate_linear_solutions,
)


def enumerate_mixed_integer_linear_program_solutions(
    model,
    integer_program_method="gurobi_generate_solutions",
    linear_program_method="enumerate_linear_solutions",
    integer_program_options=dict(),
    linear_program_options=dict(),
    merge_all_results_pools=False,
    custom_as_solution=None,
):
    """
    Finds alternative optimal solutions for a mixed-integer linear problem by solving
    for integer alternative optimal solutions and then linear alternative optimal solutions.

    This function supports two methods for IP AOS generation and one method for LP AOS generation
    IP Methods: gurobi_generate_solutions and enumerate_binary_solutions
    LP Methods: enumerate_linear_solutions

    N.B. that gurobi_generate_solutions tolerates integer variables while enumerate_binary_solutions does not

    Parameters
    ----------
    model : ConcreteModel
        A concrete Pyomo model
    integer_program_method : String
        method name for IP AOS method
    linear_program_method : String
        method name for LP AOS method
    integer_program_options : dict
        dictionary of options for the integer program AOS solver
        passed to method using **kwargs style
    linear_program_options : dict
        dictionary of options for the linear program AOS solver
        passed to method using **kwargs style
    merge_all_results_pools : boolean
        choice to either return one pool per IP solution or a single combined pool
        in the returned pool manager
    custom_as_solution : Function or None,
        as_solution method to the pool managers, None results in use of default



    Returns
    -------
    milp_pool_manager
        A PyomoPoolManager object for the combined MILP alternative optimal solution results
    ip_pool_manager
        A PyomoPoolManager object for the IP alternative optimal solution results

    """
    supported_integer_program_methods = {
        "gurobi_generate_solutions": gurobi_generate_solutions,
        "enumerate_binary_solutions": enumerate_binary_solutions,
    }
    supported_linear_program_methods = {
        "enumerate_linear_solutions": enumerate_linear_solutions
    }

    assert (
        integer_program_method in supported_integer_program_methods
    ), f"Requested {integer_program_method=} not in {supported_integer_program_methods=}"
    assert (
        linear_program_method in supported_linear_program_methods
    ), f"Requested {linear_program_method=} not in {supported_linear_program_methods=}"

    # pool managers set up
    ip_pm_name = f"{model.name}_aos_ip_pool"
    milp_pm_name = f"aos_milp_pool_index_"
    ip_pool_manager = PyomoPoolManager(
        name=ip_pm_name, policy=PoolPolicy.keep_best, as_solution=custom_as_solution
    )
    milp_first_pool_name = milp_pm_name + "0"
    milp_pool_manager = PyomoPoolManager(
        name=milp_first_pool_name,
        policy=PoolPolicy.keep_best,
        as_solution=custom_as_solution,
    )

    # method selections
    ip_method = supported_integer_program_methods.get(integer_program_method)
    lp_method = supported_linear_program_methods.get(linear_program_method)

    # IP AOS Solve

    ip_pool_manager = ip_method(model, **integer_program_options)

    # loop through IP solutions
    for index, solution in enumerate(ip_pool_manager.solutions):
        pyomo_utils.load_solution_into_model(
            model,
            solution,
            descend_into=True,
            error_if_value_missing=True,
            fix_binary=True,
            fix_integer=True,
            fix_if_model_var_fixed=True,
        )
        # if not merge_all_results_pools, add new active pool for LP results
        if not merge_all_results_pools:
            milp_current_pool_name = milp_pm_name + str(index)
            milp_pool_manager.add_pool(
                name=milp_current_pool_name,
                policy=PoolPolicy.keep_best,
                as_solution=custom_as_solution,
            )

        # lp solve
        # TODO: check if we need to update the LP methods to tolerate fixed non-continuous variables
        lp_method(model, pool_manager=milp_pool_manager, **linear_program_options)

        # delete aos_block locally
        # TODO, figure out if delete in lp aos method sufficient
        # TODO: figure out how to grab the AOS block to delete
        # probably easiest to add name to model somewhere in add_aos_block method in pyomo_utils
        # could also just add ability to delete aos_block after solve to lp method
        # second option probably best

        # delete aos_block if being used persistently
        # shouldnt need this at the moment, we build new solver object in every lp_method call
        # this may be a performence enhancement for latter to carry around the lp method persistently
        # TODO: add persistent carry of LP model around

    return milp_pool_manager, ip_pool_manager
