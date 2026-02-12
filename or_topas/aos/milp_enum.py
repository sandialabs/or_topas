
import logging

logger = logging.getLogger(__name__)

import pyomo.environ as pyo
from or_topas.util import pyomo_utils
from or_topas.solnpool import PyomoPoolManager, PoolPolicy
from or_topas.aos import shifted_lp
from pyomo.contrib import appsi


def enumerate_mixed_integer_linear_solutions(
    model,
    *,
    rel_opt_gap=None,
    abs_opt_gap=None,
    lower_objective_threshold=None,
    upper_objective_threshold=None,
    ip_method="",
    ip_options={},
    lp_method="",
    lp_options={},
    ip_solver="gurobi",
    solver_options={},
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

    #check that IP AOS method is in permitted list

    #check that LP AOS method is in the permitted list

    #default to controlling the entire pool manager for first version
    pool_manager = PyomoPoolManager()
    pool_manager.add_pool(
        name="enumerate_milp_solutions_integer_var_pool", policy=PoolPolicy.keep_all
    )

    logger.info(f"PERFORMING IP AOS GENERATION WITH METHOD:{ip_method}")
    #run IP AOS method in IP pool manager
    #check that at least one solution was found
    #compute the true lower/upper threshold values on basis of best solution
    logger.info("COMPLETED IP AOS GENERATION")

    logger.info(f"PERFORMING LP AOS GENERATION WITH METHOD:{lp_method}")
    #create LP pool manager
    #for each sol in IP sol pool
    #load sol into model
    #create new pool in LP for IP sol
    #run LP AOS
    #catch errors, but errors should be limited to setting issues
    logger.info("COMPLETED LP AOS GENERATION")

    logger.info("COMPLETED MILP ENUMERATION ANALYSIS")

    #combine the IP and LP pool managers into a single pool manager

    return pool_manager
