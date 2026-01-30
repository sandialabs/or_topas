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
from pyomo.common.modeling import unique_component_name
from pyomo.common.collections import ComponentSet
import pyomo.util.vars_from_expressions as vfe
import warnings


def get_active_objective(model):
    """
    Finds and returns the active objective function for a model. Currently
    assume that there is exactly one active objective.
    """

    active_objs = list(model.component_data_objects(pyo.Objective, active=True))
    if len(active_objs) != 1:
        raise RuntimeError(
            f"Model has {len(active_objs)} active objective functions, exactly one is required."
        )

    return active_objs[0]


def add_aos_block(model, name="_aos_block"):
    """Adds an alternative optimal solution block with a unique name."""
    aos_block = pyo.Block()
    model.add_component(unique_component_name(model, name), aos_block)
    return aos_block


def add_objective_constraint(
    target_block,
    objective,
    objective_value,
    rel_opt_gap=None,
    abs_opt_gap=None,
    lower_objective_threshold=None,
    upper_objective_threshold=None,
):
    """
    Adds a relative and/or absolute objective function constraint to the
    specified block.
    target_block : Pyomo block
        block on which to add the constraints
    objective : Pyomo objective
        objective to add the constraints based off on.
    objective_value : Float
        objective value to add the constraints based on.
    rel_opt_gap : float or None
        The relative optimality gap for the original objective for which
        a constraint on feasible objectives will be added.
        None indicates that a relative gap constraint will not be
        added to the model.
    abs_opt_gap : float or None
        The absolute optimality gap for the original objective for which
        a constraint on feasible objectives will be added.
        None indicates that a relative gap constraint will not be
        added to the model.
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
    """
    try:
        if lower_objective_threshold is not None:
            lower_objective_threshold = float(lower_objective_threshold)
    except ValueError:
        raise ValueError(
            f"lower_objective_threshold ({lower_objective_threshold}) must be None or numeric"
        )
    try:
        if upper_objective_threshold is not None:
            upper_objective_threshold = float(upper_objective_threshold)
    except ValueError:
        raise ValueError(
            f"upper_objective_threshold ({upper_objective_threshold}) must be None or numeric"
        )
    if not (rel_opt_gap is None or rel_opt_gap >= 0.0):
        raise ValueError(f"rel_opt_gap ({rel_opt_gap}) must be None or >= 0.0")
    if not (abs_opt_gap is None or abs_opt_gap >= 0.0):
        raise ValueError(f"abs_opt_gap ({abs_opt_gap}) must be None or >= 0.0")

    objective_constraints = []

    objective_is_min = objective.is_minimizing()
    objective_expr = objective.expr

    objective_sense = -1
    if objective_is_min:
        objective_sense = 1

    if rel_opt_gap is not None:
        objective_cutoff = objective_value + objective_sense * rel_opt_gap * abs(
            objective_value
        )

        if objective_is_min:
            target_block.optimality_tol_rel = pyo.Constraint(
                expr=objective_expr <= objective_cutoff
            )
        else:
            target_block.optimality_tol_rel = pyo.Constraint(
                expr=objective_expr >= objective_cutoff
            )
        objective_constraints.append(target_block.optimality_tol_rel)

    if abs_opt_gap is not None:
        objective_cutoff = objective_value + objective_sense * abs_opt_gap

        if objective_is_min:
            target_block.optimality_tol_abs = pyo.Constraint(
                expr=objective_expr <= objective_cutoff
            )
        else:
            target_block.optimality_tol_abs = pyo.Constraint(
                expr=objective_expr >= objective_cutoff
            )
        objective_constraints.append(target_block.optimality_tol_abs)

    # TODO: third level value change
    if objective_is_min and upper_objective_threshold is not None:
        target_block.upper_objective_bound = pyo.Constraint(
            expr=objective_expr <= upper_objective_threshold
        )
        objective_constraints.append(target_block.lower_objective_bound)
    if (not objective_is_min) and lower_objective_threshold is not None:
        target_block.upper_objective_bound = pyo.Constraint(
            expr=objective_expr >= lower_objective_threshold
        )
        objective_constraints.append(target_block.upper_objective_bound)

    return objective_constraints


def _filter_model_variables(
    variable_set,
    var_generator,
    include_continuous=True,
    include_binary=True,
    include_integer=True,
    include_fixed=False,
):
    """
    Filters variables from a variable generator and adds them to a set.
    """
    for var in var_generator:
        if var in variable_set or (var.is_fixed() and not include_fixed):
            continue
        if (
            (var.is_continuous() and include_continuous)
            or (var.is_binary() and include_binary)
            or (var.is_integer() and include_integer)
        ):
            variable_set.add(var)


def get_model_variables(
    model,
    components=None,
    include_continuous=True,
    include_binary=True,
    include_integer=True,
    include_fixed=False,
):
    """Gathers and returns all variables or a subset of variables from a
    Pyomo model.

    Parameters
    ----------
    model : ConcreteModel
        A concrete Pyomo model.
    components: None or a collection of Pyomo components
        The components from which variables should be collected. None
        indicates that all variables will be included. Alternatively, a
        collection of Pyomo Blocks, Constraints, or Variables (indexed or
        non-indexed) from which variables will be gathered can be provided.
        If a Block is provided, all variables associated with constraints
        in that that block and its sub-blocks will be returned. To exclude
        sub-blocks, a tuple element with the format (Block, False) can be
        used.
    include_continuous : boolean
        Boolean indicating that continuous variables should be included.
    include_binary : boolean
        Boolean indicating that binary variables should be included.
    include_integer : boolean
        Boolean indicating that integer variables should be included.
    include_fixed : boolean
        Boolean indicating that fixed variables should be included.

    Returns
    -------
    variable_set
        A Pyomo ComponentSet containing _GeneralVarData variables.

    """

    component_list = (pyo.Objective, pyo.Constraint)
    variable_set = ComponentSet()
    if components == None:
        var_generator = vfe.get_vars_from_components(
            model, component_list, include_fixed=include_fixed
        )
        _filter_model_variables(
            variable_set,
            var_generator,
            include_continuous,
            include_binary,
            include_integer,
            include_fixed,
        )
    else:
        for comp in components:
            if hasattr(comp, "ctype") and comp.ctype == pyo.Block:
                blocks = comp.values() if comp.is_indexed() else (comp,)
                for item in blocks:
                    variables = vfe.get_vars_from_components(
                        item, component_list, include_fixed=include_fixed
                    )
                    _filter_model_variables(
                        variable_set,
                        variables,
                        include_continuous,
                        include_binary,
                        include_integer,
                        include_fixed,
                    )
            elif (
                isinstance(comp, tuple)
                and hasattr(comp[0], "ctype")
                and comp[0].ctype == pyo.Block
            ):
                block = comp[0]
                descend_into = pyo.Block if comp[1] else False
                blocks = block.values() if block.is_indexed() else (block,)
                for item in blocks:
                    variables = vfe.get_vars_from_components(
                        item,
                        component_list,
                        include_fixed=include_fixed,
                        descend_into=descend_into,
                    )
                    _filter_model_variables(
                        variable_set,
                        variables,
                        include_continuous,
                        include_binary,
                        include_integer,
                        include_fixed,
                    )
            elif hasattr(comp, "ctype") and comp.ctype in component_list:
                constraints = comp.values() if comp.is_indexed() else (comp,)
                for item in constraints:
                    variables = pyo.expr.identify_variables(
                        item.expr, include_fixed=include_fixed
                    )
                    _filter_model_variables(
                        variable_set,
                        variables,
                        include_continuous,
                        include_binary,
                        include_integer,
                        include_fixed,
                    )
            elif hasattr(comp, "ctype") and comp.ctype == pyo.Var:
                variables = comp.values() if comp.is_indexed() else (comp,)
                _filter_model_variables(
                    variable_set,
                    variables,
                    include_continuous,
                    include_binary,
                    include_integer,
                    include_fixed,
                )
            else:  # pragma: no cover
                logger.info(
                    ("No variables added for unrecognized component {}.").format(comp)
                )

    return variable_set


def objective_thresolds_violation_check(
    model,
    lower_objective_threshold=None,
    upper_objective_threshold=None,
    zero_threshold=0.0,
):
    """
    Checks if the model current objective value violates thresholds.
    Sense dependent, if maximizing then check the lower_objective_thresold.
    If minimizing the check upper_objective_threshold.
    Zero threshold is functionally rounding tolerance.

    model: Pyomo model with an active objective
    lower_objective_threshold : float or None
        Sense dependent, used in maximization problems to check if
        objective < lower_objective_threshold.
        None indicates that a lower objective threshold will not
        be added to the model.
    upper_objective_threshold : float or None
        Sense dependent, used in minimization problems to check if
        objective > upper_objective_threshold.
        None indicates that a lower objective threshold will not
        be added to the model.
    :param zero_threshold: Description
    """
    orig_objective = get_active_objective(model)
    orig_objective_value = pyo.value(orig_objective)
    # MPV: current behavior here is to warn but return pool with no solutions added
    if lower_objective_threshold is not None:
        if (not orig_objective.is_minimizing()) and (
            orig_objective_value + zero_threshold <= lower_objective_threshold
        ):
            warnings.warn(
                "lower_objective_threshold violated at optimum, no valid solutions",
                category=RuntimeWarning,
                stacklevel=2,
            )
            return True
    if upper_objective_threshold is not None:
        if (orig_objective.is_minimizing()) and (
            orig_objective_value - zero_threshold >= upper_objective_threshold
        ):
            warnings.warn(
                "upper_objective_threshold violated at optimum, no valid solutions",
                category=RuntimeWarning,
                stacklevel=2,
            )
            return True
    return False
