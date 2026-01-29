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
    aos_block,
    objective,
    objective_value,
    rel_opt_gap=None,
    abs_opt_gap=None,
    level_value=None,
):
    """
    Adds a relative and/or absolute objective function constraint to the
    specified block.
    """
    try:
        if level_value is not None:
            level_value = float(level_value)
    except ValueError:
        raise ValueError(f"level_value ({level_value}) must be None or numeric")
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
            aos_block.optimality_tol_rel = pyo.Constraint(
                expr=objective_expr <= objective_cutoff
            )
        else:
            aos_block.optimality_tol_rel = pyo.Constraint(
                expr=objective_expr >= objective_cutoff
            )
        objective_constraints.append(aos_block.optimality_tol_rel)

    if abs_opt_gap is not None:
        objective_cutoff = objective_value + objective_sense * abs_opt_gap

        if objective_is_min:
            aos_block.optimality_tol_abs = pyo.Constraint(
                expr=objective_expr <= objective_cutoff
            )
        else:
            aos_block.optimality_tol_abs = pyo.Constraint(
                expr=objective_expr >= objective_cutoff
            )
        objective_constraints.append(aos_block.optimality_tol_abs)

    # TODO: third level value change
    if level_value is not None:
        if objective_is_min:
            aos_block.level_val_bound = pyo.Constraint(
                expr=objective_expr <= level_value
            )
        else:
            aos_block.level_val_bound = pyo.Constraint(
                expr=objective_expr >= level_value
            )
        objective_constraints.append(aos_block.level_val_bound)

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


def load_solution_into_model(
    model,
    solution,
    solution_override_values_dict=dict(),
    descend_into=True,
    error_if_value_missing=False,
    return_vars_missing_values=True,
    fix_continuous=False,
    fix_binary=False,
    fix_integer=False,
    fix_if_model_var_fixed=False,
    fix_if_sol_var_fixed=False,
    vars_to_fix_set=set(),
):
    """
    Docstring for load_solution_into_model
    Loads into a Pyomo model value data from an or_topas Solution object
    returns vars_missing_values_set and vars_fixed_set
    vars_missing_values_set is None if return_vars_missing_values = False, otherwise set of variables unable to be assigned values
    vars_fixed_set is set of variables fixed

    :param model: Pyomo model to load solution values into
    :param solution: or_topas Solution object to load values from
    :param solution_override_values_dict: dictionary, keys are variables to use corresponding key-value value for
    :param descend_into: boolean to decide to decend blocks
    :param error_if_value_missing: boolean to raise runtime error if variable not found in Solution object
    :param return_vars_missing_values: boolean to choose if model variables without corresponding values returned
    :param fix_continuous: boolean to decide if continuous variables are fixed
    :param fix_binary: boolean to decide if binary variables are fixed
    :param fix_integer: boolean to decide if integer variables are fixed
    :param fix_if_model_var_fixed: boolean to decide to fix if model variable fixed in model
    :param fix_if_sol_var_fixed: boolean to decide to fix if model variable fixed in solution
    :param vars_to_fix_set: set of variables to fix model variable if in set
    """
    if return_vars_missing_values:
        vars_missing_values_set = {}
    else:
        vars_missing_values_set = None
    vars_fixed_set = {}

    for model_var in model.component_data_objects(
        ctype=pyo.Var, descend_into=descend_into
    ):
        # handle all the fix if conditions that do not need the Solution object variable

        var_name = model_var.name
        try:
            solution_var = solution.get(var_name)
        except RuntimeError as e:
            if error_if_value_missing:
                raise RuntimeError(
                    f"Variable {var_name} has no value in Solution with id: {id(solution)}"
                )
            if return_vars_missing_values:
                vars_missing_values_set.add(model_var)
        # if we get past the try/except, we have the solution level variable as solution_var

        # if in override map, use that, otherwise get from solution level variable
        var_value = solution_override_values_dict.get(model_var, solution_var.value)

        # handle all the need to fix if conditions
        need_to_fix_value = (
            (model_var in vars_to_fix_set)
            or (fix_continuous and model_var.is_continuous())
            or (fix_binary and model_var.is_binary())
            or (fix_integer and model_var.is_integer())
            or (fix_if_model_var_fixed and model_var.is_fixed())
            or (fix_if_sol_var_fixed and solution_var.fixed)
        )

        if need_to_fix_value:
            # if need to fix variable, fix it to correct value
            model_var.fix(var_value)
            vars_fixed_set.add(model_var)
        else:
            # if we get here, model_var does not need to be fixed
            # value loading can be done by assignment
            model_var = var_value

    return vars_missing_values_set, vars_fixed_set


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
