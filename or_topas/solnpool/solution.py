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

import sys
import heapq
import collections
import dataclasses
import json
import functools

import pyomo.environ as pyo

from or_topas.util.mymunch import MyMunch, to_dict

nan = float("nan")


def _custom_dict_factory(data):
    return {k: to_dict(v) for k, v in data}


if sys.version_info >= (3, 10):
    dataclass_kwargs = dict(kw_only=True)
else:
    dataclass_kwargs = dict()


@dataclasses.dataclass(**dataclass_kwargs)
class VariableInfo:
    """
    Represents a variable in a solution.

    Attributes
    ----------
    value : float
        The value of the variable.
    fixed : bool
        If True, then the variable was fixed during optimization.
    name : str
        The name of the variable.
    index : int
        The unique identifier for this variable.
    discrete : bool
        If True, then this is a discrete variable
    suffix : dict
        Other information about this variable.
    """

    value: float = nan
    fixed: bool = False
    name: str = None
    repn = None
    index: int = None
    discrete: bool = False
    suffix: MyMunch = dataclasses.field(default_factory=MyMunch)

    def to_dict(self):
        return dataclasses.asdict(self, dict_factory=_custom_dict_factory)


@dataclasses.dataclass(**dataclass_kwargs)
class ObjectiveInfo:
    """
    Represents an objective in a solution.

    Attributes
    ----------
    value : float
        The objective value.
    name : str
        The name of the objective.
    index : int
        The unique identifier for this objective.
    suffix : dict
        Other information about this objective.
    """

    #
    # TODO: add ability to keep track of if objective is active
    #

    value: float = nan
    name: str = None
    index: int = None
    suffix: MyMunch = dataclasses.field(default_factory=MyMunch)

    def to_dict(self):
        return dataclasses.asdict(self, dict_factory=_custom_dict_factory)


@functools.total_ordering
class Solution:
    """
    An object that describes an optimization solution.

    Parameters
    -----------
    variables : None or list
        A list of :py:class:`VariableInfo` objects. (default is None)
    objective : None or :py:class:`ObjectiveInfo`
        A :py:class:`ObjectiveInfo` object. (default is None)
    objectives : None or list
        A list of :py:class:`ObjectiveInfo` objects. (default is None)
    kwargs : dict
        A dictionary of auxiliary data that is stored with the core solution values.  If the 'suffix'
        keyword is specified, then its value is use to define suffix data.  Otherwise, all
        of the keyword arguments are treated as suffix data.
    """

    def __init__(
        self,
        *,
        variables=None,
        objective=None,
        objectives=None,
        use_given_index_maps=False,
        keep_fixed_var_list=True,
        variable_name_to_index=None,
        fixed_variable_indices=None,
        **kwargs,
    ):
        if objective is not None and objectives is not None:
            raise ValueError(
                "The objective= and objectives= keywords cannot both be specified."
            )
        self.id = None
        self.index_maps_used_elsewhere = False

        self._variables = []
        self.index_maps_from_external = use_given_index_maps
        self.keep_fixed_var_list = keep_fixed_var_list

        # load the variable data
        if variables is not None:
            self._variables = variables

        # logic to control where variable_name_to_index and fixed_variable_indicies
        #    data comes from. 2 possible cases:
        #    Case 1. created external to this solution instance and used here
        # .   Case 2. created in this solution instance
        # Note, there is also an option to not keep track of fixed_variable_indicies
        # this is handled in both cases.
        if self.index_maps_from_external:
            # case 1: created external to this solution instance and used here

            # assertation check that the user actually gave variable_name_to_index data
            assert (
                variable_name_to_index is not None
            ), f"Attempted to create solution using external index maps without passing variable_name_to_index map"
            self.variable_name_to_index = variable_name_to_index

            # keep track of fixed_variable_indicies check
            if self.keep_fixed_var_list:
                assert (
                    fixed_variable_indices is not None
                ), f"Attempted to create solution using external index maps without passing fixed_variable_indices set"
                self.fixed_variable_indices = fixed_variable_indices

        else:
            # case 2: created in this solution instance
            self.variable_name_to_index = {}
            self.fixed_variable_indices = set()
            self._rebuild_indices_maps()

            # _rebuild_indicies_map builds fixed_variable_indices by default
            # in case 2, we can optionally throw it out
            if self.keep_fixed_var_list == False:
                self.fixed_variable_indices = None

        self._objectives = []
        self.name_to_objective = {}
        if objective is not None:
            objectives = [objective]
        if objectives is not None:
            self._objectives = objectives
            for o in objectives:
                if getattr(o, "name", None) is not None:
                    self.name_to_objective[o.name] = o

        if "suffix" in kwargs:
            self.suffix = MyMunch(kwargs.pop("suffix"))
        else:
            self.suffix = MyMunch(**kwargs)

    def _rebuild_indices_maps(
        self, error_if_maps_used_elsewhere=False, rebuild_in_place=False
    ):
        if self.index_maps_used_elsewhere:
            message_text = f"Rebuilding index maps used elsewhere, {rebuild_in_place=} in Solution with id {id(self)}"
            if error_if_maps_used_elsewhere:
                raise RuntimeError(message_text)
            else:
                raise RuntimeWarning(message_text)
        if rebuild_in_place:
            self.variable_name_to_index.clear()
            if self.keep_fixed_var_list:
                self.fixed_variable_indices.clear()
        else:
            self.variable_name_to_index = {}
            self.fixed_variable_indices = set()
            self.index_maps_used_elsewhere = False
        for i, v in enumerate(self._variables):
            if v.name is not None:
                if self.keep_fixed_var_list and v.fixed:
                    self.fixed_variable_indices.add(i)
                self.variable_name_to_index[v.name] = i

    def variable(self, index=0, map_consistency_check=False):
        """Returns the specified variable.

        Parameters
        ----------
        index : int, str, object
            The index or name of the variable if directly known. (default is 0)
            May also pass in an object with a .name attribute, which will be used if available
            If .name attribute does not exist, to_string method will be attempted.

        Raises
        ------
        AssertationError, if invalid index used
        RuntimeError, if using name based lookup and name map inconsistent

        Returns
        -------
        VariableInfo
        """
        if type(index) is int:
            assert (
                0 <= index < len(self._variables)
            ), f"Index {index} is invalid in Solution with id {id(self)}"
            return self._variables[index]
        else:
            return self._variable_by_name(
                name=index, map_consistency_check=map_consistency_check
            )

    def _variable_by_name(self, name, map_consistency_check=False):
        """Returns the specified variable.

        Parameters
        ----------
        index : str or object
            The name of the variable if directly known.
            May also pass in an object with a .name attribute, which will be used if available

        Raises
        ------
        AssertationError, if invalid index used in maps based off name
        RuntimeError, if using name based lookup and name map inconsistent

        Returns
        -------
        VariableInfo
        """
        if isinstance(name, str):
            variable_name = name
        elif hasattr(name, "name"):
            variable_name = name.name
        else:
            raise RuntimeError(
                f"Index {name} is invalid in Solution with id {id(self)}"
            )
        assert (
            variable_name in self.variable_name_to_index
        ), f"Key {variable_name} is not a valid key in the variable_name_to_index map in Solution with id {id(self)}"
        assert (
            0 <= self.variable_name_to_index[variable_name] < len(self._variables)
        ), f"Index {self.variable_name_to_index[variable_name]} corresponding to key {variable_name} is not a valid variable list index in Solution with id {id(self)}"
        solution_variable_info = self._variables[
            self.variable_name_to_index[variable_name]
        ]
        if map_consistency_check and solution_variable_info.name != variable_name:
            # present design assumes consistency on variable_name_to_index maps across solutions
            # this consistency check will detect violations of that assumption.
            # current use defaults to error in this case, another option is to rebuild the local mappings
            raise RuntimeError(
                f"Mismatch between input variable name, {variable_name}, and mapped to variable, {solution_variable_info.name} in Solution with id {id(self)}"
            )
        return solution_variable_info

    def variables(self):
        """
        Returns
        -------
        list
            The list of variables in the solution.
        """
        return self._variables

    def objective(self, index=0):
        """Returns the specified objective.

        Parameters
        ----------
        index : int or str
            The index or name of the objective. (default is 0)

        Returns
        -------
        :py:class:`ObjectiveInfo`
        """
        if type(index) is int:
            return self._objectives[index]
        else:
            return self.name_to_objective[index]

    def objectives(self):
        """
        Returns
        -------
        list
            The list of objectives in the solution.
        """
        return self._objectives

    def to_dict(self):
        """
        Returns
        -------
        dict
            A dictionary representation of the solution.
        """
        return dict(
            id=self.id,
            variables=[v.to_dict() for v in self.variables()],
            objectives=[o.to_dict() for o in self.objectives()],
            suffix=self.suffix.to_dict(),
        )

    def to_string(self, sort_keys=True, indent=4):
        """
        Returns a string representation of the solution, which is generated
        from a dictionary representation of the solution.

        Parameters
        ----------
        sort_keys : bool
            If True, then sort the keys in the dictionary representation. (default is True)
        indent : int
            Specifies the number of whitespaces to indent each element of the dictionary.

        Returns
        -------
        str
            A string representation of the solution.
        """
        return json.dumps(self.to_dict(), sort_keys=sort_keys, indent=indent)

    def __str__(self):
        return self.to_string()

    __repn__ = __str__

    def _tuple_repn(self):
        """
        Generate a tuple that represents the variables in the model.

        We use string names if possible, because they more explicit than the integer index values.
        """
        if len(self.variable_name_to_index) == len(self._variables):
            return tuple(tuple([var.name, var.value]) for var in self._variables)
        else:
            return tuple(tuple([k, var.value]) for k, var in enumerate(self._variables))

    def __eq__(self, soln):
        if not isinstance(soln, Solution):
            return NotImplemented
        return self._tuple_repn() == soln._tuple_repn()

    def __lt__(self, soln):
        if not isinstance(soln, Solution):
            return NotImplemented
        return self._tuple_repn() <= soln._tuple_repn()


# TODO: we need to extend this to for a SparowSolution
# that makes the sparow as_solution method simply
# def _as_sparow_solution(*args, **kwargs):
#     return SparowSolution(*args, **kwargs)
class PyomoSolution(Solution):

    def __init__(self, *, variables=None, objective=None, objectives=None, **kwargs):
        #
        # Q: Do we want to use an index relative to the list of variables specified here?  Or use the Pyomo variable ID?
        # Q: Should this object cache the Pyomo variable object?  Or CUID?
        #
        # TODO: Capture suffix info here.
        #
        vlist = []
        if variables is not None:
            index = 0
            for var in variables:
                vlist.append(
                    VariableInfo(
                        value=(
                            pyo.value(var)
                            if var.is_continuous()
                            else round(pyo.value(var))
                        ),
                        fixed=var.is_fixed(),
                        name=str(var),
                        index=index,
                        discrete=not var.is_continuous(),
                    )
                )
                index += 1
        else:
            raise (RuntimeWarning("variable data was None"))
        #
        # TODO: Capture suffix info here.
        #
        if objective is not None:
            objectives = [objective]
        olist = []
        #
        # TODO: Add some way here of keeping track of what objective is active
        # Suggest reordering objective list so active objectives put at front of list
        # So if only one objective active it goes as olist[0]
        #
        if objectives is not None:
            index = 0
            for obj in objectives:
                olist.append(
                    ObjectiveInfo(value=pyo.value(obj), name=str(obj), index=index)
                )
                index += 1

        super().__init__(variables=vlist, objectives=olist, **kwargs)

    # TODO: update to have solution come from self not external
    # convert to instance method
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
