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
from math import isnan, isinf

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

    def load_into_model(
        self,
        model: pyo.Model,
        *,
        value_overrides: dict[str, float | None] | None = None,
        descend_into: bool = True,
        skip_nan_inf: bool = True,
        error_if_value_missing: bool = False,
        track_missing: bool = True,
        track_fixed: bool = True,
        track_unfixed: bool = True,
        track_nan_inf: bool = True,
        fix_continuous: bool = False,
        fix_binary: bool = False,
        fix_integer: bool = False,
        fix_if_sol_var_fixed: bool = False,
        fix_var_names: set[str] | None = None,
        # check_assignment_domains: bool = False, #TODO: implement domain check
    ) -> MyMunch:
        """
        Loads into a Pyomo model the variable data from an or_topas Solution object.
        Variable fixing pattern can be subtle.
        Enforces pattern that all model variables are unfixed,
        then variables values may be fixed according to fix flag list

        **Important – fixing behavior**:
        - This method **unfixes all variables first**, regardless of their prior status in the model.
        - A variable is only fixed again if it matches **at least one** of:
        • category flags (`fix_continuous`, `fix_binary`, `fix_integer`)
        • name present in `fix_var_names`
        • `fix_if_sol_var_fixed=True` **and** solution marked it fixed
        - Previous model fix status is **never automatically preserved**.

        Returns Munch with members
        var_names_missing_values: None or set[str]
            None if track_missing == False,
            otherwise set of variable names that could not be assigned a value
            (missing from solution and no override provided or override was None)
        var_names_fixed: None or set[str]
            None if track_fixed == False,
            otherwise set of names of variables fixed
            (one of the flags indicated to fix this variable)
        var_names_unfixed: None or set[str]
            None if track_unfixed == False,
            otherwise set of names of variables unfixed
            (model variable was previously fixed and now is not)
        var_names_nan_inf: None or set[str]
            None if track_nan_inf == False
            otherwise set of names of variables with nan/inf values
            If skip_nan_inf, this is the set of variables skipped for this reason

        model: Pyomo model
            model to load solution values into
        value_overrides: dictionary
            keys are variable names to use corresponding key-value value for
        descend_into: boolean
            value True causes descent into all active blocks.
            value False treats only current block
        skip_nan_inf: boolean
            Handle NaN and Inf values by ignoring them
        error_if_value_missing: boolean
            flag to raise runtime error if variable not found in Solution object
        track_missing: boolean
            flag to track model variables without corresponding values
        track_fixed: boolean
            flag to track model variables fixed
        track_unfixed: boolean
            flag to track model variables unfixed
        fix_continuous: boolean
            flag to decide if continuous variables are fixed
        fix_binary: boolean
            flag to decide if binary variables are fixed
        fix_integer: boolean
            flag to decide if integer variables are fixed
            N.B. binary counts as integer for this flag
        fix_if_sol_var_fixed: boolean
            flag to decide to fix if model variable fixed in solution
        fix_var_names: None or set[str]
            set of variable names to fix model variable if in set
        check_assignment_domains: boolean
            flag to checking assignment value in variable domain
        """
        value_overrides = dict() if value_overrides is None else value_overrides
        fix_var_names = set() if fix_var_names is None else fix_var_names
        var_names_missing_values = set() if track_missing else None
        var_names_fixed = set() if track_fixed else None
        var_names_unfixed = set() if track_unfixed else None
        var_names_nan_inf = set() if track_nan_inf else None

        for model_var in model.component_data_objects(
            ctype=pyo.Var, descend_into=descend_into
        ):
            # handle all the fix if conditions that do not need the Solution object variable

            var_name = model_var.name
            was_fixed = model_var.is_fixed()
            model_var.unfix()
            need_to_fix_value = False
            is_nan_inf = False
            try:
                # get solution variable
                # this is what needs the error checking
                solution_var = self.variable(var_name)

                # handle solution variable specific fix check
                need_to_fix_value = fix_if_sol_var_fixed and solution_var.fixed

                # if in override map, use that, otherwise get from solution level variable
                var_value = value_overrides.get(var_name, solution_var.value)
            except (AssertionError, RuntimeError) as e:

                if error_if_value_missing:
                    raise RuntimeError(
                        f"Variable {var_name} has no value in Solution with id: {id(self)}"
                    )

                # error_if_value_missing is False block
                if track_missing:
                    # add missing var to tracking set
                    var_names_missing_values.add(var_name)

                if var_name in value_overrides:
                    # if there is an override value, use it
                    var_value = value_overrides.get(var_name)
                else:
                    # if we get here, there is not a solution variable for this var_name
                    # there is also not an override value
                    # so skip to next var
                    continue

            # treat var_value is None as missing
            if var_value is None:
                if track_missing:
                    var_names_missing_values.add(var_name)
                continue

            # if we get past the try/except, we have a value for the model_var
            if isnan(var_value) or isinf(var_value):
                if track_nan_inf and var_names_nan_inf is not None:
                    var_names_nan_inf.add(var_name)
                if skip_nan_inf:
                    continue

            # handle the remaining fix conditions
            need_to_fix_value = (
                need_to_fix_value
                or (var_name in fix_var_names)
                or (fix_continuous and model_var.is_continuous())
                or (fix_binary and model_var.is_binary())
                or (fix_integer and model_var.is_integer())
            )

            # #domain check
            # #ignores nan/inf values
            # if check_assignment_domains and not is_nan_inf:
            #     if not model_var.domain.contains(var_value):
            #         raise ValueError(  # or keep assert, but ValueError is friendlier
            #             f"Domain violation for {var_name}: "
            #             f"domain={model_var.domain}, value={var_value!r}"
            #             )
            if need_to_fix_value:
                # if need to fix variable, fix it to correct value
                model_var.fix(var_value)
                var_names_fixed.add(var_name)
            else:
                # if we get here, model_var does not need to be fixed
                # value loading can be done by assignment
                model_var.value = var_value
                if was_fixed and var_names_unfixed is not None:
                    var_names_unfixed.add(var_name)

        return MyMunch(
            var_names_missing_values=var_names_missing_values,
            var_names_fixed=var_names_fixed,
            var_names_unfixed=var_names_unfixed,
            var_names_nan_inf=var_names_nan_inf,
        )
