#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
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

    def __init__(self, *, variables=None, objective=None, objectives=None, **kwargs):
        if objective is not None and objectives is not None:
            raise ValueError(
                "The objective= and objectives= keywords cannot both be specified."
            )
        self.id = None
        self.index_maps_used_elsewhere = False

        self._variables = []
        self.index_maps_from_external = kwargs.get("use_given_index_maps", False)
        if self.index_maps_from_external:
            name_index_map_string = "variable_name_to_index"
            assert (
                name_index_map_string in kwargs
            ), f"Attempted to create solution using external index maps without passing {name_index_map_string} map"
            self.variable_name_to_index = kwargs["variable_name_to_index"]
            fixed_variable_indicies_string = "fixed_variable_names"
            assert (
                name_index_map_string in kwargs
            ), f"Attempted to create solution using external index maps without passing {fixed_variable_indicies_string} map"
            self.fixed_variable_indices = kwargs["fixed_variable_indicies_string"]
        else:
            self.variable_name_to_index = {}
            self.fixed_variable_indices = set()
            if variables is not None:
                self._variables = variables
            self._rebuild_indices_maps()

        self._objectives = []
        self.name_to_objective = {}
        if objective is not None:
            objectives = [objective]
        if objectives is not None:
            self._objectives = objectives
            for o in objectives:
                if o.name is not None:
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
            self.fixed_variable_indices.clear()
        else:
            self.variable_name_to_index = {}
            self.fixed_variable_indices = set()
            self.index_maps_used_elsewhere = False
        for i, v in enumerate(self._variables):
            if v.name is not None:
                if v.fixed:
                    self.fixed_variable_indices.add(i)
                self.variable_name_to_index[v.name] = i

    def variable(self, index, map_consistency_check=False):
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
            The index or name of the variable if directly known. (default is 0)
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
        ), f"{variable_name=} is not a valid key in {self.variable_name_to_index=} in Solution with id {id(self)}"
        assert (
            0 <= self.variable_name_to_index[variable_name] < len(self._variables)
        ), f"Index {self.variable_name_to_index[variable_name]} corresponding to {variable_name=} is not a valid variable list index in Solution with id {id(self)}"
        solution_variable_info = self._variables[
            self.variable_name_to_index[variable_name]
        ]
        if map_consistency_check and solution_variable_info.name != variable_name:
            # present design assumes consistency on variable_name_to_index maps across solutions
            # this consistency check will detect violations of that assumption.
            # current use defaults to error in this case, another option is to rebuild the local mappings
            raise RuntimeError(
                f"Mismatch between input variable name, {variable_name}, and mapped to variable, {solution_variable_info} in Solution with id {id(self)}"
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

        #
        # TODO: Capture suffix info here.
        #
        if objective is not None:
            objectives = [objective]
        olist = []
        if objectives is not None:
            index = 0
            for obj in objectives:
                olist.append(
                    ObjectiveInfo(value=pyo.value(obj), name=str(obj), index=index)
                )
                index += 1

        super().__init__(variables=vlist, objectives=olist, **kwargs)
