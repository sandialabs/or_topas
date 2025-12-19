import sys
import pprint
import json
import copy
import munch
import logging

import logging

from pyomo.common.collections import ComponentSet
from pyomo.common.dependencies import (
    mpi4py,
    mpi4py_available,
    numpy as np,
    numpy_available,
)
from pyomo.core.base.block import BlockData, declare_custom_block
from pyomo.core.expr.visitor import identify_variables
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver

import pyomo.environ as pyo


# TODO: do we have a topas logger?
logger = logging.getLogger(__name__)


# @declare_custom_block(name="BendersGenerator_Abstract")
class Benders_Abstract(BlockData):
    solver_dual_sign_convention = dict(
        ipopt=-1,
        gurobi=-1,
        gurobi_direct=-1,
        gurobi_persistent=-1,
        cplex=-1,
        cplex_direct=-1,
        cplexdirect=-1,
        cplex_persistent=-1,
        glpk=-1,
        cbc=-1,
        xpress_direct=-1,
        highs=-1,
    )
    default_transform_name = "default"

    def __init__(self, component):
        BlockData.__init__(self, component)
        self.transform_map = {
            Benders_Abstract.default_transform_name: Benders_Abstract._feasibility_subproblem_transform,
            "feasibility": Benders_Abstract._feasibility_subproblem_transform,
        }
        self.default_subproblem_solver = "gurobi_persistent"
        self.default_transform_name = Benders_Abstract.default_transform_name
        self.default_relax_subproblem_cons = False
        self.subproblems = list()
        self.complicating_vars_maps = list()
        self.root_vars = list()
        self.root_vars_indices = pyo.ComponentMap()
        self.root_etas = list()
        self.cuts = None
        self.subproblem_solvers = list()
        self.tol = None

    # TODO: what methods do we want here
    def set_input(self, *args, **kwargs):
        """
        It is very important for root_vars to be in the same order for every process.

        Parameters
        ----------
        root_vars
        tol
        """
        assert "root_vars" in kwargs, "Need argument root_vars in set_input"
        root_vars = kwargs.get("root_vars")

        # TODO: do we want to add a warning when cuts are deleted?
        # we could return it?
        del self.cuts
        self.cuts = pyo.ConstraintList()
        self.subproblems = list()
        self.root_etas = list()
        self.complicating_vars_maps = list()
        self.root_vars = list(root_vars)
        self.root_vars_indices = pyo.ComponentMap()
        self.transform = kwargs.get("transform", self.default_transform_name)
        for i, v in enumerate(self.root_vars):
            self.root_vars_indices[v] = i
        self.tol = kwargs.get("tol", 1e-6)
        self.subproblem_solvers = list()

    def add_subproblem(self, *args, **kwargs):
        # old required arguments, we will want these to all be kwargs now
        # subproblem_fn,
        # subproblem_fn_kwargs,
        # root_eta,
        # subproblem_solver="gurobi_persistent",
        # relax_subproblem_cons=False,
        assert (
            "subproblem_fn" in kwargs
        ), "Need argument subproblem_fn in add_subproblem"
        assert (
            "subproblem_fn_kwargs" in kwargs
        ), "Need argument subproblem_fn_kwargs in add_subproblem"
        assert "root_eta" in kwargs, "Need argument root_eta in add_subproblem"
        subproblem_fn = kwargs.get("subproblem_fn")
        subproblem_fn_kwargs = kwargs.get("subproblem_fn_kwargs")
        root_eta = kwargs.get("root_eta")
        subproblem_solver = kwargs.get(
            "subproblem_solver", self.default_subproblem_solver
        )
        relax_subproblem_cons = kwargs.get(
            "relax_subproblem_cons", self.default_relax_subproblem_cons
        )

        # parallel specific code
        # so comment out
        # _rank = np.argmin(self.num_subproblems_by_rank)
        # self.num_subproblems_by_rank[_rank] += 1
        # self.all_root_etas.append(root_eta)
        # if _rank == self.comm.Get_rank():
        # in parallel code, everything else was indented
        # not parallel specific code
        self.root_etas.append(root_eta)
        subproblem, complicating_vars_map = subproblem_fn(**subproblem_fn_kwargs)
        self.subproblems.append(subproblem)
        self.complicating_vars_maps.append(complicating_vars_map)
        b = subproblem
        root_vars = [
            complicating_vars_map[i]
            for i in self.root_vars
            if i in complicating_vars_map
        ]
        relax_subproblem_cons = relax_subproblem_cons
        self._setup_subproblem(
            b=b,
            root_vars=root_vars,
            relax_subproblem_cons=relax_subproblem_cons,
        )
        # parallel specific code
        # this also does not impact the general code below
        # so it can be commented out
        # self._subproblem_ndx_map[len(self.subproblems) - 1] = (
        #     self.global_num_subproblems() - 1
        # )

        # general code
        if isinstance(subproblem_solver, str):
            subproblem_solver = pyo.SolverFactory(subproblem_solver)
        self.subproblem_solvers.append(subproblem_solver)
        if isinstance(subproblem_solver, PersistentSolver):
            subproblem_solver.set_instance(subproblem)

    # need transform
    # need evaluate a model (possibly several)
    def evaluate_all_subproblems(self):
        # TODO: we can always evaluate given a transform
        # it may or may not make a ton of sense but we should always be able to do it
        # take the x information from root problem in parent block
        raise NotImplementedError(
            "Inheriting classes must implement evaluate_all_subproblems"
        )

    def evaluate_single_subproblem(self, index):
        raise NotImplementedError(
            "Inheriting classes must implement evaluate_single_subproblem"
        )

    # need a create cut
    def generate_single_subproblem_cut(self, index):
        raise NotImplementedError(
            "Inheriting classes must implement generate_single_subproblem_cut"
        )

    def generate_all_subproblem_cut(self):
        raise NotImplementedError(
            "Inheriting classes must implement generate_all_subproblem_cut"
        )

    def generate_cut(self):
        return self.generate_all_subproblem_cut()

    #
    # helper methods go here
    #

    @staticmethod
    def _del_con(c):
        parent = c.parent_component()
        if parent.is_indexed():
            parent.__delitem__(c.index())
        else:
            assert parent is c
            c.parent_block().del_component(c)

    @staticmethod
    def _any_common_elements(a, b):
        if len(a) < len(b):
            for i in a:
                if i in b:
                    return True
        else:
            for i in b:
                if i in a:
                    return True
        return False

    @staticmethod
    def _relax_first_stage_var_copies():
        pass

    @staticmethod
    def _fix_first_stage_var_copies():
        pass

    @staticmethod
    def _fix_eta_copies():
        pass

    #
    # Transform methods go here
    #

    # Need transform to standard form
    # TODO: convert this from the parallel transform to a standard form
    # Probably want it as min f(x,y) s.t. Wy = h - Tx, By <= q - Ax
    # TODO: need to check how the general linear cuts work for this format
    def _setup_subproblem(self, *args, **kwargs):
        # default params, b, root_vars, relax_subproblem_cons, transform = 'feasibility'
        if self.transform in self.transform_map:
            self.transform_map[self.transform](*args, **kwargs)
        else:
            raise NotImplementedError(f"{self.transform=} is not implemented")

    @staticmethod
    def _feasibility_subproblem_transform(*args, **kwargs):
        assert "b" in kwargs, "Need argument b in _feasibility_subproblem_transform"
        assert (
            "root_vars" in kwargs
        ), "Need argument root_vars in _feasibility_subproblem_transform"
        assert (
            "relax_subproblem_cons" in kwargs
        ), "Need argument relax_subproblem_cons in _feasibility_subproblem_transform"
        b = kwargs.get("b")
        root_vars = kwargs.get("root_vars")
        relax_subproblem_cons = kwargs.get("relax_subproblem_cons")
        # check for all of these b, root_vars, relax_subproblem_cons
        # first get the objective and turn it into a constraint
        root_vars = ComponentSet(root_vars)

        objs = list(
            b.component_data_objects(pyo.Objective, descend_into=False, active=True)
        )
        if len(objs) != 1:
            raise ValueError("Subproblem must have exactly one objective")
        orig_obj = objs[0]
        orig_obj_expr = orig_obj.expr
        b.del_component(orig_obj)

        b._z = pyo.Var(bounds=(0, None))
        b.objective = pyo.Objective(expr=b._z)
        b.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        b._eta = pyo.Var()

        b.aux_cons = pyo.ConstraintList()
        for c in list(
            b.component_data_objects(
                pyo.Constraint, descend_into=True, active=True, sort=True
            )
        ):
            if not relax_subproblem_cons:
                c_vars = ComponentSet(identify_variables(c.body, include_fixed=False))
                if not Benders_Abstract._any_common_elements(root_vars, c_vars):
                    continue
            if c.equality:
                body = c.body
                rhs = pyo.value(c.lower)
                body -= rhs
                b.aux_cons.add(body - b._z <= 0)
                b.aux_cons.add(-body - b._z <= 0)
                Benders_Abstract._del_con(c)
            else:
                body = c.body
                lower = pyo.value(c.lower)
                upper = pyo.value(c.upper)
                if upper is not None:
                    body_upper = body - upper - b._z
                    b.aux_cons.add(body_upper <= 0)
                if lower is not None:
                    body_lower = body - lower
                    body_lower = -body_lower
                    body_lower -= b._z
                    b.aux_cons.add(body_lower <= 0)
                Benders_Abstract._del_con(c)

        b.obj_con = pyo.Constraint(expr=orig_obj_expr - b._eta - b._z <= 0)


"""
    def evaluate(y):
        return (optimal_x, optimal_y, dual_alpha, dual_beta)
    def create_cut(dual_alpha, dual_beta):
        return cut(...)
    
    def generate_cut(y):
        _,_,dual_alpha, dual_beta = evaluate(y)
        return create_cut
"""
