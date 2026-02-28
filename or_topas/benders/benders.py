import sys
import pprint
import json
import copy
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
from or_topas.util import pyomo_utils

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
        self.default_relax_subproblem_complicating_vars = True
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
        relax_subproblem_complicating_vars = kwargs.get(
            "relax_subproblem_complicating_vars",
            self.default_relax_subproblem_complicating_vars,
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
        if relax_subproblem_complicating_vars:
            Benders_Abstract._relax_first_stage_var_copies(
                complicating_vars_map=complicating_vars_map
            )
        self.subproblems.append(subproblem)
        self.complicating_vars_maps.append(complicating_vars_map)
        b = subproblem
        # TODO: how is this any different from list(complicating_vars_map.keys())
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
            if "scip" in subproblem_solver:
                raise NotImplementedError(
                    "Unable to use SCIP as a subproblem solver due to SCIPAMPL interface not supporting LP dual information"
                )
            else:
                self.check_dual_info = False
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
    def _relax_first_stage_var_copies(complicating_vars_map):
        for subproblem_var in complicating_vars_map.values():
            subproblem_var.bounds = (None, None)

    @staticmethod
    def _fix_first_stage_var_copies(
        *, subproblem, root_vars, complicating_vars_map, mode=1
    ):
        """
        There are several ways to handle enforcing sub_var.val = root_var.val
        First is by adding constraints to enforce equality.
        Second is by fixing the values of the sub_var variables to root_var values.
        Third is by replacing the sub_var variables with parameters and setting the value of the parameters to the root_var values
        """
        if mode == 1:
            # method 1
            return Benders_Abstract._fix_first_stage_var_copies_by_constraint(
                subproblem=subproblem,
                root_vars=root_vars,
                complicating_vars_map=complicating_vars_map,
            )
        elif mode == 2:
            # method 2
            return Benders_Abstract._fix_first_stage_var_copies_by_variable_fixing(
                subproblem=subproblem,
                root_vars=root_vars,
                complicating_vars_map=complicating_vars_map,
            )
        elif mode == 3:
            return (
                Benders_Abstract._fix_first_stage_var_copies_by_parameter_value_setting(
                    subproblem=subproblem,
                    root_vars=root_vars,
                    complicating_vars_map=complicating_vars_map,
                )
            )
        else:
            raise RuntimeError(
                f"Asked for fix_first_Stage_var_copies {mode=} that does not exist"
            )

    @staticmethod
    def _fix_first_stage_var_copies_by_constraint(
        *, subproblem, root_vars, complicating_vars_map
    ):
        """
        There are several ways to handle enforcing sub_var.val = root_var.val
        One of the most direct is to directly add constraints that enforce it.
        This method handles the enforcement of the constraint based equality method.

        This requires that the values in the complicating_vars_map are variables.
        It then adds a constraint for each key-value pair where the constraint is of form
        for k_var, v_var:
            new_con = Constraint(v_var - k_var.value == 0)

        Returns a ComponentMap from the root variables as keys to the equality enforcing constraint
        """
        subproblem.fix_complicating_vars = pyo.ConstraintList()
        var_to_con_map = pyo.ComponentMap()

        for root_var, sub_var in complicating_vars_map.items():
            sub_var.set_value(root_var.value, skip_validation=True)
            new_con = subproblem.fix_complicating_vars.add(
                sub_var - root_var.value == 0
            )
            var_to_con_map[root_var] = new_con
        return var_to_con_map

    @staticmethod
    def _fix_first_stage_var_copies_by_variable_fixing(
        *, subproblem, root_vars, complicating_vars_map
    ):
        """
        There are several ways to handle enforcing sub_var.val = root_var.val
        This method handles the enforcement by fixing the sub_vars to the root_var value

        This requires that the values in the complicating_vars_map are variables.
        It then adds a constraint for each key-value pair where the constraint is of form
        for k_var, v_var:
            v_var.fix(k_var.value)

        Returns an empty ComponentMap since no constraints are added.
        This is generally meant to be used with solver_options={'treat_fixed_vars_as_params': True}
        In absence of that as_params flag, need custom way of handling dual information from variable bounds
        This is normally different from constraint duals, called reduced costs 'rc' suffix.
        """
        subproblem.fix_complicating_vars = pyo.ConstraintList()
        var_to_con_map = pyo.ComponentMap()
        for root_var, sub_var in complicating_vars_map.items():
            sub_var.fix(root_var.value)
        return var_to_con_map

    @staticmethod
    def _fix_first_stage_var_copies_by_parameter_value_setting(
        *, subproblem, root_vars, complicating_vars_map
    ):
        """
        There are several ways to handle enforcing sub_var.val = root_var.val
        This method handles the enforcement by fixing the parameter replacements of sub_vars to the root_var value
        Retuires parameters to be mutable

        This methods is meant for when the values in the complicating_vars_map are mutable parameters.
        It then adds a constraint for each key-value pair where the constraint is of form
        for k_var, v_param:
            v_param = k_var.value

        Returns an empty ComponentMap since no constraints are added.
        If used with values as variables, this will change level values but not gurantee fixing.
        """
        subproblem.fix_complicating_vars = pyo.ConstraintList()
        var_to_con_map = pyo.ComponentMap()
        for root_var, sub_param in complicating_vars_map.items():
            sub_param = root_var.value
        return var_to_con_map

    @staticmethod
    def _fix_eta_copies(*, subproblem, root_eta):
        subproblem.fix_eta = pyo.Constraint(expr=subproblem._eta - root_eta.value == 0)
        subproblem._eta.set_value(root_eta.value, skip_validation=True)

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

    # TODO: work in progress, no changes made
    @staticmethod
    def _standard_lp_subproblem_transform(*args, **kwargs):
        """
        The goal of this is to take a program of the form:
        min <p, x> + <q, y>
        Ax + By <= c
        Dx + Ey >= f
        Gx + Hy == l

        Where x are the first-stage (or 'master') problem variables,
        and y are the second-stage (or 'subproblem') problem variables

        And convert it to:
        min <p, x> + <q, y>
        By  <= c - Ax (alpha)
        -Ey <= Dx - f (beta)
        Hy  == l - Gx (gamma)

        So all the subproblem variables on the lhs, and constants/master variables on rhs.
        This is so the dual objective takes the form:
        <c - Ax, alpha> + <Dx - f, beta> + <l - Gx, gamma>

        Or otherwise formable as:
        sum_{c \in cons} c.val * c.dual

        This directly makes the forming of classical Benders optimality and feasibility cuts easier.
        """
        assert "b" in kwargs, "Need argument b in _standard_lp_subproblem_transform"
        assert (
            "root_vars" in kwargs
        ), "Need argument root_vars in _standard_lp_subproblem_transform"
        assert (
            "relax_subproblem_cons" in kwargs
        ), "Need argument relax_subproblem_cons in _standard_lp_subproblem_transform"
        assert (
            "complicating_vars_map" in kwargs
        ), "Need argument complicating_vars_map in _standard_lp_subproblem_transform"
        b = kwargs.get("b")
        root_vars = kwargs.get("root_vars")
        relax_subproblem_cons = kwargs.get("relax_subproblem_cons")
        complicating_vars_map = kwargs.get("complicating_vars_map")
        subproblem_master_vars = complicating_vars_map.values()
        # check for all of these b, root_vars, relax_subproblem_cons
        root_vars = ComponentSet(root_vars)

        objs = list(
            b.component_data_objects(pyo.Objective, descend_into=False, active=True)
        )
        if len(objs) != 1:
            raise ValueError("Subproblem must have exactly one objective")
        orig_obj = objs[0]

        # make sure dual vars are imported
        b.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        b.aux_cons = pyo.ConstraintList()
        b.aux_cons_rhs_exprs = []
        for c in list(
            b.component_data_objects(
                pyo.Constraint, descend_into=True, active=True, sort=True
            )
        ):
            if not relax_subproblem_cons:
                # TODO: MPV figure out what this code actually does
                # should this be root_vars here or subproblem_master_vars?
                # why would there be root versions in the subproblems?
                c_vars = ComponentSet(identify_variables(c.body, include_fixed=False))
                if not Benders_Abstract._any_common_elements(root_vars, c_vars):
                    continue

            body_split = pyomo_utils.split_expr(c.body, subproblem_master_vars)
            # TODO: there is a version of this where we check c.upper.is_constant or c.lower
            # and if it is, we don't need to split the expr, skipping that for now as we work on correctness

            # TODO: the new constraints are probably switched into a cannonical form
            # do we just need to make a map from which con in aux_cons constaint list as what is the RHS to form cuts?

            # so we have the reformatted constraints in aux_con as Wy+Tx - h <= 0, and fix_cons as x - x_general = 0
            # the single scenario cut then becomes dual_sign_conv*[sum(aux_con.dual[c]*pyo.value(aux_con_rhs[i]) for i,c in enumerate(aux_con)) + sum(fix_cons.dual[root_var_to_fix_con[rv]]*(rv-rv.value)) for rv in root_vars]
            # note that then the following evals to a scalar: sum(aux_con.dual[c]*pyo.value(aux_con_rhs[i]) for i,c in enumerate(aux_con))
            # this is then the part dealing with master problem vars: sum(fix_cons.dual[root_var_to_fix_con[rv]]*(rv-rv.value) for rv in root_vars)
            # we then have |root_vars| + 1 params to move around to form the multiple scenario cut when probablity weighted
            if c.equality:
                # in this case upper and lower eval to the same thing
                # so use upper
                # body.expr == upper.expr

                #TODO: if we know the constraint is cannonical, will c.upper have anything in it?

                #TODO: this needs to be lower now upper, need to fix

                upper_split = pyomo_utils.split_expr(c.upper, subproblem_master_vars)
                rhs = -body_split.in_plus_cons + upper_split.in_plus_cons
                lhs = body_split.out - upper_split.out
                b.aux_cons_rhs_exprs.append(rhs)
                b.aux_cons.add(lhs == rhs)
                Benders_Abstract._del_con(c)
            else:
                lower = pyo.value(c.lower)
                upper = pyo.value(c.upper)
                # TODO: if we know the constraint is cannonical, will c.upper/lower have anything non-zero in it?
                if upper is not None:
                    # case where upper has contents
                    # body.expr <= upper.expr

                    #TODO: upper and lower should both be constants
                    #don't need to split upper and lower, treat as constants

                    upper_split = pyomo_utils.split_expr(
                        c.upper, subproblem_master_vars
                    )
                    rhs = -body_split.in_plus_cons + upper_split.in_plus_cons
                    lhs = body_split.out - upper_split.out
                    b.aux_cons_rhs_exprs.append(rhs)
                    b.aux_cons.add(lhs <= rhs)
                if lower is not None:
                    # case where lower has contents
                    # lower.expr <= body.expr

                    #TODO: upper and lower should both be constants
                    #don't need to split upper and lower, treat as constants

                    lower_split = pyomo_utils.split_expr(
                        c.upper, subproblem_master_vars
                    )
                    rhs = body_split.in_plus_cons - lower_split.in_plus_cons
                    lhs = -body_split.out + lower_split.in_plus_cons
                    b.aux_cons_rhs_exprs.append(rhs)
                    b.aux_cons.add(lhs <= rhs)
                Benders_Abstract._del_con(c)

        # b.obj_con = pyo.Constraint(expr=orig_obj_expr - b._eta - b._z <= 0)


"""
    def evaluate(y):
        return (optimal_x, optimal_y, dual_alpha, dual_beta)
    def create_cut(dual_alpha, dual_beta):
        return cut(...)
    
    def generate_cut(y):
        _,_,dual_alpha, dual_beta = evaluate(y)
        return create_cut
"""
