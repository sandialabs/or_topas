import logging

from pyomo.common.collections import ComponentSet
from pyomo.common.dependencies import (
    mpi4py,
    mpi4py_available,
    numpy as np,
    numpy_available,
)
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.core.base.block import BlockData, declare_custom_block
import pyomo.environ as pyo
from .benders import Benders_Abstract

logger = logging.getLogger(__name__)


@declare_custom_block(name="BendersGenerator_Serial")
class Benders_Serial(Benders_Abstract):
    # TODO: this is for generate multi-subproblem cut
    # This Serial solver is designed to closely mirror the parallel solver as much as possible
    # to ease testing and moving back and forth between serial and parallel tools.

    def __init__(self, component):
        if not numpy_available:
            raise ImportError("BendersGenerator_Serial requires numpy.")
        super().__init__(component)
        self.transform_to_cut_map = {
            "feasibility": Benders_Serial.generate_cut_feasibility_transform,
            "standard_lp": Benders_Serial.generate_cut_standard_lp_transform,
        }
        self.default_transform_name = "feasibility"
        self.num_subproblems_by_rank = 0  # np.zeros(self.comm.Get_size())
        self.all_root_etas = list()
        # map from ndx in self.subproblems (local) to the global subproblem ndx
        self._subproblem_ndx_map = dict()

    def global_num_subproblems(self):
        return self.local_num_subproblems()

    def local_num_subproblems(self):
        return len(self.subproblems)

    def set_input(self, *args, **kwargs):
        """
        It is very important for root_vars to be in the same order for every process.

        Parameters
        ----------
        root_vars
        tol
        """

        # if kwargs.get("comm", None) is not None:
        #     self.comm = kwargs.get("comm")
        # else:
        #     self.comm = MPI.COMM_WORLD
        # self.num_subproblems_by_rank = np.zeros(self.comm.Get_size())
        self.num_subproblems_by_rank = np.zeros(1)
        super().set_input(*args, **kwargs)
        self.all_root_etas = list()
        self._subproblem_ndx_map = dict()

    def add_subproblem(
        self,
        subproblem_fn,
        subproblem_fn_kwargs,
        root_eta,
        subproblem_solver="gurobi_persistent",
        relax_subproblem_cons=False,
    ):
        _rank = np.argmin(self.num_subproblems_by_rank)
        self.num_subproblems_by_rank[_rank] += 1
        self.all_root_etas.append(root_eta)
        # if _rank == self.comm.Get_rank():
        super().add_subproblem(
            subproblem_fn=subproblem_fn,
            subproblem_fn_kwargs=subproblem_fn_kwargs,
            root_eta=root_eta,
            subproblem_solver=subproblem_solver,
            relax_subproblem_cons=relax_subproblem_cons,
        )
        self._subproblem_ndx_map[len(self.subproblems) - 1] = (
            self.global_num_subproblems() - 1
        )

    def generate_cut_feasibility_transform(self):
        coefficients = np.zeros(
            self.global_num_subproblems() * len(self.root_vars), dtype="d"
        )
        constants = np.zeros(self.global_num_subproblems(), dtype="d")
        eta_coeffs = np.zeros(self.global_num_subproblems(), dtype="d")

        for local_subproblem_ndx in range(len(self.subproblems)):
            subproblem = self.subproblems[local_subproblem_ndx]
            global_subproblem_ndx = self._subproblem_ndx_map[local_subproblem_ndx]
            coeff_ndx = global_subproblem_ndx * len(self.root_vars)

            #
            # solve the feasibility subproblem
            #
            results_munch = self._solve_feasibility_subproblem(
                subproblem=subproblem,
                local_subproblem_ndx = local_subproblem_ndx,
            )

            #
            # Pull out data from results munch
            #
            
            subproblem_constant= results_munch.subproblem_constant
            subproblem_eta = results_munch.subproblem_eta
            subproblem_coeff= results_munch.subproblem_coeff

            #
            # Put data in data sharing resources
            #

            constants[global_subproblem_ndx] = subproblem_constant

            for coeff in subproblem_coeff:
                coefficients[coeff_ndx] = coeff
                coeff_ndx += 1

            eta_coeffs[global_subproblem_ndx] = subproblem_eta

        #
        # Cut formation logic
        #
        #
        # N.B. the inclusion of the global vars and Allreduce commands under comments are intetional
        # this is meant to show the connection to the parallel version
        #

        total_num_subproblems = self.global_num_subproblems()
        # global_constants = np.zeros(total_num_subproblems, dtype="d")
        # global_coeffs = np.zeros(total_num_subproblems * len(self.root_vars), dtype="d")
        # global_eta_coeffs = np.zeros(total_num_subproblems, dtype="d")

        # comm = self.comm
        # comm.Allreduce([constants, MPI.DOUBLE], [global_constants, MPI.DOUBLE])
        # comm.Allreduce([eta_coeffs, MPI.DOUBLE], [global_eta_coeffs, MPI.DOUBLE])
        # comm.Allreduce([coefficients, MPI.DOUBLE], [global_coeffs, MPI.DOUBLE])

        global_constants = constants
        global_coeffs = coefficients
        global_eta_coeffs = eta_coeffs

        global_constants = [float(i) for i in global_constants]
        global_coeffs = [float(i) for i in global_coeffs]
        global_eta_coeffs = [float(i) for i in global_eta_coeffs]

        coeff_ndx = 0
        cuts_added = list()
        #
        # Form cuts
        #
        for global_subproblem_ndx in range(total_num_subproblems):
            cut_expr = global_constants[global_subproblem_ndx]
            #check if cut needed for this subproblem
            if cut_expr > self.tol:
                #
                # add needed cut
                #
                root_eta = self.all_root_etas[global_subproblem_ndx]
                cut_expr -= global_eta_coeffs[global_subproblem_ndx] * (
                    root_eta - root_eta.value
                )
                for root_var in self.root_vars:
                    coeff = global_coeffs[coeff_ndx]
                    cut_expr -= coeff * (root_var - root_var.value)
                    coeff_ndx += 1
                new_cut = self.cuts.add(cut_expr <= 0)
                cuts_added.append(new_cut)
            else:
                coeff_ndx += len(self.root_vars)

        return cuts_added

    def generate_cut_standard_lp_transform(self):
        #setup global data
        coefficients = np.zeros(
            self.global_num_subproblems() * len(self.root_vars), dtype="d"
        )
        constants = np.zeros(self.global_num_subproblems(), dtype="d")
        subproblem_etas = np.zeros(self.global_num_subproblems(), dtype="d")

        for local_subproblem_ndx in range(len(self.subproblems)):
            #set up subproblem data
            subproblem = self.subproblems[local_subproblem_ndx]
            global_subproblem_ndx = self._subproblem_ndx_map[local_subproblem_ndx]
            coeff_ndx = global_subproblem_ndx * len(self.root_vars)
            subproblem_solver = self.subproblem_solvers[local_subproblem_ndx]
            complicating_vars_map = self.complicating_vars_maps[local_subproblem_ndx]
            root_eta = self.root_etas[local_subproblem_ndx]

            var_to_con_map = Benders_Abstract._fix_first_stage_var_copies(
                subproblem=subproblem,
                root_vars=self.root_vars,
                complicating_vars_map=complicating_vars_map,
            )

            sign_convention = Benders_Abstract._get_solver_sign_convention(
                solver_name=subproblem_solver.name
            )

            # handle subproblem solve

            # added_constraints = fix_complicating_vars.values()
            res = Benders_Abstract._update_and_solve_model(
                subproblem=subproblem,
                subproblem_solver=subproblem_solver,
                added_constraints=subproblem.fix_complicating_vars.values(),
            )

            # TODO: this is the breakpoint for where to split the solve subproblem from collect subproblem data and compute cut

            #
            # Start of collect subproblem data for subproblem global_subproblem_ndx
            #

            # so we have the reformatted constraints in aux_con as Wy+Tx - h <= 0, and fix_cons as x - x_general = 0
            # the single scenario cut then becomes dual_sign_conv*[sum(aux_con.dual[c]*pyo.value(aux_con_rhs[i]) for i,c in enumerate(aux_con)) + sum(fix_cons.dual[root_var_to_fix_con[rv]]*(rv-rv.value)) for rv in root_vars]
            # note that then the following evals to a scalar: sum(aux_con.dual[c]*pyo.value(aux_con_rhs[i]) for i,c in enumerate(aux_con))
            # this is then the part dealing with master problem vars: sum(fix_cons.dual[root_var_to_fix_con[rv]]*(rv-rv.value) for rv in root_vars)
            # we then have |root_vars| + 2 params to move around to form cut and cut needed (obj val)

            # so the coefficients come from the fixed first stage variables
            # so coefficients[coeff_ndx] = fix_cons.dual[root_var_to_fix_con[rv]], where coeff_ndx = global_subproblem_ndx * len(self.root_vars) + (i for i, v in enumerate(root_var_to_fix_con.keys() if v == rv)

            
            #
            # Subproblem data collection
            #

            # the constants are come from the sum of everything else for the subproblem as:
            # constants[global_subproblem_ndx] = sum(aux_con.dual[c]*pyo.value(aux_con_rhs[i]) for i,c in enumerate(aux_con))
            # all terms need to include the subproblem solver sign convention

            subproblem_constant = -sign_convention * sum(
                subproblem.dual[subproblem.aux_cons[c]]
                * pyo.value(subproblem.aux_cons_rhs_exprs[i])
                for i, c in enumerate(subproblem.aux_cons)
            )

            subproblem_eta = pyo.value(subproblem.orig_obj_expr)

            subproblem_coeff = np.zeros(len(self.root_vars), dtype="d")
            temp_ndx = 0
            for root_var, c in var_to_con_map.items():
                subproblem_coeff[temp_ndx] = sign_convention * pyo.value(
                    subproblem.dual[c]
                )
                temp_ndx += 1

            if isinstance(subproblem_solver, PersistentSolver):
                for c in subproblem.fix_complicating_vars.values():
                    subproblem_solver.remove_constraint(c)
            del subproblem.fix_complicating_vars

            #old code
            # for root_var, c in var_to_con_map.items():
            #     coefficients[coeff_ndx] = sign_convention * pyo.value(
            #         subproblem.dual[c]
            #     )
            #     coeff_ndx += 1

            for coeff in subproblem_coeff:
                coefficients[coeff_ndx] = coeff
                coeff_ndx += 1

            # the constants are come from the sum of everything else for the subproblem as:
            # constants[global_subproblem_ndx] = sum(aux_con.dual[c]*pyo.value(aux_con_rhs[i]) for i,c in enumerate(aux_con))
            # all terms need to include the subproblem solver sign convention


            # constants[global_subproblem_ndx] = -sign_convention * sum(
            #     subproblem.dual[subproblem.aux_cons[c]]
            #     * pyo.value(subproblem.aux_cons_rhs_exprs[i])
            #     for i, c in enumerate(subproblem.aux_cons)
            # )

            constants[global_subproblem_ndx] = subproblem_constant

            # eta term directly comes from subproblem objective
            # subproblem_etas[global_subproblem_ndx] = pyo.value(subproblem.orig_obj_expr)
            subproblem_etas[global_subproblem_ndx] = subproblem_eta

            # remove the added fixing constraints for fix_complicating vars
            # if isinstance(subproblem_solver, PersistentSolver):
            #     for c in subproblem.fix_complicating_vars.values():
            #         subproblem_solver.remove_constraint(c)
            # del subproblem.fix_complicating_vars

            # end of compute,
            # this is where to return for solve subproblem, collect data, but don't generate cuts yet

        total_num_subproblems = self.global_num_subproblems()
        # global_constants = np.zeros(total_num_subproblems, dtype="d")
        # global_coeffs = np.zeros(total_num_subproblems * len(self.root_vars), dtype="d")
        # global_subproblem_etas = np.zeros(total_num_subproblems, dtype="d")

        # comm = self.comm
        # comm.Allreduce([constants, MPI.DOUBLE], [global_constants, MPI.DOUBLE])
        # comm.Allreduce([coefficients, MPI.DOUBLE], [global_coeffs, MPI.DOUBLE])
        # comm.Allreduce([subproblem_etas, MPI.DOUBLE], [global_subproblem_etas, MPI.DOUBLE])

        global_constants = constants
        global_coeffs = coefficients
        global_subproblem_etas = subproblem_etas

        global_constants = [float(i) for i in global_constants]
        global_coeffs = [float(i) for i in global_coeffs]
        global_subproblem_etas = [float(i) for i in global_subproblem_etas]

        global_eta_gaps = [
            abs(pyo.value(self.root_etas[i]) - v)
            for i, v in enumerate(global_subproblem_etas)
        ]

        # TODO: why do we need all threads computing the same cuts?
        # shouldn't this be blocked behind a rank check?
        # This is a point for parallel, not serial

        coeff_ndx = 0
        cuts_added = list()
        for global_subproblem_ndx in range(total_num_subproblems):
            cut_expr = global_constants[global_subproblem_ndx]
            eta_gap = global_eta_gaps[global_subproblem_ndx]
            if eta_gap > self.tol:
                # difference above tolerance, add cut
                root_eta = self.all_root_etas[global_subproblem_ndx]
                for root_var in self.root_vars:
                    coeff = global_coeffs[coeff_ndx]

                    # transform case 1 for cut building, see transform details in Benders_Abstract
                    cut_expr -= coeff * root_var

                    # transform case 2 for cut building
                    # cut_expr -= coeff * (root_var - root_var.value)

                    coeff_ndx += 1
                new_cut = self.cuts.add(cut_expr <= root_eta)
                cuts_added.append(new_cut)
            else:
                # skip cut, update ndx counter
                coeff_ndx += len(self.root_vars)

        return cuts_added

    def evaluate_all_subproblems(self):
        # take the x information from root problem in parent block
        raise NotImplementedError(
            "Benders_Serial does not have evaluate_all_subproblems"
        )

    def evaluate_single_subproblem(self, index):
        raise NotImplementedError(
            "Benders_Serial does not have evaluate_single_subproblem"
        )

    # need a create cut
    def generate_single_subproblem_cut(self, index):
        raise NotImplementedError(
            "Benders_Serial does not have generate_single_subproblem_cut"
        )

    def generate_all_subproblem_cut(self):
        if self.transform in self.transform_to_cut_map:
            return self.transform_to_cut_map[self.transform](self)
        else:
            raise NotImplementedError(
                f"Benders_Serial does not have {self.transform=} implemented"
            )

    def generate_cut(self):
        return self.generate_all_subproblem_cut()
