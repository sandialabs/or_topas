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

# from or_topas.benders.benders_cuts import BendersCutGenerator
from or_topas.benders.benders_parallel import (
    BendersGenerator_Parallel as BendersCutGenerator,
)

# from or_topas.benders.benders_serial import (
#     BendersGenerator_Serial as BendersCutGenerator,
# )

import pyomo.environ as pyo

"""
This example presently is designed to work with gurobi_persistent and ipopt.
Adapted from the Pyomo.contrib.Benders Grothey problem to work with new parallel solver.
Note that this only has one subproblem, so there is limited pratical value in running in parallel.

To run this example:

mpirun -np 1 python grothey_parallel.py

If altering this script, note the more robust error handling available by using:
mpirun -np X python -m mpi4py grothey_parallel.py
Where X is again the number of processes.
This standardizes error handling especially for unexpected and unhandled errors
"""


def create_root():
    m = pyo.ConcreteModel()
    m.y = pyo.Var(bounds=(1, None))
    m.eta = pyo.Var(bounds=(-10, None))
    m.obj = pyo.Objective(expr=m.y**2 + m.eta)
    return m


def create_subproblem(root):
    m = pyo.ConcreteModel()
    m.x1 = pyo.Var()
    m.x2 = pyo.Var()
    m.y = pyo.Var()
    m.obj = pyo.Objective(expr=-m.x2)
    m.c1 = pyo.Constraint(expr=(m.x1 - 1) ** 2 + m.x2**2 <= pyo.log(m.y))
    m.c2 = pyo.Constraint(expr=(m.x1 + 1) ** 2 + m.x2**2 <= pyo.log(m.y))

    complicating_vars_map = pyo.ComponentMap()
    complicating_vars_map[root.y] = m.y

    return m, complicating_vars_map


def main():
    m = create_root()
    root_vars = [m.y]
    m.benders = BendersCutGenerator()
    m.benders.set_input(root_vars=root_vars, tol=1e-8)
    m.benders.add_subproblem(
        subproblem_fn=create_subproblem,
        subproblem_fn_kwargs={"root": m},
        root_eta=m.eta,
        subproblem_solver="ipopt",
    )
    opt = pyo.SolverFactory("gurobi_direct")

    for i in range(30):
        res = opt.solve(m, tee=False)
        cuts_added = m.benders.generate_cut()
        print(len(cuts_added), m.y.value, m.eta.value)
        if len(cuts_added) == 0:
            break


if __name__ == "__main__":
    main()
