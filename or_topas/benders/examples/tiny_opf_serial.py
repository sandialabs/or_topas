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
import time
from or_topas.benders.benders_serial import (
    BendersGenerator_Serial as BendersCutGenerator,
)
import pyomo.environ as pyo

import itertools
from math import pi as pi_value
import pprint
import or_topas.benders.tests.test_cases as tc
gurobi_available = pyo.SolverFactory("gurobi_persistent").available(
    exception_flag=False
)

"""
EXAMPLE IN DEVELOPMENT
To run this example:

python tiny_opf_serial.py

mpirun -np X python tiny_opf_parallel.py
Where X is the number of processes, suggest using 2-3

If altering this script, note the more robust error handling available by using:
mpirun -np X python -m mpi4py tiny_opf_ex.py
Where X is again the number of processes.
This standardizes error handling especially for unexpected and unhandled errors

DC-Optimal Power Flow Example.
This is a feasibility-only Benders Decomposition example.
The second-stage contributes only to feasibility cuts.

Specific model and example details adapted from:
Viens, Skolfied, Hart, and Ferris "An Optimal Solution is Not Enough: Alternative Solutions and Optimal Power Systems"
https://arxiv.org/abs/2511.08805
"""


def main():
    assert gurobi_available, "This Example Requires the persistent gurobi solver"
    t0 = time.time()
    transform = "standard_lp"
    grid = tc.EnergyGrid()
    m = tc.EnergyGrid.create_root(grid=grid)
    root_vars = list(m.generation.values())
    m.benders = BendersCutGenerator()
    m.benders.set_input(
        root_vars=root_vars,
        tol=1e-8,
        transform=transform,
        allow_infeasible=True,
    )
    m.benders.add_subproblem(
        subproblem_fn=tc.EnergyGrid.create_subproblem,
        subproblem_fn_kwargs={"root": m, "grid": grid},
        root_eta=m.eta,
        subproblem_solver="gurobi_persistent",
    )
    for b in grid.buses:
        m.generation[b] = 0
    opt = pyo.SolverFactory("gurobi_persistent")
    opt.set_instance(m)

    print(
        "{0:<15}{1:<15}{2:<15}{3:<15}{4:<15}".format(
            "# Cuts", "Bus 1", "Bus 2", "Bus 3", "Time"
        )
    )
    for i in range(30):
        res = opt.solve(m, tee=False)
        cuts_added = m.benders.generate_cut()
        for c in cuts_added:
            opt.add_constraint(c)
        print(
            "{0:<15}{1:<15.2f}{2:<15.2f}{3:<15.2f}{4:<15.2f}".format(
                len(cuts_added),
                pyo.value(m.generation["bus1"]),
                pyo.value(m.generation["bus2"]),
                pyo.value(m.generation["bus3"]),
                time.time() - t0,
            )
        )
        if len(cuts_added) == 0:
            break


if __name__ == "__main__":
    main()
