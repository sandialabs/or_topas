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

from or_topas.benders.benders_serial import (
    BendersGenerator_Serial as BendersCutGenerator,
)
import pyomo.environ as pyo
import time
import pprint

"""
To run this example:

mpirun -np X python newsvendor_ex.py
Where X is the number of processes, suggest using 2-3

If altering this script, note the more robust error handling available by using:
mpirun -np X python -m mpi4py newsvendor_ex.py
Where X is again the number of processes.
This standardizes error handling especially for unexpected and unhandled errors

A Tutorial on Stochastic Programming
Alexander Shapiro∗ and Andy Philpott†
March 21, 2007
https://www.epoc.org.nz/papers/ShapiroTutorialSP.pdf
"""


#
# Data for a simple newsvendor example
#
class Newsvendor:
    def __init__(self):
        self.c = 1.0
        self.b = 1.5
        self.h = 0.1
        self.scenario_demand = {1: 15, 2: 60, 3: 72, 4: 78, 5: 82}
        self.scenarios = self.scenario_demand.keys()
        self.scenario_probabilities = {
            i: 1 / len(self.scenarios) for i in self.scenarios
        }


# creates benders master problem model for newsvendor
def create_root(newsvendor):
    # print(newsvendor)
    M = pyo.ConcreteModel()

    # need to initialize in root since x not in objective or constraints without Benders Cuts
    # if not initialized, can take value None, which causes issues in the benders_cut code
    M.x = pyo.Var(bounds=(0.0, None), initialize=0.0)
    M.scenarios = pyo.Set(initialize=newsvendor.scenarios, ordered=True)

    M.eta = pyo.Var(M.scenarios)

    M.obj = pyo.Objective(expr=sum(M.eta.values()))

    for s in M.scenarios:
        # using max demand as 100, so worst case is either buy 100 sell none (cost+holding), or buy 0 demand 100 (shortfall)
        # lower bound is -max_demand*max{c+h,b}
        # since scenario weighting is done in subproblems (not done in master objective above), need to weight bounds below
        M.eta[s].setlb(
            -1
            * newsvendor.scenario_probabilities[s]
            * 100
            * max(newsvendor.c + newsvendor.h, newsvendor.b)
        )
    return M


def create_subproblem(root, newsvendor, scenario):
    M = pyo.ConcreteModel()

    M.x = pyo.Var(within=pyo.NonNegativeReals)

    b = newsvendor.b
    c = newsvendor.c
    h = newsvendor.h
    d = newsvendor.scenario_demand[scenario]

    M.y = pyo.Var()
    M.greater = pyo.Constraint(expr=M.y >= (c - b) * M.x + b * d)
    M.less = pyo.Constraint(expr=M.y >= (c + h) * M.x - h * d)

    M.obj = pyo.Objective(expr=newsvendor.scenario_probabilities[scenario] * M.y)

    complicating_vars_map = pyo.ComponentMap()
    complicating_vars_map[root.x] = M.x

    return M, complicating_vars_map


def main():

    t0 = time.time()
    newsvendor = Newsvendor()
    m = create_root(newsvendor=newsvendor)
    root_vars = list(m.x.values())
    m.benders = BendersCutGenerator()
    m.benders.set_input(root_vars=root_vars, tol=1e-8)
    for s in newsvendor.scenarios:
        subproblem_fn_kwargs = dict()
        subproblem_fn_kwargs["root"] = m
        subproblem_fn_kwargs["newsvendor"] = newsvendor
        subproblem_fn_kwargs["scenario"] = s
        m.benders.add_subproblem(
            subproblem_fn=create_subproblem,
            subproblem_fn_kwargs=subproblem_fn_kwargs,
            root_eta=m.eta[s],
            subproblem_solver="gurobi_persistent",
        )
    # pprint.pprint(m)
    # m.pprint()
    opt = pyo.SolverFactory("gurobi_persistent")
    opt.set_instance(m)

    print("{0:<15}{1:<15}{2:<15}".format("# Cuts", "x", "Time"))
    for i in range(30):
        res = opt.solve(tee=False, save_results=False)
        # if i == 0:
        #    print(f"Solver Status: {str(res.solver.status)}")
        #    print(f"Termination Condition: {str(res.solver.termination_condition)}")
        #    m.pprint()
        cuts_added = m.benders.generate_cut()
        for c in cuts_added:
            # c.pprint()
            opt.add_constraint(c)
        # TODO add time deltas from last cut
        print(
            "{0:<15}{1:<15.2f}{2:<15.2f}".format(
                len(cuts_added), pyo.value(m.x), time.time() - t0
            )
        )
        if len(cuts_added) == 0:
            break


if __name__ == "__main__":
    main()
