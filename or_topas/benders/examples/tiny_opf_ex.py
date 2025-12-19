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

"""
EXAMPLE IN DEVELOPMENT
To run this example:

mpirun -np X python tiny_opf_ex.py
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


class Grid:
    def __init__(self):
        # model data

        # we can change these three parameters to get a unique value
        # originally not designed to be unique

        # TODO: revisit these values to make a non-trivial feasibility cut
        # will likely need to change flow limits
        # system solves with this data (always feasible)

        # original values
        self.load_dict = {"bus1": 0, "bus2": 0, "bus3": 100}
        self.gen_max_dict = {"bus1": 100, "bus2": 100, "bus3": 0}
        self.cost_dict = {"bus1": 50, "bus2": 50, "bus3": 0}
        self.flow_bounds_dict = {
            ("bus1", "bus2"): 100,
            ("bus1", "bus3"): 100,
            ("bus2", "bus3"): 100,
        }

        # fails with this data, feasible set gen_bus2 = 100, flow_2_to_3 = 100
        # TODO: confirm with the nondecomposed script that this solves with this data
        # self.load_dict = {"bus1": 0, "bus2": 0, "bus3": 100}
        # self.gen_max_dict = {"bus1": 50, "bus2": 100, "bus3": 0}
        # self.cost_dict = {"bus1": 50, "bus2": 50, "bus3": 0}
        # self.flow_bounds_dict = {
        #     ("bus1", "bus2"): 0,
        #     ("bus1", "bus3"): 0,
        #     ("bus2", "bus3"): 100,
        # }

        # rest of data

        self.susceptance_dict = {
            ("bus1", "bus2"): 100,
            ("bus1", "bus3"): 100,
            ("bus2", "bus3"): 100,
        }

        self.flow_bounds_dict = {
            # ("bus1", "bus2"): 100,
            ("bus1", "bus2"): 0,
            # ("bus1", "bus3"): 100,
            ("bus1", "bus3"): 0,
            ("bus2", "bus3"): 100,
        }
        self.theta_bounds_dict = {"bus1": pi_value, "bus2": pi_value, "bus3": pi_value}
        self.buses = ["bus1", "bus2", "bus3"]
        self.lines = itertools.combinations(self.buses, 2)
        # {('bus1', 'bus2'), ('bus1', 'bus3'), ('bus2', 'bus3')}


# DC-Optimal Power Flow Example
# Adapted from TPEC paper example model
# mode 0 is Copper Plate
# mode 1 is Network Flow
# mode 2 is DC-OPF with flow variables
def create_tiny_opf(grid, mode=2):
    # print(f"{grid=}")
    # print(f"{vars(grid)=}")
    # print(f"{grid.buses=}")
    assert mode in [0, 1, 2], "Needs to be mode 0, 1, or 2"
    if mode == 0:
        print("Model in Copper Plate Mode")
    elif mode == 1:
        print("Model in Network Flow Mode")
    elif mode == 2:
        print("Model in DC-OPF")
    # model creation
    model = pyo.ConcreteModel()
    model.buses = pyo.Set(initialize=grid.buses)

    # always on components
    model.loads = pyo.Param(model.buses, initialize=grid.load_dict)
    model.capacity = pyo.Param(model.buses, initialize=grid.gen_max_dict)
    model.costs = pyo.Param(model.buses, initialize=grid.cost_dict)

    def gen_max_rule(m, bus):
        return (0, m.capacity[bus])

    model.generation = pyo.Var(
        model.buses, domain=pyo.PositiveReals, bounds=gen_max_rule
    )

    def obj_rule(m):
        return pyo.summation(m.costs, m.generation)

    model.obj = pyo.Objective(rule=obj_rule)

    # copper plate mode
    if mode == 0:
        model.copper_plate_balance = pyo.Constraint(
            expr=sum(model.generation[i] for i in model.buses)
            - sum(model.loads[i] for i in model.buses)
            == 0
        )
    else:
        # Network flow or DC OPF Mode
        model.lines = pyo.Set(initialize=grid.lines)
        model.flow_max = pyo.Param(model.lines, initialize=grid.flow_bounds_dict)

        def NodesOut_init(m, node):
            for i, j in m.lines:
                if i == node:
                    yield j

        model.NodesOut = pyo.Set(model.buses, initialize=NodesOut_init)

        def NodesIn_init(m, node):
            for i, j in m.lines:
                if j == node:
                    yield i

        model.NodesIn = pyo.Set(model.buses, initialize=NodesIn_init)

        def flow_max_rule(m, i, j):
            return (-m.flow_max[i, j], m.flow_max[i, j])

        model.flows = pyo.Var(model.lines, bounds=flow_max_rule)
        # where is the flow max rule???

        def FlowBalance_rule(m, node):
            return (
                m.generation[node]
                + sum(m.flows[i, node] for i in m.NodesIn[node])
                - m.loads[node]
                - sum(m.flows[node, j] for j in m.NodesOut[node])
                == 0
            )

        model.FlowBalance = pyo.Constraint(model.buses, rule=FlowBalance_rule)

        if mode == 2:
            # DC-OPF Mode
            model.theta_max = pyo.Param(model.buses, initialize=grid.theta_bounds_dict)

            def theta_max_rule(m, bus):
                return (-m.theta_max[bus], m.theta_max[bus])

            model.angles = pyo.Var(model.buses, bounds=theta_max_rule)
            model.susceptance = pyo.Param(model.lines, initialize=grid.susceptance_dict)

            # what happens when flow on line is zero, this enforces a_i = a_j when f_{i,j} = 0
            def power_rule(m, *line):
                return m.flows[line] == m.susceptance[line] * (
                    m.angles[line[1]] - m.angles[line[0]]
                )

            model.PowerBalance = pyo.Constraint(model.lines, rule=power_rule)
    return model


def create_root(grid):
    # If first-stage is generation and second-stage is flow and angles
    # the first-stage model is just copper-plate
    # print('Creating Root')
    m = create_tiny_opf(grid=grid, mode=0)
    # add unbounded var as eta placeholder since this is feasibility only
    # need to initialize to trivial value to avoid None issues
    m.eta = pyo.Var(initialize=0.0)
    # print('Done Creating Root')
    return m


def create_subproblem(root, grid):
    # the second-stage model is full DC-OPF
    # print('Creating Subproblem')
    m = create_tiny_opf(grid=grid, mode=2)

    complicating_vars_map = pyo.ComponentMap()
    for b in grid.buses:
        complicating_vars_map[root.generation[b]] = m.generation[b]
    # print('Done creating subproblem')
    return m, complicating_vars_map


def main():
    t0 = time.time()
    grid = Grid()
    m = create_root(grid=grid)
    root_vars = list(m.generation.values())
    m.benders = BendersCutGenerator()
    m.benders.set_input(root_vars=root_vars, tol=1e-8)
    m.benders.add_subproblem(
        subproblem_fn=create_subproblem,
        subproblem_fn_kwargs={"root": m, "grid": grid},
        root_eta=m.eta,
        subproblem_solver="gurobi_persistent",
    )
    opt = pyo.SolverFactory("gurobi_persistent")
    opt.set_instance(m)

    print(
        "{0:<15}{1:<15}{2:<15}{3:<15}{4:<15}".format(
            "# Cuts", "Bus 1", "Bus 2", "Bus 3", "Time"
        )
    )
    for i in range(30):
        res = opt.solve(m, tee=False)
        # if i == 0:
        #     print(f"Solver Status: {str(res.solver.status)}")
        #     print(f"Termination Condition: {str(res.solver.termination_condition)}")
        #     m.pprint()
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
