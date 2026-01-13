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

import pyomo.opt
import pyomo.environ as pyo
from pyomo.common.dependencies import attempt_import
import pyomo.common.unittest as unittest

import or_topas.util as au
from or_topas.solnpool import PyomoSolution
from or_topas.aos import enumerate_binary_solutions

solvers = list(pyomo.opt.check_available_solvers("glpk", "gurobi"))

parameterized, param_available = attempt_import("parameterized")
if not param_available:
    raise unittest.SkipTest("Parameterized is not available.")
parameterized = parameterized.parameterized


class TestSolutionUnit(unittest.TestCase):

    def get_model(self):
        """
        Simple model with all variable types and fixed variables to test the
        Solution code.
        """
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.NonNegativeReals)
        m.y = pyo.Var(domain=pyo.Binary)
        m.z = pyo.Var(domain=pyo.NonNegativeIntegers)
        m.f = pyo.Var(domain=pyo.Reals)

        m.f.fix(1)
        m.obj = pyo.Objective(expr=m.x + m.y + m.z + m.f, sense=pyo.maximize)

        m.con_x = pyo.Constraint(expr=m.x <= 1.5)
        m.con_y = pyo.Constraint(expr=m.y <= 1)
        m.con_z = pyo.Constraint(expr=m.z <= 3)
        return m

    @parameterized.expand(input=solvers)
    def test_solution(self, mip_solver):
        """
        Create a Solution Object, call its functions, and ensure the correct
        data is returned.
        """
        model = self.get_model()
        opt = pyo.SolverFactory(mip_solver)
        opt.solve(model)
        all_vars = au.pyomo_utils.get_model_variables(model, include_fixed=False)
        obj = au.pyomo_utils.get_active_objective(model)

        solution = PyomoSolution(variables=all_vars, objective=obj)
        sol_str = """{
    "id": null,
    "objectives": [
        {
            "index": 0,
            "name": "obj",
            "suffix": {},
            "value": 6.5
        }
    ],
    "suffix": {},
    "variables": [
        {
            "discrete": false,
            "fixed": false,
            "index": 0,
            "name": "x",
            "suffix": {},
            "value": 1.5
        },
        {
            "discrete": true,
            "fixed": false,
            "index": 1,
            "name": "y",
            "suffix": {},
            "value": 1
        },
        {
            "discrete": true,
            "fixed": false,
            "index": 2,
            "name": "z",
            "suffix": {},
            "value": 3
        }
    ]
}"""
        assert str(solution) == sol_str

        all_vars = au.pyomo_utils.get_model_variables(model, include_fixed=True)
        solution = PyomoSolution(variables=all_vars, objective=obj)
        sol_str = """{
    "id": null,
    "objectives": [
        {
            "index": 0,
            "name": "obj",
            "suffix": {},
            "value": 6.5
        }
    ],
    "suffix": {},
    "variables": [
        {
            "discrete": false,
            "fixed": false,
            "index": 0,
            "name": "x",
            "suffix": {},
            "value": 1.5
        },
        {
            "discrete": true,
            "fixed": false,
            "index": 1,
            "name": "y",
            "suffix": {},
            "value": 1
        },
        {
            "discrete": true,
            "fixed": false,
            "index": 2,
            "name": "z",
            "suffix": {},
            "value": 3
        },
        {
            "discrete": false,
            "fixed": true,
            "index": 3,
            "name": "f",
            "suffix": {},
            "value": 1
        }
    ]
}"""
        assert solution.to_string() == sol_str

        sol_vars = solution.variable_name_to_index
        assert len(sol_vars) == len(solution._variables)
        assert set(sol_vars.keys()) == {"x", "y", "z", "f"}
        
        for key, value in sol_vars.items():
            #checks accuracy of variable_name_to_index map
            assert solution._variables[value].name == key
            #checks accuracy of variable() method for integer index
            assert id(solution.variable(value)) == id(solution._variables[value])
            #checks accuracy of variable() method for string inputs
            assert id(solution.variable(key)) == id(solution._variables[value])
            #checks accuracy of variable() method for objects with .name attribute
            temp = solution._variables[value]
            assert id(solution.variable(temp)) == id(solution._variables[value])
            #checks accuracy of _variable_by_name() method for string inputs
            assert id(solution._variable_by_name(key)) == id(solution._variables[value])
            #checks accuracy of _variable_by_name() method for objects with .name attribute
            temp = solution._variables[value]
            assert id(solution._variable_by_name(temp)) == id(solution._variables[value])
        # old solution.fixed_variable_names functionality replaced with map
        sol_vars_fixed = solution.fixed_variable_indices
        fixed_variable_names = map(
            lambda x: solution._variables[x].name, sol_vars_fixed
        )
        assert set(fixed_variable_names) == {"f"}

    @parameterized.expand(input=solvers)
    def test_soln_order(self, mip_solver):
        """ """
        values = [10, 9, 2, 1, 1]
        weights = [10, 9, 2, 1, 1]

        K = len(values)
        capacity = 12

        m = pyo.ConcreteModel()
        m.x = pyo.Var(range(K), within=pyo.Binary)
        m.o = pyo.Objective(
            expr=sum(values[i] * m.x[i] for i in range(K)), sense=pyo.maximize
        )
        m.c = pyo.Constraint(
            expr=sum(weights[i] * m.x[i] for i in range(K)) <= capacity
        )

        solns = enumerate_binary_solutions(
            m, num_solutions=10, solver=mip_solver, abs_opt_gap=0.5
        )
        assert len(solns) == 4
        assert [[v.value for v in soln.variables()] for soln in sorted(solns)] == [
            [0, 1, 1, 0, 1],
            [0, 1, 1, 1, 0],
            [1, 0, 0, 1, 1],
            [1, 0, 1, 0, 0],
        ]


if __name__ == "__main__":
    unittest.main()
