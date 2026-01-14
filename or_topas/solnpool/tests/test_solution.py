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
from or_topas.solnpool import PyomoSolution, Solution
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
        
        #test objective lookup by index
        assert str(solution.objective(0).value) == '6.5'
        #test objective lookup by objective name
        assert str(solution.objective('obj').value) == '6.5'

        #test eq comparison
        assert solution.__eq__('test_string') is NotImplemented
        #test lt comparison
        assert solution.__lt__('test_string') is NotImplemented

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
            # checks accuracy of variable_name_to_index map
            assert solution._variables[value].name == key
            # checks accuracy of variable() method for integer index
            assert id(solution.variable(value)) == id(solution._variables[value])
            # checks accuracy of variable() method for string inputs
            assert id(solution.variable(key)) == id(solution._variables[value])
            # checks accuracy of variable() method for objects with .name attribute
            temp = solution._variables[value]
            assert id(solution.variable(temp)) == id(solution._variables[value])
            # checks accuracy of _variable_by_name() method for string inputs
            assert id(solution._variable_by_name(key)) == id(solution._variables[value])
            # checks accuracy of _variable_by_name() method for objects with .name attribute
            temp = solution._variables[value]
            assert id(solution._variable_by_name(temp)) == id(
                solution._variables[value]
            )
        # old solution.fixed_variable_names functionality replaced with map
        sol_vars_fixed = solution.fixed_variable_indices
        fixed_variable_names = map(
            lambda x: solution._variables[x].name, sol_vars_fixed
        )
        assert set(fixed_variable_names) == {"f"}

        solution = PyomoSolution(variables=all_vars, objective=obj, keep_fixed_var_list = False)
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
        sol_vars = solution.variable_name_to_index
        assert len(sol_vars) == len(solution._variables)
        assert set(sol_vars.keys()) == {"x", "y", "z", "f"}
        assert solution.fixed_variable_indices == None


    @parameterized.expand(input=solvers)
    def test_solution_rebuild_indices_maps(self, mip_solver):
        """
        Create a Solution Object, call its functions, and ensure the correct
        data is returned.
        """
        model = self.get_model()
        opt = pyo.SolverFactory(mip_solver)
        opt.solve(model)

        all_vars = au.pyomo_utils.get_model_variables(model, include_fixed=True)
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
        sol_vars_id_1 = id(sol_vars)
        assert len(sol_vars) == len(solution._variables)
        assert set(sol_vars.keys()) == {"x", "y", "z", "f"}

        for key, value in sol_vars.items():
            # checks accuracy of variable_name_to_index map
            assert solution._variables[value].name == key
        # old solution.fixed_variable_names functionality replaced with map
        sol_vars_fixed = solution.fixed_variable_indices
        fixed_variable_names = map(
            lambda x: solution._variables[x].name, sol_vars_fixed
        )
        assert set(fixed_variable_names) == {"f"}
        fix_var_id_1 = id(solution.fixed_variable_indices)

        sol_vars.clear()
        sol_vars_fixed.clear()
        solution._rebuild_indices_maps(rebuild_in_place=True)
        sol_vars = solution.variable_name_to_index
        sol_vars_id_2 = id(sol_vars)
        assert len(sol_vars) == len(solution._variables)
        assert set(sol_vars.keys()) == {"x", "y", "z", "f"}

        for key, value in sol_vars.items():
            # checks accuracy of variable_name_to_index map
            assert solution._variables[value].name == key
        # old solution.fixed_variable_names functionality replaced with map
        sol_vars_fixed = solution.fixed_variable_indices
        fixed_variable_names = map(
            lambda x: solution._variables[x].name, sol_vars_fixed
        )
        assert set(fixed_variable_names) == {"f"}
        fix_var_id_2 = id(solution.fixed_variable_indices)
        assert sol_vars_id_1 == sol_vars_id_2
        assert fix_var_id_1 == fix_var_id_2

        # testing not in place rebuild
        # ids should be different between x_old and x for both sol_vars and sol_vars_fixed
        # adding explicit forcing of solution.index_maps_used_elsewhere = False even though this is default
        # no error should occur
        sol_vars_old = sol_vars
        sol_vars_fixed_old = sol_vars_fixed
        solution.index_maps_used_elsewhere = False
        solution._rebuild_indices_maps(rebuild_in_place=False)
        sol_vars = solution.variable_name_to_index
        sol_vars_id_3 = id(sol_vars)
        assert len(sol_vars) == len(solution._variables)
        assert set(sol_vars.keys()) == {"x", "y", "z", "f"}

        for key, value in sol_vars.items():
            # checks accuracy of variable_name_to_index map
            assert solution._variables[value].name == key
        # old solution.fixed_variable_names functionality replaced with map
        sol_vars_fixed = solution.fixed_variable_indices
        fixed_variable_names = map(
            lambda x: solution._variables[x].name, sol_vars_fixed
        )
        assert set(fixed_variable_names) == {"f"}
        fix_var_id_3 = id(solution.fixed_variable_indices)
        assert sol_vars_id_1 != sol_vars_id_3
        assert fix_var_id_1 != fix_var_id_3

        solution.index_maps_used_elsewhere = True
        rebuild_in_place = False
        expected_message = f"Rebuilding index maps used elsewhere, {rebuild_in_place=} in Solution with id {id(solution)}"
        with self.assertRaises(RuntimeError) as cm:
            solution._rebuild_indices_maps(
                error_if_maps_used_elsewhere=True, rebuild_in_place=False
            )
        
        
        self.assertEqual(str(cm.exception), expected_message)
        with self.assertRaises(RuntimeWarning) as cm:
            solution._rebuild_indices_maps(
                error_if_maps_used_elsewhere=False, rebuild_in_place=False
            )
        self.assertEqual(str(cm.exception), expected_message)

    @parameterized.expand(input=solvers)
    def test_solution_variable_errors_check(self, mip_solver):
        """
        Create a Solution Object, call its functions, and ensure the correct
        data is returned.
        """
        model = self.get_model()
        opt = pyo.SolverFactory(mip_solver)
        opt.solve(model)

        all_vars = au.pyomo_utils.get_model_variables(model, include_fixed=True)
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
        sol_vars_id_1 = id(sol_vars)
        assert len(sol_vars) == len(solution._variables)
        assert set(sol_vars.keys()) == {"x", "y", "z", "f"}

        for key, value in sol_vars.items():
            # checks accuracy of variable_name_to_index map
            assert solution._variables[value].name == key
        # old solution.fixed_variable_names functionality replaced with map
        sol_vars_fixed = solution.fixed_variable_indices
        fixed_variable_names = map(
            lambda x: solution._variables[x].name, sol_vars_fixed
        )
        assert set(fixed_variable_names) == {"f"}

        # testing index out of range lookup in solution.variable
        test_index = len(solution._variables)
        expected_message = (
            f"Index {test_index} is invalid in Solution with id {id(solution)}"
        )
        with self.assertRaises(AssertionError) as cm:
            solution.variable(test_index)
        self.assertEqual(str(cm.exception), expected_message)

        # testing invalid type in solution._variable_by_name
        test_index = len(solution._variables)
        expected_message = (
            f"Index {test_index} is invalid in Solution with id {id(solution)}"
        )
        with self.assertRaises(RuntimeError) as cm:
            solution._variable_by_name(test_index)
        self.assertEqual(str(cm.exception), expected_message)

        # testing valid type, not present key in solution._variable_by_name
        test_index = "topas"
        expected_message = f"Key {test_index} is not a valid key in the variable_name_to_index map in Solution with id {id(solution)}"
        with self.assertRaises(AssertionError) as cm:
            solution.variable(test_index)
        self.assertEqual(str(cm.exception), expected_message)

        # testing present key but maps to invalid index in solution._variable_by_name
        invalid_index = len(solution._variables)
        test_index = "x"
        solution.variable_name_to_index[test_index] = invalid_index
        expected_message = f"Index {invalid_index} corresponding to key {test_index} is not a valid variable list index in Solution with id {id(solution)}"
        with self.assertRaises(AssertionError) as cm:
            solution.variable(test_index)
        self.assertEqual(str(cm.exception), expected_message)

        # testing present key but maps to valid index in solution._variable_by_name corresponding to wrong variable
        # this is the map_consistency_check
        test_index = "x"
        wrong_index = 2
        solution.variable_name_to_index[test_index] = wrong_index
        expected_message = f"Mismatch between input variable name, {test_index}, and mapped to variable, {solution._variables[wrong_index].name} in Solution with id {id(solution)}"
        with self.assertRaises(RuntimeError) as cm:
            solution.variable(test_index, map_consistency_check=True)
        self.assertEqual(str(cm.exception), expected_message)

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
    @parameterized.expand(input=solvers)
    def test_solution_default_solution_not_pyomo_solution(self, mip_solver):
        #check that objective and objectives both being used in Solution gets the proper error
        expected_message = "The objective= and objectives= keywords cannot both be specified."
        with self.assertRaises(ValueError) as cm:
            solution = Solution(variables=None, objective=1, objectives = 2)
        self.assertEqual(str(cm.exception), expected_message)

        #check that objective to objecitves conversion works in solution
        solution = Solution(variables=None, objective=1)
        assert solution._objectives == [1]
    
    @parameterized.expand(input=solvers)
    def test_solution_external_var_maps(self, mip_solver):
        model = self.get_model()
        opt = pyo.SolverFactory(mip_solver)
        opt.solve(model)

        all_vars = au.pyomo_utils.get_model_variables(model, include_fixed=True)
        obj = au.pyomo_utils.get_active_objective(model)

        #check that saying to use an external variable_name_index without providing one errors
        name_index_map_string = "variable_name_to_index"
        expected_message = f"Attempted to create solution using external index maps without passing {name_index_map_string} map"
        with self.assertRaises(AssertionError) as cm:
            solution = PyomoSolution(variables=all_vars, objective=obj, use_given_index_maps = True)
        self.assertEqual(str(cm.exception), expected_message)

        #check that saying to use an external variable_name_index results in use of that object
        t = {'x':0}
        solution = PyomoSolution(variables=all_vars, objective=obj, use_given_index_maps = True, variable_name_to_index = t, keep_fixed_var_list = False)
        assert id(solution.variable_name_to_index) == id(t)
        assert solution.variable_name_to_index == {'x':0}

        #check that saying to use an external fixed_variable_indices without providing one errors
        fixed_variable_indicies_string = "fixed_variable_indices"
        expected_message = f"Attempted to create solution using external index maps without passing {fixed_variable_indicies_string} map"
        with self.assertRaises(AssertionError) as cm:
            PyomoSolution(variables=all_vars, objective=obj, use_given_index_maps = True, variable_name_to_index = t, keep_fixed_var_list = True)
        self.assertEqual(str(cm.exception), expected_message)

        #check that saying to use an external variable_name_index and fixed_variable_indices results in use of that object
        t = {'x':0}
        s = {0}
        solution = PyomoSolution(variables=all_vars, objective=obj, use_given_index_maps = True, variable_name_to_index = t, keep_fixed_var_list = True, fixed_variable_indices = s)
        assert id(solution.variable_name_to_index) == id(t)
        assert solution.variable_name_to_index == {'x':0}
        assert id(solution.fixed_variable_indices) == id(s)
        assert solution.fixed_variable_indices == {0}


if __name__ == "__main__":
    unittest.main()
