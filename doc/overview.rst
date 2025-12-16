###############################################
Generating Alternative (Near-)Optimal Solutions
###############################################

.. py:currentmodule:: or_topas

Optimization solvers are generally designed to return a feasible solution
to the user. However, there are many applications where a user needs
more context than this result. For example,

* alternative optimal solutions can be used to assess trade-offs between
  competing objectives;

* comparisons amongst alternative solutions provide
  insights into the efficacy of model predictions with inaccurate or
  untrusted optimization formulations; or

* alternative optimal solutions create an opportunity to understand a
  design space, including assessments of unexpressed objectives and
  constraints;

* alternative solutions can be identified to support the future
  analysis of model revisions (e.g. to account for previously unexpressed
  constraints).

The *alternative-solutions library* provides a variety of functions that
can be used to generate optimal or near-optimal solutions for a pyomo
model. Conceptually, these functions are like pyomo solvers. They can
be configured with solver names and options, and they return a pool of
solutions for the pyomo model. However, these functions are independent of
pyomo's solver interfaces because they return a custom pool manager object.

The following functions are defined in the alternative-solutions library:

* :py:func:`enumerate_binary_solutions`

    * Finds alternative optimal solutions for a binary problem using no-good cuts.

* :py:func:`enumerate_linear_solutions`

    * Finds alternative optimal solutions for continuous variables in a
      (mixed-integer) linear program using iterative solutions of an
      integer programming formulation.

* :py:func:`gurobi_enumerate_linear_solutions`

    * Finds alternative optimal solutions for a (mixed-binary) linear
      program using Gurobi to generate lazy cuts.

* :py:func:`gurobi_generate_solutions`

    * Finds alternative optimal solutions for discrete variables using
      Gurobi's built-in solution pool capability.

* :py:func:`obbt_analysis_bounds_and_solutions`

    * Calculates the bounds on each variable by solving a series of min
      and max optimization problems where each variable is used as the
      objective function. This can be applied to any class of problem
      supported by the selected solver.


A Simple Example
----------------

Many of the functions in the alternative-solutions library have similar
options, so we simply illustrate the :py:func:`enumerate_binary_solutions`
function.  

We define a simple knapsack example whose alternative
solutions have integer objective values ranging from 0 to 70.

.. doctest::

   >>> import pyomo.environ as pyo

   >>> values = [20, 10, 60, 50]
   >>> weights = [5, 4, 6, 5]
   >>> capacity = 10

   >>> m = pyo.ConcreteModel()
   >>> m.x = pyo.Var(range(4), within=pyo.Binary)
   >>> m.o = pyo.Objective(expr=sum(values[i] * m.x[i] for i in range(4)), sense=pyo.maximize)
   >>> m.c = pyo.Constraint(expr=sum(weights[i] * m.x[i] for i in range(4)) <= capacity)

The function :py:func:`enumerate_binary_solutions` generates a
pool of :py:class:`Solution` objects that represent alternative optimal
solutions:

.. doctest::
   :skipif: not glpk_available

   >>> import pyomo.contrib.alternative_solutions as aos
   >>> solns = aos.enumerate_binary_solutions(m, num_solutions=100, solver="glpk")
   >>> assert len(solns) == 9
   >>> print( [soln.objective().value for soln in solns] )
   [70.0, 70.0, 60.0, 60.0, 50.0, 30.0, 20.0, 10.0, 0.0]


Enumerating Near-Optimal Solutions
----------------------------------

The previous example enumerated all feasible solutions. However optimization models are typically
used to identify optimal or near-optimal solutions.  The ``abs_opt_gap`` and ``rel_opt_gap``
arguments are used to limit the search to these solutions:

* ``rel_opt_gap`` : non-negative float or None

  * The relative optimality gap for allowable alternative solutions.  Specifying a gap of ``None`` indicates that there is no limit on the relative optimality gap (i.e. that any feasible solution can be considered).

* ``abs_opt_gap`` : non-negative float or None

  * The absolute optimality gap for allowable alternative solutions.  Specifying a gap of ``None`` indicates that there is no limit on the absolute optimality gap (i.e. that any feasible solution can be considered).

For example, we can generate all optimal solutions as follows:

.. doctest::
   :skipif: not glpk_available

   >>> solns = aos.enumerate_binary_solutions(m, num_solutions=100, solver="glpk", abs_opt_gap=0.0)
   >>> print( [soln.objective().value for soln in solns] )
   [70.0, 70.0]

Similarly, we can generate the six solutions within 40 of the optimum:

.. doctest::
   :skipif: not glpk_available

   >>> solns = aos.enumerate_binary_solutions(m, num_solutions=100, solver="glpk", abs_opt_gap=40.0)
   >>> print( [soln.objective().value for soln in solns] )
   [70.0, 70.0, 60.0, 60.0, 50.0, 30.0]
