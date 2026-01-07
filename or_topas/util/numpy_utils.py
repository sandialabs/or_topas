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

from pyomo.common.dependencies import numpy as numpy, numpy_available

if numpy_available:
    import numpy.random
    from numpy.linalg import norm

if numpy_available:
    rng = numpy.random.default_rng(9283749387)
else:
    rng = None


def set_numpy_rng(seed):
    global rng
    rng = numpy.random.default_rng(seed)


def get_random_direction(num_dimensions, iterations=1000, min_norm=1e-4):
    """
    Get a unit vector of dimension num_dimensions by sampling from and
    normalizing a standard multivariate Gaussian distribution.
    """
    for idx in range(iterations):
        samples = rng.normal(size=num_dimensions)
        samples_norm = norm(samples)
        if samples_norm > min_norm:
            return samples / samples_norm
    raise Exception(  # pragma: no cover
        (
            "Generated {} sequential Gaussian draws with a norm of "
            "less than {}.".format(iterations, min_norm)
        )
    )
