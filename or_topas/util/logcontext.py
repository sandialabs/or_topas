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

import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def logcontext(level):
    """
    This context manager is used to dynamically set the specified logging level
    and then execute a block of code using that logging level.  When the context is
    deleted, the logging level is reset to the original value.

    Examples
    --------
    >>> with logcontext(logging.INFO):
    ...    logging.debug("This will not be printed")
    ...    logging.info("This will be printed")

    """
    logger = logging.getLogger()
    current_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(current_level)
