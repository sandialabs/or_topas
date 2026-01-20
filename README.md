[![Pytest Tests](https://github.com/sandialabs/or_topas/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/sandialabs/or_topas/actions/workflows/pytest.yml?query=branch%3Amain)
[![Coverage Status](https://github.com/sandialabs/or_topas/actions/workflows/coverage.yml/badge.svg?branch=main)](https://github.com/sandialabs/or_topas/actions/workflows/coverage.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/sandialabs/or_topas/branch/main/graph/badge.svg)](https://codecov.io/gh/sandialabs/or_topas)
[![Documentation Status](https://readthedocs.org/projects/or_topas/badge/?version=latest)](https://or_topas.readthedocs.org/en/latest/)
[![GitHub contributors](https://img.shields.io/github/contributors/sandialabs/or_topas.svg)](https://github.com/sandialabs/or_topas/graphs/contributors)
[![Merged PRs](https://img.shields.io/github/issues-pr-closed-raw/sandialabs/or_topas.svg?label=merged+PRs)](https://github.com/sandialabs/or_topas/pulls?q=is:pr+is:merged)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# OR-Topas

Operations Research Toolkit for Pyomo Alternative Solutions

## Overview

OR-Topas provides a python toolkit for generating and analyzing
alternative solutions for pyomo models.  This library is adapted from the
pyomo.contrib.alternative_solutions and pyomo.contrib.benders libraries.
OR-Topas extends the solution pool definition from pyomo, and it includes
new Benders implementations that support generation for alternative
solutions.

## Testing

OR-Topas tests can be executed using pytest:

```
cd or_topas
pytest .
```

If the pytest-cov package is installed, pytest can provide coverage statistics:

```
cd or_topas
pytest --cov=or_topas .
```

The following options list the lines that are missing from coverage tests:
```
cd or_topas
pytest --cov=or_topas --cov-report term-missing .
```

Note that pytest coverage includes coverage of test files themselves.
This gives a somewhat skewed sense of coverage for the code base, but
it helps identify tests that are omitted or not executed completely.

