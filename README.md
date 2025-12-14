[![Pytest Tests](https://github.com/sandialabs/topas/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/sandialabs/topas/actions/workflows/pytest.yml?query=branch%3Amain)
[![Coverage Status](https://github.com/sandialabs/topas/actions/workflows/coverage.yml/badge.svg?branch=main)](https://github.com/sandialabs/topas/actions/workflows/coverage.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/sandialabs/topas/branch/main/graph/badge.svg)](https://codecov.io/gh/sandialabs/topas)
[![Documentation Status](https://readthedocs.org/projects/topas/badge/?version=latest)](http://topas.readthedocs.org/en/latest/)
[![GitHub contributors](https://img.shields.io/github/contributors/sandialabs/topas.svg)](https://github.com/sandialabs/topas/graphs/contributors)
[![Merged PRs](https://img.shields.io/github/issues-pr-closed-raw/sandialabs/topas.svg?label=merged+PRs)](https://github.com/sandialabs/topas/pulls?q=is:pr+is:merged)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# topas

Toolkit for Pyomo Alternative Solutions

## Overview

Topas provides a python toolkit for generating and analyzing alternative solutions for pyomo models.

## Testing

Topas tests can be executed using pytest:

```
cd topas
pytest .
```

If the pytest-cov package is installed, pytest can provide coverage statistics:

```
cd topas
pytest --cov=topas .
```

The following options list the lines that are missing from coverage tests:
```
cd topas
pytest --cov=topas --cov-report term-missing .
```

Note that pytest coverage includes coverage of test files themselves.  This gives a somewhat skewed sense of coverage for the code base, but it helps identify tests that are omitted or not executed completely.
