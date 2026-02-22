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

from munch import Munch


class MyMunch(Munch):
    # WEH, MPV needed to add a to_dict since Bunch did not have one
    def to_dict(self):
        return to_dict(self)


def to_dict(x):
    xtype = type(x)
    if xtype in [tuple, set, frozenset]:
        return list(x)
    elif xtype in [dict, Munch, MyMunch]:
        return {k: to_dict(v) for k, v in x.items()}
    elif hasattr(x, "to_dict"):
        return x.to_dict()
    else:
        return x
