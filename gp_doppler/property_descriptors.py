from __future__ import (print_function, absolute_import)

import autograd.numpy as np
import warnings
from astropy import constants as const

"""
Below we define some property descriptors that help us avoid setting supercritical
rotation and also clear the grid when necessary
"""


class AffectsOmegaCrit:
    def __init__(self, storage_name, unit):
        self.storage_name = storage_name
        self.unit = unit

    def __set__(self, instance, value):
        if not value.unit.is_equivalent(self.unit):
            msg = "Setting {} with a value that has wrong unit".format(self.storage_name)
            raise ValueError(msg)

        if self.storage_name == 'mass':
            radius = instance.__dict__.get('radius', None)
            mass = value
        elif self.storage_name == 'radius':
            mass = instance.__dict__.get('mass', None)
            radius = value

        if mass is not None and radius is not None:
            # check and set omega crit
            omega_crit = np.sqrt(8*const.G*mass/radius**3/27).decompose()
            Omega = getattr(instance.__dict__, 'Omega', None)
            instance.__dict__['omega_crit'] = omega_crit
            if Omega is not None and Omega/omega_crit > 1:
                warnings.warn("Critical value is now less than current rotation rate")

        instance.__dict__[self.storage_name] = value


class ResetsGrid:
    def __init__(self, storage_name, unit=None):
        self.storage_name = storage_name
        self.unit = unit

    def __set__(self, instance, value):
        if self.unit is not None and not value.unit.is_equivalent(self.unit):
            msg = "Setting {} with a value that has wrong unit".format(self.storage_name)
            raise ValueError(msg)
        instance.__dict__[self.storage_name] = value
        instance.clear_grid()
