from __future__ import (print_function, absolute_import)

import numpy as np
from scipy.optimize import newton

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

from .geometry import set_earth, dot
from .property_descriptors import AffectsOmegaCrit, ResetsGrid

from astropy import constants as const, units as u
from astropy.coordinates import SphericalRepresentation, UnitSphericalRepresentation
from astropy.analytic_functions import blackbody_nu


def surface(x, omega, theta):
    """
    Potential of star at co-latitude theta.

    For a rotating star the scaled radius x = R/Rpole
    obeys the equation defined by this function. Combine
    with a newton-raphson solver to find x.

    Parameters
    ----------
    x : float
        scaled radius of star at colatitude theta
    omega : float
        ratio of stellar angular velocity rate to critical rate
    theta : `~astropy.units.Quantity`
        co-latitude

    Returns
    -------
    val : float
        this should be zero at surface
    """
    return 1/x + 4 * omega**2 * x**2 * np.sin(theta)**2 / 27 - 1


class Star:

    mass = AffectsOmegaCrit('mass', u.kg)
    radius = AffectsOmegaCrit('radius', u.m)
    beta = ResetsGrid('beta')
    ntiles = ResetsGrid('ntiles')
    distortion = ResetsGrid('distortion')

    @u.quantity_input(mass=u.kg)
    @u.quantity_input(radius=u.m)
    @u.quantity_input(period=u.s)
    def __init__(self, mass, radius, period, beta=0.08, ulimb=0.9,
                 ntiles=400, distortion=True):
        self.distortion = distortion
        self.mass = mass
        self.radius = radius  # will also set self.omega_crit
        self.beta = beta
        self.ulimb = ulimb
        self.ntiles = ntiles

        self.period = period
        self.clear_grid()

    def clear_grid(self):
        # now set up tile locs and directions.
        # These will be (nlon, nlat) CartesianRepresentations
        self.tile_locs = None
        self.tile_dirs = None

        # and arrays of tile properties - shape (nlon, nlat)
        self.tile_areas = None
        self.tile_fluxes = None

    """
    We define many of the attributes as properties, so we can wipe the grid when they are set,
    and also check for violation of critical rotation
    """
    @property
    def omega(self):
        """
        Ratio of angular velocity to critical number. read-only property
        """
        return (self.Omega/self.omega_crit).decompose()

    @property
    def period(self):
        return 2.0*np.pi/self.Omega

    @period.setter
    @u.quantity_input(value=u.s)
    def period(self, value):
        Omega = 2.0*np.pi/value
        if (Omega/self.omega_crit).decompose() > 1:
            raise ValueError('This rotation period exceeds critical value')
        self.Omega = Omega
        self.clear_grid()

    @u.quantity_input(wavelength=u.nm)
    def setup_grid(self, wavelength=656*u.nm):
        # how many meridians do we need?
        nlat = int(np.sqrt(self.ntiles))
        nlon = nlat
        # co-latitude
        theta_values = np.linspace(0, 180, nlat) * u.degree
        dtheta = 180*u.deg/(nlat-1)
        # longitude
        phi_values = np.linspace(0.0, 2*np.pi, nlon)*u.rad
        dphi = 360*u.deg/(nlon-1)

        # the following formulae use the Roche approximation and assume
        # solid body rotation
        # solve for radius of rotating star at these co-latitudes
        if self.distortion:
            radii = self.radius*np.array([newton(surface, 1.01, args=(self.omega, x)) for x in theta_values])
        else:
            radii = self.radius*np.ones_like(theta_values).value
        print(radii.mean(), radii.std(), radii.shape)
        # and effective gravities
        geff = np.sqrt((-const.G*self.mass/radii**2 + self.Omega**2 * radii * np.sin(theta_values)**2)**2 +
                       self.Omega**4 * radii**2 * np.sin(theta_values)**2 * np.cos(theta_values)**2)
        print(geff.mean(), geff.std(), geff.shape)
        # now make a (3, nlon, nlat) array of positions
        self.tile_locs = self.tile_locs = SphericalRepresentation(phi_values[:, np.newaxis],
                                                                  90*u.deg-theta_values,
                                                                  radii).to_cartesian()

        # normal to tile is the direction of the derivate of the potential
        # this is the vector form of geff above
        # the easiest way to express it is that it differs from (r, theta, phi)
        # by a small amount in the theta direction epsilon
        # also (3, nlon, nlat)
        x = radii/self.radius
        a = 1./x**2 - (8./27.)*self.omega**2 * x * np.sin(theta_values)**2
        b = np.sqrt(
                (-1./x**2 + (8./27)*self.omega**2 * x * np.sin(theta_values)**2)**2 +
                ((8./27)*self.omega**2 * x * np.sin(theta_values) * np.cos(theta_values))**2
            )
        epsilon = np.arccos(a/b)
        self.tile_dirs = UnitSphericalRepresentation(phi_values[:, np.newaxis],
                                                     90*u.deg - theta_values - epsilon)
        self.tile_dirs = self.tile_dirs.to_cartesian()

        # and (nlon, nlat) arrays of tile properties
        tile_temperatures = 2000.0 * u.K * (geff / geff.max())**self.beta
        tile_temperatures = tile_temperatures * np.ones_like(phi_values[:, np.newaxis]/u.deg)

        # fluxes, not accounting for limb darkening
        self.tile_fluxes = blackbody_nu(wavelength, tile_temperatures)

        # tile areas
        spher = self.tile_locs.represent_as(SphericalRepresentation)
        self.tile_areas = spher.distance**2 * np.sin(90*u.deg-spher.lat) * dtheta * dphi

    def plot(self, savefig=False, filename='star_surface.png',
             cmap='magma'):
        ax = plt.axes(projection='3d')
        norm_fluxes = self.tile_fluxes / self.tile_fluxes.max()
        colors = getattr(cm, cmap)(norm_fluxes.value)
        x, y, z = self.tile_locs.xyz.to(const.R_jup)
        ax.plot_surface(x.value, y.value, z.value, cstride=1, rstride=1, facecolors=colors,
                        shade=False)
        if savefig:
            plt.savefig(filename)
        else:
            plt.show()

    @u.quantity_input(inclination=u.deg)
    def calc_luminosity(self, phase, inclination):
        if self.tile_locs is None:
            self.setup_grid()

        # get xyz of tile directions now since it's expensice
        xyz = self.tile_dirs.xyz
        earth = set_earth(inclination.to(u.deg).value, phase)

        mu = dot(earth, xyz)
        # mask of visible elements
        mask = mu > -0.01
        return np.sum(
            self.tile_fluxes[mask] *
            (1.0 - self.ulimb + np.fabs(mu[mask])*self.ulimb) *
            self.tile_areas[mask] * mu[mask]
        )


