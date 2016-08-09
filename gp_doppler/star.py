from __future__ import (print_function, absolute_import)

import autograd.numpy as np
from scipy.optimize import newton

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

from .geometry import set_earth, dot, cross
from .property_descriptors import AffectsOmegaCrit, ResetsGrid

from astropy import constants as const, units as u
from astropy.coordinates import (SphericalRepresentation, UnitSphericalRepresentation,
                                 CartesianRepresentation)
from astropy.analytic_functions import blackbody_nu
from astropy.convolution import convolve, Gaussian1DKernel


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
        # now set up tile locations, velocities and directions.
        # These will be (nlon, nlat) CartesianRepresentations
        self.tile_locs = None
        self.tile_dirs = None
        self.tile_velocities = None

        # and arrays of tile properties - shape (nlon, nlat)
        self.tile_areas = None
        self.tile_fluxes = None

        # the next array is the main one that gets tweaked
        self.tile_scales = None

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

        # and effective gravities
        geff = np.sqrt((-const.G*self.mass/radii**2 + self.Omega**2 * radii * np.sin(theta_values)**2)**2 +
                       self.Omega**4 * radii**2 * np.sin(theta_values)**2 * np.cos(theta_values)**2)

        # now make a (3, nlon, nlat) array of positions
        self.tile_locs = SphericalRepresentation(phi_values[:, np.newaxis],
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
        self.tile_scales = np.ones((nlon, nlat))
        self.tile_fluxes = blackbody_nu(wavelength, tile_temperatures)

        # tile areas
        spher = self.tile_locs.represent_as(SphericalRepresentation)
        self.tile_areas = spher.distance**2 * np.sin(90*u.deg-spher.lat) * dtheta * dphi

        omega_vec = u.Quantity([0.0, 0.0, self.Omega.value], unit=self.Omega.unit)
        # get velocities of tiles
        loc_xyz = self.tile_locs.xyz
        cross_product = cross(omega_vec, loc_xyz)
        self.tile_velocities = CartesianRepresentation(
            u.Quantity(cross_product, unit=omega_vec.unit*loc_xyz.unit)
        )

    @u.quantity_input(inclination=u.deg)
    def plot(self, inclination=80*u.deg, phase=0.0, savefig=False, filename='star_surface.png',
             cmap='magma', what='fluxes', cstride=1, rstride=1, shade=False):
        ax = plt.axes(projection='3d')
        ax.view_init(90-inclination.to(u.deg).value, 360*phase)
        if what == 'fluxes':
            vals = self.tile_fluxes * self.tile_scales
            vals = vals / vals.max()
        elif what == 'vels':
            earth = set_earth(inclination.to(u.deg).value, phase)
            velocities = self.tile_velocities.xyz
            vals = dot(earth, velocities).to(u.km/u.s)
            # can't plot negative values, so rescale from 0 - 1
            vals = (vals - vals.min())/(vals.max()-vals.min())
        elif what == 'areas':
            vals = self.tile_areas / self.tile_areas.max()

        colors = getattr(cm, cmap)(vals.value)
        x, y, z = self.tile_locs.xyz.to(const.R_jup)
        ax.plot_surface(x.value, y.value, z.value, cstride=cstride, rstride=rstride, facecolors=colors,
                        shade=shade)
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

        mu = dot(earth, xyz) / np.sqrt(np.sum(xyz*xyz, axis=0))
        # mask of visible elements
        mask = mu >= 0.0
        return np.sum(
            self.tile_fluxes[mask] * self.tile_scales[mask] *
            (1.0 - self.ulimb + np.fabs(mu[mask])*self.ulimb) *
            self.tile_areas[mask] * mu[mask]
        )

    @u.quantity_input(inclination=u.deg)
    @u.quantity_input(v_macro=u.km/u.s)
    @u.quantity_input(v_inst=u.km/u.s)
    @u.quantity_input(v_min=u.km/u.s)
    @u.quantity_input(v_max=u.km/u.s)
    def calc_line_profile(self, phase, inclination, nbins=100,
                          v_macro=2*u.km/u.s, v_inst=4*u.km/u.s,
                          v_min=-40*u.km/u.s, v_max=40*u.km/u.s):

        # project velocities to get L.O.S
        earth = set_earth(inclination.to(u.deg).value, phase)
        velocities = self.tile_velocities.xyz
        vproj = dot(earth, velocities).to(u.km/u.s)

        # visible?
        xyz = self.tile_dirs.xyz
        # projection factor for tiles, mu = cos(theta)
        mu = dot(earth, xyz) / np.sqrt(np.sum(xyz*xyz, axis=0))
        vis_mask = mu >= 0.0

        bins = np.linspace(v_min, v_max, nbins)
        indices = np.digitize(vproj, bins)
        fluxes = np.zeros(nbins)
        for i in range(nbins):
            mask = (indices == i) & vis_mask
            # fluxes, including limb darkening
            fluxes[i] += np.sum(
                self.tile_fluxes[mask] * self.tile_scales[mask] *
                (1.0 - self.ulimb + np.fabs(mu[mask])*self.ulimb) *
                self.tile_areas[mask] * mu[mask]).value

        # convolve with instrumental and local line profiles
        bin_width = (v_max-v_min)/(nbins-1)
        profile_width_in_bins = np.sqrt(v_macro**2 + v_inst**2) / bin_width
        gauss_kernel = Gaussian1DKernel(stddev=profile_width_in_bins, mode='linear_interp')
        fluxes = convolve(fluxes, gauss_kernel, boundary='extend')
        return bins, fluxes
