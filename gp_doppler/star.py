from __future__ import (print_function, absolute_import)

import numpy as np
from scipy.optimize import newton

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

from .geometry import set_earth, dot, cross, earth_grad
from .property_descriptors import AffectsOmegaCrit, ResetsGrid

from astropy import constants as const, units as u
from astropy.coordinates import (SphericalRepresentation, UnitSphericalRepresentation,
                                 CartesianRepresentation)
from astropy.analytic_functions import blackbody_nu
from astropy.convolution import convolve, Gaussian1DKernel

import healpy as hp


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
    distortion = ResetsGrid('distortion')

    @u.quantity_input(mass=u.kg)
    @u.quantity_input(radius=u.m)
    @u.quantity_input(period=u.s)
    def __init__(self, mass, radius, period, beta=0.08, ulimb=0.9,
                 ntiles=3072, distortion=True):
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

    """
    ntiles is also a property, since we need to remap to an appropriate
    value for HEALPIX.
    """
    @property
    def ntiles(self):
        """
        Number of tiles.

        Is checked to see if appropriate for HEALPIX algorithm.
        """
        return self._ntiles

    @ntiles.setter
    def ntiles(self, value):
        allowed_values = [48, 192, 768, 3072, 12288, 49152, 196608]
        if int(value) not in allowed_values:
            raise ValueError('{} not one of allowed values: {!r}'.format(
                value, allowed_values
            ))
        self._ntiles = int(12*np.floor(np.sqrt(value/12.))**2)
        self.clear_grid()

    @u.quantity_input(wavelength=u.nm)
    def setup_grid(self, wavelength=656*u.nm):
        # use HEALPIX to get evenly sized tiles
        NSIDE = hp.npix2nside(self.ntiles)

        colat, lon = hp.pix2ang(NSIDE, np.arange(0, self.ntiles))
        # co-latitude
        theta_values = u.Quantity(colat, unit=u.rad)
        # longitude
        phi_values = u.Quantity(lon, unit=u.rad)

        # the following formulae use the Roche approximation and assume
        # solid body rotation
        # solve for radius of rotating star at these co-latitudes
        if self.distortion:
            radii = self.radius*np.array([newton(surface, 1.01, args=(self.omega, x)) for x in theta_values])
        else:
            radii = self.radius*np.ones(self.ntiles)

        # and effective gravities
        geff = np.sqrt((-const.G*self.mass/radii**2 + self.Omega**2 * radii * np.sin(theta_values)**2)**2 +
                       self.Omega**4 * radii**2 * np.sin(theta_values)**2 * np.cos(theta_values)**2)

        # now make a ntiles sized CartesianRepresentation of positions
        self.tile_locs = SphericalRepresentation(phi_values,
                                                 90*u.deg-theta_values,
                                                 radii).to_cartesian()

        # normal to tile is the direction of the derivate of the potential
        # this is the vector form of geff above
        # the easiest way to express it is that it differs from (r, theta, phi)
        # by a small amount in the theta direction epsilon
        x = radii/self.radius
        a = 1./x**2 - (8./27.)*self.omega**2 * x * np.sin(theta_values)**2
        b = np.sqrt(
                (-1./x**2 + (8./27)*self.omega**2 * x * np.sin(theta_values)**2)**2 +
                ((8./27)*self.omega**2 * x * np.sin(theta_values) * np.cos(theta_values))**2
            )
        epsilon = np.arccos(a/b)
        self.tile_dirs = UnitSphericalRepresentation(phi_values,
                                                     90*u.deg - theta_values - epsilon)
        self.tile_dirs = self.tile_dirs.to_cartesian()

        # and ntiles sized arrays of tile properties
        tile_temperatures = 2000.0 * u.K * (geff / geff.max())**self.beta

        # fluxes, not accounting for limb darkening
        self.tile_scales = np.ones(self.ntiles)
        self.tile_fluxes = blackbody_nu(wavelength, tile_temperatures)

        # tile areas
        spher = self.tile_locs.represent_as(SphericalRepresentation)
        self.tile_areas = spher.distance**2 * hp.nside2pixarea(NSIDE) * u.rad * u.rad

        omega_vec = CartesianRepresentation(
            u.Quantity([0.0, 0.0, self.Omega.value],
                       unit=self.Omega.unit)
        )
        # get velocities of tiles
        self.tile_velocities = cross(omega_vec, self.tile_locs)

    @u.quantity_input(inclination=u.deg)
    def plot(self, inclination=90*u.deg, phase=0.0, savefig=False, filename='star_surface.png',
             cmap='magma', what='fluxes', cstride=1, rstride=1, shade=False):
        ax = plt.axes(projection='3d')
        ax.view_init(90-inclination.to(u.deg).value, 360*phase)

        # get map values
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

        # project the map to a rectangular matrix
        nlat = nlon = int(np.floor(np.sqrt(self.ntiles)))
        theta = np.linspace(np.pi, 0, nlat)
        phi = np.linspace(-np.pi, np.pi, nlon)
        PHI, THETA = np.meshgrid(phi, theta)
        NSIDE = hp.npix2nside(self.ntiles)
        grid_pix = hp.ang2pix(NSIDE, THETA, PHI)
        grid_map = colors[grid_pix]

        # Create a sphere
        r = 0.3
        x = r*np.sin(THETA)*np.cos(PHI)
        y = r*np.sin(THETA)*np.sin(PHI)
        z = r*np.cos(THETA)

        ax.plot_surface(x, y, z, cstride=cstride, rstride=rstride, facecolors=grid_map,
                        shade=shade)
        if savefig:
            plt.savefig(filename)
        else:
            plt.show()

    @u.quantity_input(inclination=u.deg)
    def view(self, inclination=90*u.deg, phase=0.0, what='fluxes',
             projection='mollweide', cmap='magma',
             savefig=False, filename='star_surface.png',
             dlat=30, dlon=30, **kwargs):
        rot = (360*phase, 90-inclination.to(u.deg).value, 0)
        if what == 'fluxes':
            vals = self.tile_fluxes * self.tile_scales
            vals = vals / vals.max()
        elif what == 'areas':
            vals = self.tile_areas / self.tile_areas.max()

        if 'mollweide'.find(projection) == 0:
            hp.mollview(vals, rot=rot, cmap=cmap, **kwargs)
        elif 'cartesian'.find(projection) == 0:
            hp.cartview(vals, rot=rot, cmap=cmap, **kwargs)
        elif 'orthographic'.find(projection) == 0:
            hp.orthview(vals, rot=rot, cmap=cmap, **kwargs)
        else:
            raise ValueError('Unrecognised projection')
        hp.graticule(dlat, dlon)
        if savefig:
            plt.savefig(filename)
        else:
            plt.show()

    @u.quantity_input(inclination=u.deg)
    def _luminosity_array(self, phase, inclination):
        if self.tile_locs is None:
            self.setup_grid()

        # get CartesianRepresentation pointing to earth at these phases
        earth = set_earth(inclination, phase)

        mu = dot(earth, self.tile_dirs, normalise=True)
        # mask of visible elements
        mask = mu >= 0.0

        # broadcast and calculate
        phase = np.asarray(phase)
        new_shape = phase.shape + self.tile_fluxes.shape
        assert(new_shape == mu.shape), "Broadcasting has gone wrong"

        fluxes = np.tile(self.tile_fluxes, phase.size).reshape(new_shape)
        scales = np.tile(self.tile_scales, phase.size).reshape(new_shape)
        areas = np.tile(self.tile_areas, phase.size).reshape(new_shape)

        # limb darkened sum of all tile fluxes
        lum = (fluxes * scales * (1.0 - self.ulimb + np.fabs(mu)*self.ulimb) *
               areas * mu)
        # no contribution from invisible tiles
        lum[mask] = 0.0
        return lum

    @u.quantity_input(inclination=u.deg)
    def calc_luminosity(self, phase, inclination):
        lum = self._luminosity_array(phase, inclination)
        return np.sum(lum, axis=1)

    @u.quantity_input(inclination=u.deg)
    @u.quantity_input(v_macro=u.km/u.s)
    @u.quantity_input(v_inst=u.km/u.s)
    @u.quantity_input(v_min=u.km/u.s)
    @u.quantity_input(v_max=u.km/u.s)
    def calc_line_profile(self, phase, inclination, nbins=100,
                          v_macro=2*u.km/u.s, v_inst=4*u.km/u.s,
                          v_min=-40*u.km/u.s, v_max=40*u.km/u.s):

        # get CartesianRepresentation pointing to earth at these phases
        earth = set_earth(inclination, phase)
        # get CartesianRepresentation of projected velocities
        vproj = dot(earth, self.tile_velocities).to(u.km/u.s)

        # which tiles fall in which bin?
        bins = np.linspace(v_min, v_max, nbins)
        indices = np.digitize(vproj, bins)

        lum = self._luminosity_array(phase, inclination)
        phase = np.asarray(phase)
        trailed_spectrum = np.zeros((phase.size, nbins))

        for i in range(nbins):
            mask = (indices == i)
            trailed_spectrum[:, i] = np.sum(lum*mask, axis=1)

        # convolve with instrumental and local line profiles
        # TODO: realistic Line Profile Treatment
        # For now we assume every element has same intrinsic
        # line profile
        bin_width = (v_max-v_min)/(nbins-1)
        profile_width_in_bins = np.sqrt(v_macro**2 + v_inst**2) / bin_width
        gauss_kernel = Gaussian1DKernel(stddev=profile_width_in_bins, mode='linear_interp')
        for i in range(phase.size):
            trailed_spectrum[i, :] = convolve(trailed_spectrum[i, :], gauss_kernel, boundary='extend')

        return bins, trailed_spectrum
