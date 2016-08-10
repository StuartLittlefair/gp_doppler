from __future__ import (print_function, absolute_import)

import numpy as np
from astropy.coordinates import CartesianRepresentation
from astropy import units as u


@u.quantity_input(inclination=u.deg)
def set_earth(inclination, phases):
    """
    Calculate vector that points towards observer.

    Parameters
    ----------
    inclination : float
        inclination of star to los in degrees. 90=edge on
    phases : float or `np.ndarray`
        rotational phase of star

    Returns
    -------
    earth : `CartesianRepresentation`
        vector pointing to Earth
    """
    cosi, sini = np.cos(inclination), np.sin(inclination)
    cosp = np.cos(2*np.pi*phases)
    sinp = np.sin(2*np.pi*phases)
    return CartesianRepresentation(sini*cosp, -sini*sinp, cosi)


@u.quantity_input(inclination=u.deg)
def earth_grad(inclination, phases):
    """
    Calculate gradient of earth vector w.r.t inclination.

    Parameters
    ----------
    inclination : float
        inclination of star to los in degrees. 90=edge on
    phases : float or `np.ndarray`
        rotational phase of star

    Returns
    -------
    earth : `CartesianRepresentation`
        gradient of earth vector w.r.t inclination
    """
    cosi, sini = np.cos(inclination), np.sin(inclination)
    cosp = np.cos(2*np.pi*phases)
    sinp = np.sin(2*np.pi*phases)
    return CartesianRepresentation(cosi*cosp, -cosi*sinp, -sini)


def dot(a, b, normalise=False):
    """
    Dot product of two CartesianRepresentations.

    Parameters
    ----------
    a : `CartesianRepresentation`
        First CartesianRepresentation, e.g directions to earth
    b : `CartesianRepresentation`
        Second CartesianRepresentation, e.g directions of tiles
    normalise : bool
        if True, normalise so dot product is cos(theta), where
        theta is angle between a and b

    Returns
    -------
    dot_product : `u.Quantity`
        array of dot products. If `a` has shape (N, M) and `b` has shape (K,L)
        returned Quantity has shape (N, M, K, L)
    """
    # since the data can be n-dimensional, reshape
    # to a 2-d (3, N) array
    xyz_a, xyz_b = a.xyz, b.xyz
    orig_shape_a = xyz_a.shape
    orig_shape_b = xyz_b.shape
    xyz_a = xyz_a.reshape((3, xyz_a.size // 3))
    xyz_b = xyz_b.reshape((3, xyz_b.size // 3))

    # take the dot product, broadcast over both axes so we
    # get a result of shape (xyz_a.size, xyz_b.size)
    dot_product = np.dot(xyz_a.T, xyz_b)
    dot_product_unit = xyz_a.unit * xyz_b.unit
    dot_product = u.Quantity(dot_product.value, unit=dot_product_unit)

    # normalise if requested
    if normalise:
        length_a = np.sqrt(np.sum(xyz_a*xyz_a, axis=0))
        length_b = np.sqrt(np.sum(xyz_b*xyz_b, axis=0))
        length = length_a[:, np.newaxis]*length_b
        dot_product /= length

    # restore the correct shape
    # should be, e.g (Na, Ma, Nb, Mb)
    return dot_product.reshape(orig_shape_a[1:] + orig_shape_b[1:])


def cross(a, b):
    """
    Compute cross product of two CartesianRepresentations.

    Parameters
    ----------
    a : `CartesianRepresentation`
        First CartesianRepresentation, e.g directions to earth
    b : `CartesianRepresentation`
        Second CartesianRepresentation, e.g directions of tiles

    Returns
    -------
    cross_product : `CartesianRepresentation`
        array of cross products. If `a` has shape (N, M) and `b` has shape (K,L)
        returned Quantity has shape (N, M, K, L)
    """
    # since the data can be n-dimensional, reshape
    # to a 2-d (3, N) array
    xyz_a, xyz_b = a.xyz, b.xyz
    orig_shape_a = xyz_a.shape
    orig_shape_b = xyz_b.shape
    xyz_a = xyz_a.reshape((3, xyz_a.size // 3))
    xyz_b = xyz_b.reshape((3, xyz_b.size // 3))

    # take the cross product
    cross_product = np.cross(xyz_a[:, :, np.newaxis], xyz_b,
                             axisa=0, axisb=0, axisc=0)
    cross_product_unit = xyz_a.unit * xyz_b.unit
    cross_product = u.Quantity(cross_product, unit=cross_product_unit)

    cartrep = CartesianRepresentation(cross_product)
    return cartrep.reshape(orig_shape_a[1:] + orig_shape_b[1:])
