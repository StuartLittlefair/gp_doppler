from __future__ import (print_function, absolute_import)

import autograd.numpy as np


def set_earth(inclination, phase):
    """
    Calculate vector that points towards observer.

    Parameters
    ----------
    inclination : float
        inclination of star to los in degrees. 90=edge on
    phase : float
        rotational phase of star

    Returns
    -------
    earth : `numpy.ndarray`
        vector shape (3,) pointing to Earth
    """
    ri = np.radians(inclination)
    cosi, sini = np.cos(ri), np.sin(ri)
    cosp = np.cos(2*np.pi*phase)
    sinp = np.sin(2*np.pi*phase)
    return np.asarray((sini*cosp, -sini*sinp, cosi))


def dot(vec, vec_array):
    """
    Compute dot product of vector with an array of vectors.

    We assume that the vector array has the vectors residing in the
    first dimension - i.e it is of shape (3, ...)

    Parameters
    ----------
    vec : `numpy.ndarray`
        vector
    vec_array : `numpy.ndarray`
        array of vectors of shape (3, ...)

    Returns
    -------
    dot_product : `numpy.ndarray`
        array of dot products. If vec_array has shape (3, N, M), the returned
        dot_product array has shape (N, M).
    """
    # since the data can be n-dimensional, reshape
    # to a 2-d (3, N) array
    xyz = vec_array.reshape((3, vec_array.size // 3))

    # take the dot product
    dot_product = np.dot(vec, xyz)

    # restore the correct shape
    return dot_product.reshape(vec_array.shape[1:])


def cross(vec, vec_array):
    """
    Compute cross product of vector with an array of vectors.

    We assume that the vector array has the vectors residing in the
    first dimension - i.e it is of shape (3, ...)

    Parameters
    ----------
    vec : `numpy.ndarray`
        vector
    vec_array : `numpy.ndarray`
        array of vectors of shape (3, ...)

    Returns
    -------
    cross_product : `numpy.ndarray`
        array of cross products. If vec_array has shape (3, N, M), the returned
        cross_product array has shape (3, N, M).
    """
    # take the cross product
    return np.cross(vec, vec_array, axisb=0, axisc=0)
