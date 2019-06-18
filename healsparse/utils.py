import numpy as np
import healpy as hp
import warnings


def reduce_array(x, reduction='mean', axis=2):
    """
    Auxiliary method to perform one of the following operations:
    nanmean, nanmax, nanmedian, nanmin, nanstd

    Args:
    ----
    x: `ndarray`
        input array in which to perform the operation
    reduction: `str`
        reduction method. Valid options: mean, median, std, max, min
        (default: mean).
    axis: `int`
        axis in which to perform the operation (default: 2)

    Returns:
    --------
    out: `ndarray`.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        if reduction == 'mean':
            ret = np.nanmean(x, axis=2).flatten()
        elif reduction == 'median':
            ret = np.nanmedian(x, axis=2).flatten()
        elif reduction == 'std':
            ret = np.nanstd(x, axis=2).flatten()
        elif reduction == 'max':
            ret = np.nanmax(x, axis=2).flatten()
        elif reduction == 'min':
            ret = np.nanmin(x, axis=2).flatten()
        else:
            raise ValueError('Reduction method %s not recognized.' % reduction)

    return ret


def checkSentinel(type, sentinel):
    """
    Check if the sentinel value works for the given dtype.

    Parameters
    ----------
    type: `type`
    sentinel: `int`, `float`, or None

    Returns
    -------
    Default sentinel if input is None.

    Raises
    ------
    ValueError if sentinel is of wrong type
    """

    if issubclass(type, np.floating):
        # If we don't have a sentinel, use hp.UNSEEN
        if sentinel is None:
            return hp.UNSEEN
        # If input is a float, we're okay.  Otherwise, Raise.
        if (issubclass(sentinel.__class__, np.floating) or
                issubclass(sentinel.__class__, float)):
            return sentinel
        else:
            raise ValueError("Sentinel not of floating type")
    elif issubclass(type, np.integer):
        # If we don't have a sentinel, MININT
        if sentinel is None:
            return np.iinfo(type).min
        if (issubclass(sentinel.__class__, np.integer) or
                issubclass(sentinel.__class__, int)):
            if (sentinel < np.iinfo(type).min or
                    sentinel > np.iinfo(type).max):
                raise ValueError("Sentinel out of range of type")
            return sentinel
        else:
            raise ValueError("Sentinel not of integer type")


def eq2ang(ra, dec):
    """
    convert ra, dec to theta, phi in healpix conventions

    Parameters
    ----------
    ra: scalar or array
        ra in degrees
    dec: scalar or array
        decin radians

    Returns
    -------
    theta, phi in radians
    """

    # make sure ra in [0,360] and dec within [-90,90]
    ra, dec = _reset_bounds2(dec, ra)

    phi = np.deg2rad(ra)
    theta = np.pi/2.0 - np.deg2rad(dec)

    return theta, phi


def ang2xyz(theta, phi):
    """
    convert theta, phi to x, y, z in healpix conventions

    Parameters
    ----------
    theta: scalar or array
        theta in radians
    phi: scalar or array
        phi in radians

    Returns
    -------
    x, y, z on unit sphere
    """
    sintheta = 0.0

    sintheta = np.sin(theta)
    x = sintheta * np.cos(phi)
    y = sintheta * np.sin(phi)
    z = np.cos(theta)

    return x, y, z


def eq2xyz(ra, dec):
    """
    convert ra,dec to x, y, z in healpix conventions

    Parameters
    ----------
    ra: scalar or array
        ra in degrees
    dec: scalar or array
        decin radians

    Returns
    -------
    x, y, z on unit sphere
    """

    theta, phi = eq2ang(ra, dec)
    x, y, z = ang2xyz(theta, phi)

    return x, y, z


def eq2vec(ra, dec):
    """
    Convert equatorial ra,dec in degrees to x,y,z on the unit sphere

    Parameters
    ----------
    ra: scalar or array
        ra in degrees
    dec: scalar or array
        decin radians

    Returns
    -------
    vec with x,y,z on the unit sphere
        A 1-d vector[3] or a 2-d array[npoints, 3]

    """
    is_scalar = np.isscalar(ra)

    ra = np.array(ra, dtype='f8', ndmin=1, copy=False)
    dec = np.array(dec, dtype='f8', ndmin=1, copy=False)
    if ra.size != dec.size:
        raise ValueError('ra,dec not same '
                         'size: %s,%s' % (ra.size, dec.size))

    x, y, z = eq2xyz(ra, dec)

    vec = np.zeros((ra.size, 3))
    vec[:, 0] = x
    vec[:, 1] = y
    vec[:, 2] = z

    if is_scalar:
        vec = vec[0, :]

    return vec


def _reset_bounds(angle, min, max):
    while angle < min:
        angle += 360.0

    while angle >= max:
        angle -= 360.0

    return angle


def _reset_bounds2(theta, phi):
    theta = _reset_bounds(theta, -180.0, 180.0)

    if abs(theta) > 90.0:
        theta = 180.0 - theta
        phi += 180

    theta = _reset_bounds(theta, -180.0, 180.0)
    phi = _reset_bounds(phi, 0.0, 360.0)
    if abs(theta) == 90.0:
        phi = 0.0

    return theta, phi
