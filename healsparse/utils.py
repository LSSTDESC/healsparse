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


def check_sentinel(type, sentinel):
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
