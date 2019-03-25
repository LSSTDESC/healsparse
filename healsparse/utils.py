import numpy as np

def reduce_array(x, reduction='mean', axis=2):
    """
    Auxiliary method to perform one of the following operations:
    nanmean, nanmax, nanmedian, nanmin, nanstd

    Args:
    ----
    x: `ndarray`, input array in which to perform the operation
    reduction: `str`, reduction method. Valid options: mean, median, std, max, min
               (default: mean).
    axis: `int`, axis in which to perform the operation (default: 2)

    Returns:
    --------
    out: `ndarray`.
    """
    if reduction=='mean':
        return np.nanmean(x, axis=2).flatten()
    elif reduction=='median':
        return np.nanmedian(x, axis=2).flatten()
    elif reduction=='std':
        return np.nanstd(x, axis=2).flatten()
    elif reduction=='max':
        return np.nanmax(x, axis=2).flatten()
    elif reduction=='min':
        return np.nanmin(x, axis=2).flatten()
    else:
        raise ValueError('Reduction method %s not recognized.' % reduction)
