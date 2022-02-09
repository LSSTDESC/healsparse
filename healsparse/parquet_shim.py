import warnings

use_pyarrow = False
try:
    from pyarrow import dataset
    from pyarrow.lib import ArrowInvalid
    use_pyarrow = True
except ImportError:
    pass


def check_parquet_dataset(filepath):
    """
    Check if a filepath points to a parquet dataset.

    Parameters
    ----------
    filepath : `str`
        File path to check.

    Returns
    -------
    is_parquet_dataset : `bool`
        True if it is a parquet dataset.

    Raises
    ------
    Warns if pyarrow is not installed.
    """
    if not use_pyarrow:
        warnings.warn("Cannot access parquet datasets without pyarrow.",
                      UserWarning)
        return False

    try:
        ds = dataset.dataset(filepath, format='parquet', partitioning='hive')
    except (IOError, ArrowInvalid):
        return False

    del ds

    return True
