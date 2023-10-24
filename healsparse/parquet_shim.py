import warnings
import numpy as np

use_pyarrow = False
try:
    from pyarrow import dataset
    from pyarrow.lib import ArrowInvalid
    import pyarrow as pa
    use_pyarrow = True
except ImportError:
    pass

if use_pyarrow:
    DTYPE_DICT = {
        pa.int8(): np.int8,
        pa.int16(): np.int16,
        pa.int32(): np.int32,
        pa.int64(): np.int64,
        pa.float16(): np.float16,
        pa.float32(): np.float32,
        pa.float64(): np.float64,
        pa.uint8(): np.uint8,
        pa.uint16(): np.uint16,
        pa.uint32(): np.uint32,
        pa.uint64(): np.uint64,
        pa.bool_(): np.bool_,
    }


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


def to_numpy_dtype(arrow_type):
    """Return the equivalent numpy datatype for an arrow datatype.

    Parameters
    ----------
    arrow_type : `pyarrow.DataType`

    Returns
    -------
    numpy_dtype : `numpy.dtype`
    """
    try:
        numpy_dtype = DTYPE_DICT[arrow_type]
    except KeyError:
        raise ValueError("Unsupported pyarrow datatype: %s" % (str(arrow_type)))

    return numpy_dtype
