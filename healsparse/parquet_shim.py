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
    if arrow_type == pa.int64():
        return np.int64
    elif arrow_type == pa.int32():
        return np.int32
    elif arrow_type == pa.float64():
        return np.float64
    elif arrow_type == pa.float32():
        return np.float32
    elif arrow_type == pa.uint8():
        return np.uint8
    elif arrow_type == pa.uint64():
        return np.uint64
    elif arrow_type == pa.bool_():
        return np.bool_
    elif arrow_type == pa.int16():
        return np.int16
    elif arrow_type == pa.uint16():
        return np.uint16
    elif arrow_type == pa.uint32():
        return np.uint32
    elif arrow_type == pa.int8():
        return np.int8
    elif arrow_type == pa.float16():
        return np.float16
    else:
        raise ValueError("Unsupported pyarrow datatype: %s" % (str(arrow_type)))
