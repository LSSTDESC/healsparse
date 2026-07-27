import fsspec
import warnings

use_hdf5 = False
try:
    import h5py

    use_hdf5 = True
except ImportError:
    pass


def check_hdf5_file(filepath):
    """
    Check if a filepath points to an hdf5 file

    Parameters
    ----------
    filepath : `str`
        File path to check.

    Returns
    -------
    is_hdf5_file : `bool`
        True if it is an hdf5 file.

    Raises
    ------
    Warns if hdf5 is not installed.
    """
    if not use_hdf5:
        warnings.warn("Cannot access hdf5 files without h5py", UserWarning)
        return False

    HDF5_SIG = b"\x89HDF\r\n\x1a\n"

    fs, path = fsspec.core.url_to_fs(filepath)

    if fsspec.utils.get_protocol(filepath) == "file":
        # Use native h5py checking.

        return h5py.is_hdf5(path)
    else:
        # Use fsspec to get first 8 bytes.

        return fs.cat_file(path, start=0, end=8) == HDF5_SIG
