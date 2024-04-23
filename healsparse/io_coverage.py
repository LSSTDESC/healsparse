import os
import pathlib

from .fits_shim import HealSparseFits
from .io_coverage_fits import _read_coverage_fits
from .io_coverage_parquet import _read_coverage_parquet
from .parquet_shim import check_parquet_dataset


def _read_coverage(coverage_class, filename_or_fits, use_threads=False):
    """
    Internal method to read in a HealSparseCoverage map and check
    file format.

    Parameters
    ----------
    coverage_class : `type`
        Type value of the HealSparseCoverage class.
    filename_or_fits : `str` or `HealSparseFits`
        Name of fits/parquet filename or already open `HealSparseFits`
        object.
    use_threads : `bool`
        Use multithreaded reading for parquet files.  This should not
        be necessary with coverage maps.

    Returns
    -------
    cov_map : `HealSparseCoverage`
        HealSparseCoverage map from file.
    """
    is_fits = False
    is_parquet_file = False

    if isinstance(filename_or_fits, pathlib.PurePath):
        filename_or_fits = os.fspath(filename_or_fits)

    if isinstance(filename_or_fits, str):
        try:
            fits = HealSparseFits(filename_or_fits)
            is_fits = True
            fits.close()
        except OSError:
            is_fits = False

        if not is_fits:
            is_parquet_file = check_parquet_dataset(filename_or_fits)
    elif isinstance(filename_or_fits, HealSparseFits):
        is_fits = True
    else:
        raise NotImplementedError("HealSparse only supports fits and parquet files.")

    if is_fits:
        return _read_coverage_fits(coverage_class, filename_or_fits)
    elif is_parquet_file:
        return _read_coverage_parquet(coverage_class, filename_or_fits, use_threads=use_threads)
    elif not os.path.isfile(filename_or_fits):
        raise IOError("Filename %s could not be found." % (filename_or_fits))
    else:
        raise NotImplementedError("HealSparse only supports fits and parquet files.")
