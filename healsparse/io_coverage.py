from .fits_shim import HealSparseFits
from .io_coverage_fits import _read_coverage_fits


def _read_coverage(coverage_class, filename_or_fits):
    """
    Internal method to read in a HealSparseCoverage map and check
    file format.

    Parameters
    ----------
    coverage_class : `type`
        Type value of the HealSparseCoverage class.
    filename_or_fits : `str` or `HealSparseFits`
        Name of filename or already open `HealSparseFits` object.

    Returns
    -------
    cov_map : `HealSparseCoverage`
        HealSparseCoverage map from file.
    """
    is_fits = False
    if isinstance(filename_or_fits, str):
        try:
            fits = HealSparseFits(filename_or_fits)
            is_fits = True
            fits.close()
        except OSError:
            is_fits = False
    elif isinstance(filename_or_fits, HealSparseFits):
        is_fits = True
    else:
        raise NotImplementedError("HealSparse only supports fits files.")

    if is_fits:
        return _read_coverage_fits(coverage_class, filename_or_fits)
