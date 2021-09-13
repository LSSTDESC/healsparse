from .fits_shim import HealSparseFits


def _read_coverage_fits(coverage_class, filename_or_fits):
    if isinstance(filename_or_fits, str):
        fits = HealSparseFits(filename_or_fits)
    else:
        fits = filename_or_fits

    try:
        cov_index_map = fits.read_ext_data('COV')
    except (OSError, KeyError):
        raise RuntimeError("File is not a HealSparseMap")

    s_hdr = fits.read_ext_header('SPARSE')

    if isinstance(filename_or_fits, str):
        fits.close()

    return coverage_class(cov_index_map, s_hdr['NSIDE'])
