from .fits_shim import HealSparseFits
from .io_map_fits import _read_map_fits, _write_map_fits


def _read_map(healsparse_class, filename, nside_coverage=None, pixels=None, header=False,
              degrade_nside=None, weightfile=None, reduction='mean'):
    is_fits_file = False
    try:
        fits = HealSparseFits(filename)
        is_fits_file = True
        fits.close()
    except OSError:
        pass

    if is_fits_file:
        return _read_map_fits(healsparse_class, filename, nside_coverage=nside_coverage,
                              pixels=pixels, header=header, degrade_nside=degrade_nside,
                              weightfile=weightfile, reduction=reduction)
    else:
        raise NotImplementedError("HealSparse only supports fits files, and %s is not a valid fits file."
                                  % (filename))


def _write_map(hsp_map, filename, clobber=False, nocompress=False, format='fits'):
    if format == 'fits':
        _write_map_fits(hsp_map, filename, clobber=clobber, nocompress=nocompress)
    else:
        raise NotImplementedError("Only 'fits' file format is supported.")
