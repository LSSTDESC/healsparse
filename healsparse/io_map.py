from .fits_shim import HealSparseFits
from .io_map_fits import _read_map_fits, _write_map_fits


def _read_map(healsparse_class, filename, nside_coverage=None, pixels=None, header=False,
              degrade_nside=None, weightfile=None, reduction='mean'):
    """
    Internal function to check the map filetype and read in a HealSparseMap.

    Parameters
    ----------
    healsparse_class : `type`
        Type value of the HealSparseMap class.
    filename : `str`
        Name of the file to read.  May be either a regular HEALPIX
        map or a HealSparseMap
    nside_coverage : `int`, optional
        Nside of coverage map to generate if input file is healpix map.
    pixels : `list`, optional
        List of coverage map pixels to read.  Only used if input file
        is a HealSparseMap
    header : `bool`, optional
        Return the fits header metadata as well as map?  Default is False.
    degrade_nside : `int`, optional
        Degrade map to this nside on read.  None means leave as-is.
    weightfile : `str`, optional
        Floating-point map to supply weights for degrade wmean.  Must
        be a HealSparseMap (weighted degrade not supported for
        healpix degrade-on-read).
    reduction : `str`, optional
        Reduction method with degrade-on-read.
        (mean, median, std, max, min, and, or, sum, prod, wmean).

    Returns
    -------
    healSparseMap : `HealSparseMap`
        HealSparseMap from file, covered by pixels
    header : `fitsio.FITSHDR` or `astropy.io.fits` (if header=True)
        Fits header for the map file.
    """
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
    """
    Internal method to write a HealSparseMap to a file, and check formats.
    Use the `metadata` property from
    the map to persist additional information in the fits header.

    Parameters
    ----------
    hsp_map : `HealSparseMap`
        HealSparseMap to write to a file.
    filename : `str`
        Name of file to save
    clobber : `bool`, optional
        Clobber existing file?  Default is False.
    nocompress : `bool`, optional
        If this is False, then integer maps will be compressed losslessly.
        Note that `np.int64` maps cannot be compressed in the FITS standard.
        This option only applies if format='fits'.
    format : `str`, optional
        File format.  Currently only 'fits' is supported.

    Raises
    ------
    NotImplementedError if file format is not supported.
    """
    if format == 'fits':
        _write_map_fits(hsp_map, filename, clobber=clobber, nocompress=nocompress)
    else:
        raise NotImplementedError("Only 'fits' file format is supported.")
