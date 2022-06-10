import os

from .fits_shim import HealSparseFits
from .io_map_fits import _read_map_fits, _write_map_fits, _write_moc_fits
from .parquet_shim import check_parquet_dataset
from .io_map_parquet import _read_map_parquet, _write_map_parquet
from .io_map_healpix import _write_map_healpix


def _read_map(healsparse_class, filename, nside_coverage=None, pixels=None, header=False,
              degrade_nside=None, weightfile=None, reduction='mean', use_threads=False):
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
        Return the fits header or parquet metadata as well as map?  Default is False.
    degrade_nside : `int`, optional
        Degrade map to this nside on read.  None means leave as-is.
    weightfile : `str`, optional
        Floating-point map to supply weights for degrade wmean.  Must
        be a HealSparseMap (weighted degrade not supported for
        healpix degrade-on-read).
    reduction : `str`, optional
        Reduction method with degrade-on-read.
        (mean, median, std, max, min, and, or, sum, prod, wmean).
    use_threads : `bool`, optional
        Use multithreaded reading for parquet files.

    Returns
    -------
    healSparseMap : `HealSparseMap`
        HealSparseMap from file, covered by pixels
    header : `fitsio.FITSHDR` or `astropy.io.fits` (if header=True)
        Fits header for the map file.
    """
    is_fits_file = False
    is_parquet_file = False

    try:
        fits = HealSparseFits(filename)
        is_fits_file = True
        fits.close()
    except OSError:
        pass

    if not is_fits_file:
        is_parquet_file = check_parquet_dataset(filename)

    if is_fits_file:
        return _read_map_fits(healsparse_class, filename, nside_coverage=nside_coverage,
                              pixels=pixels, header=header, degrade_nside=degrade_nside,
                              weightfile=weightfile, reduction=reduction)
    elif is_parquet_file:
        return _read_map_parquet(healsparse_class, filename,
                                 pixels=pixels, header=header, degrade_nside=degrade_nside,
                                 weightfile=weightfile, reduction=reduction,
                                 use_threads=use_threads)
    elif not os.path.isfile(filename):
        raise IOError("Filename %s could not be found." % (filename))
    else:
        raise NotImplementedError("HealSparse only supports fits and parquet files (with pyarrow).")


def _write_map(hsp_map, filename, clobber=False, nocompress=False, format='fits', nside_io=4):
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
        This option only applies if format=``fits``.
    nside_io : `int`, optional
        The healpix nside to partition the output map files in parquet.
        This option only applies if format=``parquet``.
        Must be less than or equal to nside_coverage, and not greater than 16.
    format : `str`, optional
        File format.  May be ``fits``, ``parquet``, or ``healpix``.

    Raises
    ------
    NotImplementedError if file format is not supported.
    ValueError if nside_io is out of range.
    """
    if format == 'fits':
        _write_map_fits(hsp_map, filename, clobber=clobber, nocompress=nocompress)
    elif format == 'parquet':
        _write_map_parquet(hsp_map, filename, clobber=clobber, nside_io=nside_io)
    elif format == 'healpix':
        _write_map_healpix(hsp_map, filename, clobber=clobber)
    else:
        raise NotImplementedError("Only 'fits', 'parquet' and 'healpix' file formats are supported.")


def _write_moc(hsp_map, filename, clobber=False):
    """
    Write the valid pixels of a HealSparseMap to a multi-order component (MOC)
    file.  Note that values of the pixels are not persisted in MOC format.

    Parameters
    ----------
    hsp_map : `healsparse.HealSparseMap`
        HealSparseMap with valid_pixels to output.
    filename : `str`
        Name of file to save
    clobber : `bool`, optional
        Clobber existing file?  Default is False.
    """
    _write_moc_fits(hsp_map, filename, clobber=clobber)
