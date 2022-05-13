import os
import numpy as np
import warnings

from .fits_shim import _make_header, _write_healpix_filename


def _write_map_healpix(hsp_map, filename, clobber=False):
    """
    Internal method to write a HealSparseMap with healpix EXPLICIT format.

    Note that the coverage map is not persisted in this format, and only
    floating point value maps are supported.

    Parameters
    ----------
    hsp_map : `HealSparseMap`
        HealSparseMap to write to a file.
    filename : `str`
        Name of file to save
    clobber : `bool`, optional
        Clobber existing file?  Default is False.

    Raises
    ------
    RuntimeError if file exists and clobber is False.
    NotImplementedError if persisting anything other than a floating point map.
    """

    if hsp_map.is_rec_array:
        raise NotImplementedError("The healpix EXPLICIT output format is not supported for recarray maps.")

    if hsp_map.is_integer_map:
        warnings.warn("Integer maps will be converted to float on read by healpy.", UserWarning)

    if os.path.isfile(filename) and not clobber:
        raise RuntimeError("Filename %s exists and clobber is False." % (filename))

    valid_pixels = hsp_map.valid_pixels

    hdr = _make_header(hsp_map.metadata)
    hdr['PIXTYPE'] = 'HEALPIX'
    hdr['INDXSCHM'] = 'EXPLICIT'
    hdr['ORDERING'] = 'NESTED'
    hdr['NSIDE'] = hsp_map._nside_sparse
    hdr['OBS_NPIX'] = valid_pixels.size
    hdr['BAD_DATA'] = hsp_map._sentinel
    hdr['OBJECT'] = 'PARTIAL'
    hdr['COORDSYS'] = 'C'

    output_struct = np.zeros(valid_pixels.size, dtype=[('PIXEL', 'i8'),
                                                       ('SIGNAL', hsp_map.dtype)])
    output_struct['PIXEL'][:] = valid_pixels
    output_struct['SIGNAL'][:] = hsp_map[valid_pixels]

    _write_healpix_filename(filename, hdr, output_struct)
