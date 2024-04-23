import os
import numpy as np
import hpgeom as hpg

from .fits_shim import HealSparseFits, _make_header, _write_filename
from .utils import is_integer_value, _compute_bitshift, reduce_array, WIDE_MASK
from .healSparseCoverage import HealSparseCoverage
from .packedBoolArray import _PackedBoolArray


def _read_map_fits(healsparse_class, filename, nside_coverage=None, pixels=None, header=False,
                   degrade_nside=None, weightfile=None, reduction='mean'):
    """
    Internal function to read in a HealSparseMap from a fits file.

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
    header : `astropy.io.fits.Header` (if header=True)
        Fits header for the map file.
    """
    with HealSparseFits(filename) as fits:
        hdr = fits.read_ext_header(1)

    if 'PIXTYPE' in hdr and hdr['PIXTYPE'].rstrip() == 'HEALPIX':
        if nside_coverage is None:
            raise RuntimeError("Must specify nside_coverage when reading healpix map")

        if weightfile is not None and degrade_nside is not None:
            raise NotImplementedError("Cannot specify a weightfile with degrade-on-read "
                                      "with a healpix map input.")

        if 'ORDERING' not in hdr:
            raise RuntimeError("Required keyword ORDERING not in header.")

        if hdr['ORDERING'].rstrip() == 'NUNIQ':
            healsparse_map = _read_moc_fits(healsparse_class, filename, nside_coverage)

        elif hdr['INDXSCHM'].rstrip() == 'EXPLICIT':
            # This is an explicit (partial) healpix map
            with HealSparseFits(filename) as fits:
                data = fits.read_ext_data(1)

            names = set(data.dtype.names)
            names.remove('PIXEL')
            if len(names) > 1:
                raise NotImplementedError("HealSparse does not support multi-column partial maps.")

            signal_column = list(names)[0]

            if 'BAD_DATA' in hdr:
                sentinel = hdr['BAD_DATA']
            else:
                sentinel = hpg.UNSEEN

            healsparse_map = healsparse_class.make_empty(
                nside_coverage,
                hdr['NSIDE'],
                data[0][signal_column].dtype.type,
                sentinel=sentinel
            )
            if hdr['ORDERING'] == 'RING':
                _pix = hpg.ring_to_nest(hdr['NSIDE'], data['PIXEL'])
            else:
                _pix = data['PIXEL']
            healsparse_map[_pix] = data[signal_column]
        elif hdr['INDXSCHM'].rstrip() == 'IMPLICIT':
            with HealSparseFits(filename) as fits:
                data = fits.read_ext_data(1)
                if "T" in data.dtype.names:
                    field_name = "T"
                elif "TEMPERATURE" in data.dtype.names:
                    field_name = "TEMPERATURE"
                else:
                    raise RuntimeError("Healpix file does not comply with standards.")

            # Ravel the data into one contiguous array.
            data = data[field_name].ravel()

            if hdr["ORDERING"] == "RING":
                data = hpg.reorder(data, ring_to_nest=True)

            # Convert to healsparse format
            healsparse_map = healsparse_class(healpix_map=data,
                                              nside_coverage=nside_coverage,
                                              nest=True)
        else:
            raise ValueError(f"Illegal value for INDXSCHM: {hdr['INDXSCHM']}")

        if degrade_nside is not None:
            # Degrade this map.  Note that this could not be done on read
            # because healpix maps do not have that functionality.
            healsparse_map = healsparse_map.degrade(degrade_nside, reduction=reduction)

        if header:
            return (healsparse_map, hdr)
        else:
            return healsparse_map
    elif 'PIXTYPE' in hdr and hdr['PIXTYPE'].rstrip() == 'HEALSPARSE':
        if degrade_nside is None:
            cov_map, sparse_map, nside_sparse, primary, sentinel = \
                _read_healsparse_fits_file(filename, pixels=pixels)

            if 'WIDEMASK' in hdr and hdr['WIDEMASK']:
                sparse_map = sparse_map.reshape((sparse_map.size // hdr['WWIDTH'],
                                                 hdr['WWIDTH'])).astype(WIDE_MASK)
        else:
            # Read with degrade-on-read code
            cov_map, sparse_map, nside_sparse, primary, sentinel = \
                _read_healsparse_fits_file_and_degrade(filename, pixels,
                                                       degrade_nside, reduction,
                                                       weightfile)

        healsparse_map = healsparse_class(cov_map=cov_map, sparse_map=sparse_map,
                                          nside_sparse=nside_sparse, primary=primary, sentinel=sentinel,
                                          metadata=hdr)

        if header:
            return (healsparse_map, hdr)
        else:
            return healsparse_map

    elif 'MOCVERS' in hdr:
        if 'ORDERING' not in hdr:
            raise RuntimeError("MOC file %s has illegal header, missing ORDERING." % (filename))

        if hdr['ORDERING'] != 'NUNIQ':
            raise RuntimeError("MOC file %s has %s ordering; only NUNIQ supported by healsparse." %
                               (filename, hdr['ORDERING']))

        healsparse_map = _read_moc_fits(healsparse_class, filename, nside_coverage)

        if header:
            return (healsparse_map, hdr)
        else:
            return healsparse_map
    else:
        raise RuntimeError("Filename %s not in healpix or healsparse format." % (filename))


def _read_moc_fits(healsparse_class, filename, nside_coverage):
    """Read a MOC fits file.  Only supports V1 now.

    Parameters
    ----------
    healsparse_class : `type`
        Type value of the HealSparseMap class.
    filename : `str`
    nside_coverage : `int`

    Returns
    -------
    healsparse_map : `HealSparseMap`
    """
    # This is a MOC file, NUNIQ ordering.
    with HealSparseFits(filename) as fits:
        data = fits.read_ext_data(1)

    order = np.floor(np.log2(data['UNIQ']//4)).astype(np.int32)//2
    index = data['UNIQ'] - 4*(4**order)

    max_order = np.max(order)

    healsparse_map = healsparse_class.make_empty(nside_coverage,
                                                 2**max_order,
                                                 dtype=bool)

    # This is a very simple algorithm for unpacking the UNIQ
    # pixels.  This can be optimized later if necessary.
    pixel_arrays = []
    for uniq_order, uniq_index in zip(order, index):
        pixel_arrays.append(np.left_shift(uniq_index, 2*(max_order - uniq_order)) +
                            np.arange(4**(max_order - uniq_order)))
    pixels = np.concatenate(pixel_arrays)
    healsparse_map[np.sort(pixels)] = True

    return healsparse_map


def _read_healsparse_fits_file(filename, pixels=None):
    """
    Read a healsparse file, optionally with a set of coverage pixels.

    Parameters
    ----------
    filename : `str`
        Name of the file to read.
    pixels : `list`, optional
        List of integer pixels from the coverage map

    Returns
    -------
    cov_map : `HealSparseCoverage`
        Coverage map with index values
    sparse_map : `np.ndarray`
        Sparse map with map dtype
    nside_sparse : `int`
        Nside of the coverage map
    primary : `str`
        Primary key field for recarray map.  Default is None.
    sentinel : `float` or `int`
        Sentinel value for null.  Usually UNSEEN
    """
    cov_map = HealSparseCoverage.read(filename)
    primary = None

    if pixels is None:
        # Read the full map
        with HealSparseFits(filename) as fits:
            sparse_map = fits.read_ext_data('SPARSE')
            s_hdr = fits.read_ext_header('SPARSE')
        nside_sparse = s_hdr['NSIDE']
        if 'PRIMARY' in s_hdr:
            primary = s_hdr['PRIMARY'].rstrip()
        # If SENTINEL is not there then it should be UNSEEN
        if 'SENTINEL' in s_hdr:
            sentinel = s_hdr['SENTINEL']
        else:
            sentinel = hpg.UNSEEN
        if 'RESHAPED' in s_hdr and s_hdr['RESHAPED']:
            # Unravel and reshaped maps.
            sparse_map = sparse_map.ravel()
    else:
        _pixels = np.atleast_1d(pixels)
        if len(np.unique(_pixels)) < len(_pixels):
            raise RuntimeError("Input list of pixels must be unique.")

        # Which pixels are in the coverage map?
        cov_pix, = np.where(cov_map.coverage_mask)
        sub = np.clip(np.searchsorted(cov_pix, _pixels), 0, cov_pix.size - 1)
        ok, = np.where(cov_pix[sub] == _pixels)
        if ok.size == 0:
            raise RuntimeError("None of the specified pixels are in the coverage map.")
        _pixels = np.sort(_pixels[ok])

        # Read part of a map
        with HealSparseFits(filename) as fits:
            s_hdr = fits.read_ext_header('SPARSE')

            nside_sparse = s_hdr['NSIDE']
            nside_coverage = cov_map.nside_coverage

            if 'SENTINEL' in s_hdr:
                sentinel = s_hdr['SENTINEL']
            else:
                sentinel = hpg.UNSEEN

            if not fits.ext_is_image('SPARSE'):
                # This is a table extension
                primary = s_hdr['PRIMARY'].rstrip()

            if 'WIDEMASK' in s_hdr and s_hdr['WIDEMASK']:
                wmult = s_hdr['WWIDTH']
            else:
                wmult = 1

            if 'BITPACK' in s_hdr and s_hdr['BITPACK']:
                wdiv = 8
            else:
                wdiv = 1

            if 'RESHAPED' in s_hdr and s_hdr['RESHAPED']:
                reshaped = True
            else:
                reshaped = False

            # This is the map without the offset
            cov_index_map_temp = cov_map[:] + np.arange(hpg.nside_to_npixel(nside_coverage),
                                                        dtype=np.int64)*cov_map.nfine_per_cov

            # It is not 100% sure this is the most efficient way to read in,
            # but it does work.
            sparse_map = np.zeros((_pixels.size + 1)*cov_map.nfine_per_cov*wmult//wdiv,
                                  dtype=fits.get_ext_dtype('SPARSE'))
            # Read in the overflow bin
            if not reshaped:
                row_range = [0, cov_map.nfine_per_cov*wmult//wdiv]
                col_range = None
            else:
                row_range = [0, cov_map.nfine_per_cov*wmult//wdiv]
                col_range = [0, 1]

            # if reshaped:
            #     print("About to read ... ")
            sparse_map[0: cov_map.nfine_per_cov*wmult//wdiv] = \
                fits.read_ext_data('SPARSE',
                                   row_range=row_range, col_range=col_range).ravel()
            # And read in the pixels
            for i, pix in enumerate(_pixels):
                if not reshaped:
                    row_range = [cov_index_map_temp[pix]*wmult//wdiv,
                                 (cov_index_map_temp[pix] + cov_map.nfine_per_cov)*wmult//wdiv]
                    col_range = None
                else:
                    row_range = [0, cov_map.nfine_per_cov*wmult//wdiv]
                    col_start = cov_index_map_temp[pix] // cov_map.nfine_per_cov
                    col_range = [col_start, col_start + 1]

                sparse_map[(i + 1)*cov_map.nfine_per_cov*wmult//wdiv:
                           (i + 2)*cov_map.nfine_per_cov*wmult//wdiv] = fits.read_ext_data(
                               'SPARSE',
                               row_range=row_range, col_range=col_range).ravel()

            # Set the coverage index map for the pixels that we read in
            cov_map = HealSparseCoverage.make_from_pixels(nside_coverage,
                                                          nside_sparse,
                                                          _pixels)

    if isinstance(sentinel, bool):
        # Convert back to boolean or bit_packed
        if 'BITPACK' in s_hdr and s_hdr['BITPACK']:
            sparse_map = _PackedBoolArray(data_buffer=sparse_map)
        else:
            sparse_map = sparse_map.astype(bool)

    return cov_map, sparse_map, nside_sparse, primary, sentinel


def _read_healsparse_fits_file_and_degrade(filename, pixels, nside_out, reduction, weightfile):
    """
    Read a healsparse file, and degrade on read.

    Parameters
    ----------
    filename : `str`
        Name of the file to read.
    pixels : `list`
        List of integer pixels from the coverage map.  May be None (full map).
    nside_out : `int`
        Degrade map to this nside on read.
    reduction : `str`
        Reduction method with degrade-on-read.
        (mean, median, std, max, min, and, or, sum, prod, wmean).
    weightfile : `str`
        File containing weights.  May be None (no weights).

    Returns
    -------
    healsparse_map : `HealSparseMap`
    """
    cov_map = HealSparseCoverage.read(filename)
    primary = None

    if pixels is None:
        # When doing degrade-on-read, we must read in pixel-by-pixel,
        # so we get all the pixels.
        _pixels, = np.where(cov_map.coverage_mask)
    else:
        _pixels = np.atleast_1d(pixels)
        if len(np.unique(_pixels)) < len(_pixels):
            raise RuntimeError("Input list of pixels must be unique.")

        # Which pixels are in the coverage map?
        cov_pix, = np.where(cov_map.coverage_mask)
        sub = np.clip(np.searchsorted(cov_pix, _pixels), 0, cov_pix.size)
        ok, = np.where(cov_pix[sub] == _pixels)
        if ok.size == 0:
            raise RuntimeError("None of the specified pixels are in the coverage map.")
        _pixels = np.sort(_pixels[ok])

    # If we have a weight map, check that it conforms to the map we want to degrade.
    use_weightfile = False
    if weightfile is not None and reduction == 'wmean':
        cov_map_weight = HealSparseCoverage.read(weightfile)
        if cov_map_weight.nside_coverage != cov_map.nside_coverage:
            raise ValueError("The weightfile %s must have same coverage nside." % (weightfile))
        cov_pix_weight, = np.where(cov_map_weight.coverage_mask)
        if not np.all(np.in1d(_pixels, cov_pix_weight)):
            raise ValueError("The weightfile %s must have coverage in all the "
                             "pixels to read." % (weightfile))
        use_weightfile = True
    elif weightfile is not None:
        raise Warning('Weightfile specified but wmean reduction mode is not set.  Ignoring weightfile')

    nside_coverage = cov_map.nside_coverage

    cov_map_out = HealSparseCoverage.make_from_pixels(nside_coverage,
                                                      nside_out,
                                                      _pixels)
    # This is the map without the offset
    cov_index_out_temp = cov_map_out[:] + np.arange(hpg.nside_to_npixel(nside_coverage),
                                                    dtype=np.int64)*cov_map_out.nfine_per_cov
    with HealSparseFits(filename) as fits:
        s_hdr = fits.read_ext_header('SPARSE')

        nside_sparse = s_hdr['NSIDE']

        if nside_out >= nside_sparse:
            raise ValueError('Degrade nside (%d) is not smaller than sparse nside (%d)' %
                             (nside_out, nside_sparse))

        if 'SENTINEL' in s_hdr:
            sentinel = s_hdr['SENTINEL']
        else:
            sentinel = hpg.UNSEEN

        if not fits.ext_is_image('SPARSE'):
            # This is a table extension
            is_rec_array = True
            primary = s_hdr['PRIMARY'].rstrip()
        else:
            is_rec_array = False

        if 'WIDEMASK' in s_hdr and s_hdr['WIDEMASK']:
            wmult = s_hdr['WWIDTH']
            is_wide_mask = True
        else:
            wmult = 1
            is_wide_mask = False

        if 'BITPACK' in s_hdr and s_hdr['BITPACK']:
            raise NotImplementedError("degrade on read does not support bit_packed maps.")

        reshaped = False
        col_range = None
        if 'RESHAPED' in s_hdr and s_hdr['RESHAPED']:
            reshaped = True
            col_range = [0, 1]

        dtype = np.dtype(fits.get_ext_dtype('SPARSE'))

        # Check weight map
        if use_weightfile:
            wfits = HealSparseFits(weightfile)
            s_hdr_weight = fits.read_ext_header('SPARSE')
            dtype_weight = fits.get_ext_dtype('SPARSE')
            testval = np.zeros(1, dtype=dtype_weight)[0]
            if 'SENTINEL' in s_hdr_weight:
                sentinel_weight = s_hdr_weight['SENTINEL']
            else:
                sentinel_weight = hpg.UNSEEN
            if ((s_hdr_weight['NSIDE'] != nside_sparse or
                 not fits.ext_is_image('SPARSE') or
                 'WIDEMASK' in s_hdr_weight or
                 is_integer_value(testval))):
                wfits.close()
                raise ValueError("Weights must be a floating-point map with same "
                                 "nside as map to degrade.")

        bit_shift_out = _compute_bitshift(nside_coverage, nside_out)
        nfine_per_cov_out = 2**bit_shift_out

        if is_wide_mask:
            if reduction not in ['and', 'or']:
                if use_weightfile:
                    wfits.close()
                raise NotImplementedError('Cannot degrade a wide_mask map with any operation '
                                          'except for and/or')
            sentinel_out = sentinel
            dtype_out = dtype
            sparse_map_out = np.zeros(((_pixels.size + 1)*nfine_per_cov_out, wmult),
                                      dtype=dtype_out)
        elif is_rec_array:
            dtype_out = []
            sentinel_out = hpg.UNSEEN
            # We should avoid integers
            test_arr = np.zeros(1, dtype=dtype)
            for key, value in dtype.fields.items():
                if issubclass(test_arr[key].dtype.type, np.integer):
                    dtype_out.append((key, np.float64))
                else:
                    dtype_out.append((key, value[0]))
            dtype_out = np.dtype(dtype_out)
            sparse_map_out = np.zeros((_pixels.size + 1)*nfine_per_cov_out,
                                      dtype=dtype_out)
            sparse_map_out[primary] = sentinel_out
        elif (issubclass(dtype.type, np.integer) and (reduction in ['and', 'or'])):
            sentinel_out = sentinel
            dtype_out = dtype
            sparse_map_out = np.full((_pixels.size + 1)*nfine_per_cov_out,
                                     sentinel_out,
                                     dtype=dtype_out)
        else:
            if issubclass(dtype.type, np.integer):
                dtype_out = np.dtype(np.float64)
            else:
                dtype_out = dtype
            sentinel_out = hpg.UNSEEN
            sparse_map_out = np.full((_pixels.size + 1)*nfine_per_cov_out,
                                     sentinel_out,
                                     dtype=dtype_out)

        # This is the map without the offset
        cov_index_map_temp = cov_map[:] + np.arange(hpg.nside_to_npixel(nside_coverage),
                                                    dtype=np.int64)*cov_map.nfine_per_cov
        if use_weightfile:
            cov_index_map_temp_weight = (cov_map_weight[:] +
                                         np.arange(hpg.nside_to_npixel(nside_coverage),
                                                   dtype=np.int64)*cov_map.nfine_per_cov)

        for i, pix in enumerate(_pixels):
            row_range = [cov_index_map_temp[pix]*wmult,
                         (cov_index_map_temp[pix] + cov_map.nfine_per_cov)*wmult]
            pix_data = fits.read_ext_data('SPARSE', row_range=row_range, col_range=col_range)
            if reshaped:
                pix_data = pix_data.ravel()

            if use_weightfile:
                row_range_weight = [cov_index_map_temp_weight[pix],
                                    (cov_index_map_temp_weight[pix] + cov_map.nfine_per_cov)]
                weight_values = wfits.read_ext_data('SPARSE', row_range=row_range_weight, col_range=col_range)
                if reshaped:
                    weight_values = weight_values.ravel()
                weight_values[weight_values == sentinel_weight] = 0.0
                weight_values = weight_values.reshape((1,
                                                       (nside_out//nside_coverage)**2, -1))
            else:
                weight_values = None

            if is_wide_mask:
                aux = pix_data.reshape((1, (nside_out//nside_coverage)**2, -1, wmult))
                aux = reduce_array(aux, reduction=reduction, axis=2).reshape((-1, wmult))
            elif is_rec_array:
                aux = np.zeros(cov_map_out.nfine_per_cov, dtype=dtype_out)
                for key, value in sparse_map_out.dtype.fields.items():
                    auxf = pix_data[key].astype(np.float64)
                    auxf[pix_data[key] == sentinel] = np.nan
                    auxf = auxf.reshape((1, (nside_out//nside_coverage)**2, -1))
                    auxf = reduce_array(auxf, reduction=reduction, weights=weight_values)
                    auxf[np.isnan(auxf)] = sentinel_out
                    aux[key] = auxf
            elif issubclass(dtype_out.type, np.integer):
                # No weights because this is going to be a bit-wise operation.
                aux = pix_data.reshape((1, (nside_out//nside_coverage)**2, -1))
                aux = reduce_array(aux, reduction=reduction)
            else:
                aux = pix_data.astype(dtype_out)
                aux[pix_data == sentinel] = np.nan
                aux = aux.reshape((1, (nside_out//nside_coverage)**2, -1))
                aux = reduce_array(aux, reduction=reduction, weights=weight_values)
                aux[np.isnan(aux)] = sentinel_out

            sparse_map_out[cov_index_out_temp[pix]:
                           cov_index_out_temp[pix] + cov_map_out.nfine_per_cov] = aux

    if use_weightfile:
        wfits.close()

    return cov_map_out, sparse_map_out, nside_out, primary, sentinel_out


def _write_map_fits(hsp_map, filename, clobber=False, nocompress=False):
    """
    Internal method to write a HealSparseMap to a fits file.
    Use the `metadata` property from the map to persist additional
    information in the fits header.

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

    Raises
    ------
    RuntimeError if file exists and clobber is False.
    """
    if os.path.isfile(filename) and not clobber:
        raise RuntimeError("Filename %s exists and clobber is False." % (filename))

    # Note that we put the requested header information in each of the extensions.
    c_hdr = _make_header(hsp_map.metadata)
    c_hdr['PIXTYPE'] = 'HEALSPARSE'
    c_hdr['NSIDE'] = hsp_map.nside_coverage

    s_hdr = _make_header(hsp_map.metadata)
    s_hdr['PIXTYPE'] = 'HEALSPARSE'
    s_hdr['NSIDE'] = hsp_map._nside_sparse
    s_hdr['SENTINEL'] = hsp_map._sentinel
    if hsp_map._is_rec_array:
        s_hdr['PRIMARY'] = hsp_map._primary
    if hsp_map._is_wide_mask:
        s_hdr['WIDEMASK'] = hsp_map._is_wide_mask
        s_hdr['WWIDTH'] = hsp_map._wide_mask_width
        # Wide masks can be compressed.
        _write_filename(filename, c_hdr, s_hdr, hsp_map._cov_map[:], hsp_map._sparse_map.ravel(),
                        compress=not nocompress,
                        compress_tilesize=hsp_map._wide_mask_width*hsp_map._cov_map.nfine_per_cov)
    elif hsp_map._is_bit_packed:
        s_hdr['BITPACK'] = hsp_map._is_bit_packed
        # Bit array masks can be compressed.
        _write_filename(filename, c_hdr, s_hdr, hsp_map._cov_map[:], hsp_map._sparse_map.data_array,
                        compress=not nocompress,
                        compress_tilesize=hsp_map._cov_map.nfine_per_cov // 8)
    elif hsp_map._sparse_map[0].dtype == np.bool_:
        # We must convert boolean maps to int16 maps for fits storage.
        _write_filename(filename, c_hdr, s_hdr, hsp_map._cov_map[:], hsp_map._sparse_map.astype(np.int16),
                        compress=not nocompress,
                        compress_tilesize=hsp_map._cov_map.nfine_per_cov)

    elif ((hsp_map.is_integer_map and hsp_map._sparse_map[0].dtype.itemsize < 8) or
          (not hsp_map.is_integer_map and not hsp_map._is_rec_array)):
        # Integer maps < 64 bit (8 byte) can be compressed, as can
        # floating point maps
        _write_filename(filename, c_hdr, s_hdr, hsp_map._cov_map[:], hsp_map._sparse_map,
                        compress=not nocompress,
                        compress_tilesize=hsp_map._cov_map.nfine_per_cov)
    else:
        # All other maps are not compressed.
        _write_filename(filename, c_hdr, s_hdr, hsp_map._cov_map[:], hsp_map._sparse_map)


def _write_moc_fits(hsp_map, filename, clobber=False):
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
    import astropy.io.fits as fits

    max_order = int(np.round(np.log2(hsp_map.nside_sparse)))
    min_uniq_order = int(np.round(np.log2(hsp_map.nside_coverage)))

    pixels = hsp_map.valid_pixels

    uniq = 4*(4**max_order) + pixels

    uniq_map = hsp_map.make_empty(hsp_map.nside_coverage, hsp_map.nside_sparse, dtype=np.float32)
    uniq_map[pixels] = 1.0

    # Loop over orders, degrade each time, and look for pixels with full coverage.
    for uniq_order in range(max_order - 1, min_uniq_order - 1, -1):
        uniq_map = uniq_map.degrade(2**uniq_order, reduction='sum')
        pix_shift = np.right_shift(pixels, 2*(max_order - uniq_order))
        # Check if any of the pixels at uniq_order have full coverage.
        covered, = np.isclose(uniq_map[pix_shift], 4**(max_order - uniq_order)).nonzero()
        if covered.size == 0:
            # No pixels at uniq_order are fully covered, we're done.
            break
        # Replace the UNIQ pixels that are fully covered
        uniq[covered] = 4*(4**uniq_order) + pix_shift[covered]

    # Remove duplicate pixels
    uniq = np.unique(uniq)

    # Output to fits
    tbl = np.zeros(uniq.size, dtype=[('UNIQ', 'i8')])
    tbl['UNIQ'][:] = uniq

    order = np.log2(tbl['UNIQ']//4).astype(np.int32)//2
    moc_order = np.max(order)

    hdu = fits.BinTableHDU(tbl)
    hdu.header['PIXTYPE'] = 'HEALPIX'
    hdu.header['ORDERING'] = 'NUNIQ'
    hdu.header['COORDSYS'] = 'C'
    hdu.header['MOCORDER'] = moc_order
    hdu.header['MOCTOOL'] = 'healsparse'
    hdu.header['MOCVERS'] = '1.1'

    hdu.writeto(filename, overwrite=clobber)

    # Work around a bug in cds-moc-rust to change the TFORM1 header
    # value from K to 1K.
    import mmap

    with open(filename, "r+b") as f:
        try:
            mm = mmap.mmap(f.fileno(), 0)
            loc = mm.find(b"TFORM1  = 'K       '")
            if loc >= 0:
                mm.seek(loc)
                mm.write(b"TFORM1  = '1K      '")
        except OSError:
            # Some systems do not have the mmap available,
            # we need to read in the full file.
            data = f.read()
            loc = data.find(b"TFORM1  = 'K       '")
            if loc >= 0:
                f.seek(loc)
                f.write(b"TFORM1  = '1K      '")
