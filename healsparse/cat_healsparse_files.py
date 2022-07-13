import numpy as np
import hpgeom as hpg
import os

from .healSparseMap import HealSparseMap
from .healSparseCoverage import HealSparseCoverage
from .fits_shim import HealSparseFits
from .utils import _compute_bitshift, WIDE_NBIT, WIDE_MASK


def cat_healsparse_files(file_list, outfile, check_overlap=False, clobber=False,
                         in_memory=False, nside_coverage_out=None, or_overlap=False):
    """
    Concatenate healsparse files together in a memory-efficient way.

    Parameters
    ----------
    file_list : `list` of `str`
       List of filenames to concatenate
    outfile : `str`
       Output filename
    check_overlap : `bool`, optional
       Check that each file has a unique sparse map.  This may be slower.
    clobber : `bool`, optional
       Clobber existing outfile
    in_memory : `bool`, optional
       Do operations in-memory (required unless fitsio is available).
    nside_coverage_out : `int`, optional
       Output map with specific nside_coverage.  Default is nside_coverage
       of first map in file_list.
    or_overlap: `bool`, optional
       If True compute the `or` overlap of two integer maps when concatenating.
    """
    if os.path.isfile(outfile) and not clobber:
        raise RuntimeError("File %s already exists and clobber is False" % (outfile))

    if or_overlap and not check_overlap:
        check_overlap = True
        raise RuntimeWarning("""or_overlap is True and check_overlap is False,
                             will check overlap""")
    # Read in all the coverage maps
    cov_mask_summary = None
    nside_sparse = None
    nside_coverage_maps = []
    bit_shift_covs = []
    cov_index_maps = []
    cov_map_nfine_per_covs = []
    for i, f in enumerate(file_list):
        cov_map = HealSparseCoverage.read(f)

        cov_index_map = cov_map[:] + np.arange(hpg.nside_to_npixel(cov_map.nside_coverage),
                                               dtype=np.int64)*cov_map.nfine_per_cov
        cov_index_maps.append(cov_index_map)
        cov_map_nfine_per_covs.append(cov_map.nfine_per_cov)

        if cov_mask_summary is None:
            if nside_coverage_out is None:
                nside_coverage_out = cov_map.nside_coverage

            cov_mask_summary = np.zeros((len(file_list), hpg.nside_to_npixel(nside_coverage_out)),
                                        dtype=np.bool_)
            nside_sparse = cov_map.nside_sparse
        else:
            if cov_map.nside_sparse != nside_sparse:
                # This requirement cannot be relaxed
                raise RuntimeError("Map %s has a different nside_sparse (%d)" %
                                   (cov_map.nside_sparse))

        if cov_map.nside_coverage == nside_coverage_out:
            # Straight copy
            cov_mask_summary[i, :] = cov_map.coverage_mask
            bit_shift_cov = 0
        elif cov_map.nside_coverage < nside_coverage_out:
            # cov_map.nside_coverage < nside_coverage_out
            # Output coverage is finer
            bit_shift_cov = _compute_bitshift(cov_map.nside_coverage, nside_coverage_out)

            # We need to know the full map coverage to know which of the fine pixel
            # are actually covered.  So this necessitates reading in the map here,
            # even though we will read it again later.  I don't know what will happen
            # if you try to change the coverage resolution on a giant map.

            m = HealSparseMap.read(f)
            valid_pixels = m.valid_pixels

            bit_shift = _compute_bitshift(nside_coverage_out, nside_sparse)
            cov_mask_pixels_new = np.unique(np.right_shift(valid_pixels, bit_shift))

            cov_mask_summary[i, cov_mask_pixels_new] = True
        else:
            # cov_map.nside_coverage > nside_coverage_out
            # Output coverage is coarser

            bit_shift_cov = _compute_bitshift(nside_coverage_out, cov_map.nside_coverage)
            cov_mask_pixels, = np.where(cov_map.coverage_mask)
            cov_mask_pixels_new = np.unique(np.right_shift(cov_mask_pixels, bit_shift_cov))
            cov_mask_summary[i, cov_mask_pixels_new] = True

        nside_coverage_maps.append(cov_map.nside_coverage)
        bit_shift_covs.append(bit_shift_cov)

    # Combine for an overall coverage map
    # Sum across axis=0 to know which coverage pixels are there
    cov_pix, = np.where(cov_mask_summary.sum(axis=0) > 0)

    # The cov_map will only work after the full map has been written out
    cov_map = HealSparseCoverage.make_from_pixels(nside_coverage_out, nside_sparse, cov_pix)

    # We need to create a stub of a sparse map (the overflow), with the correct dtype
    with HealSparseFits(file_list[0]) as fits:
        s_hdr = fits.read_ext_header('SPARSE')

        if 'SENTINEL' in s_hdr:
            sentinel = s_hdr['SENTINEL']
        else:
            sentinel = hpg.UNSEEN

        if not fits.ext_is_image('SPARSE'):
            # This is a table extension
            primary = s_hdr['PRIMARY'].rstrip()
        else:
            primary = None

        if 'WWIDTH' in s_hdr:
            wide_mask_maxbits = WIDE_NBIT*s_hdr['WWIDTH']
            wmult = s_hdr['WWIDTH']
        else:
            wide_mask_maxbits = None
            wmult = 1

        sparse_stub = fits.read_ext_data('SPARSE',
                                         row_range=[0, cov_map.nfine_per_cov*wmult])
        if wide_mask_maxbits is not None:
            sparse_stub = np.reshape(sparse_stub, (cov_map.nfine_per_cov, wmult))
            # This fixes a bug in astropy<4.0
            sparse_stub = sparse_stub.astype(WIDE_MASK)

    if not in_memory:
        # When spooling to disk, we need a stub to write (with the final cov_map)
        # to append to.  The stub map cannot have compression turned on,
        # or else appending doesn't work.

        stub_map = HealSparseMap(cov_map=cov_map,
                                 sparse_map=sparse_stub, nside_sparse=nside_sparse,
                                 primary=primary, sentinel=sentinel)

        # And write this out to a temporary filename
        outfile_temp = outfile + '.incomplete'
        stub_map.write(outfile_temp, clobber=True, nocompress=True)

        try:
            outfits = HealSparseFits(outfile_temp, mode='rw')
        except RuntimeError:
            raise RuntimeError("Running cat_healsparse_files with in_memory=False requires fitsio.")
    else:
        # When building in memory, we just need a blank map
        sparse_map = HealSparseMap.make_empty(nside_coverage_out, nside_sparse,
                                              sparse_stub.dtype, primary=primary,
                                              sentinel=sentinel, wide_mask_maxbits=wide_mask_maxbits)

    # Load in pointers to all the input fits files
    fitses = []
    for f in file_list:
        fitses.append(HealSparseFits(f))
    sparse_map_temp_matchcov = None

    # And prepare to append, coverage pixel by coverage pixel!
    for pix in cov_pix:
        # Figure out which input files overlap this coverage pixel
        u_cov_pix, = np.where(cov_mask_summary[:, pix])

        if not in_memory:
            # We need a temporary sparse_map
            sparse_map = HealSparseMap.make_empty(nside_coverage_out, nside_sparse,
                                                  sparse_stub.dtype, primary=primary,
                                                  sentinel=sentinel, wide_mask_maxbits=wide_mask_maxbits)

        # Read in each of these files and set to the sparse_map.  This will either
        # be the full map (in_memory) or the temp map (not in_memory)
        for index in u_cov_pix:
            if nside_coverage_maps[index] == nside_coverage_out:
                # Straightforward -- matched coverage
                in_map = _read_partial_sparsemap(fitses[index], cov_map_nfine_per_covs[index],
                                                 wmult, cov_index_maps[index], np.array([pix]),
                                                 sparse_stub.dtype, wide_mask_maxbits,
                                                 nside_coverage_maps[index], nside_sparse,
                                                 sparse_map_temp_input=sparse_map_temp_matchcov,
                                                 primary=primary, sentinel=sentinel)
                if sparse_map_temp_matchcov is None:
                    # Save this for caching
                    sparse_map_temp_matchcov = in_map._sparse_map.ravel()

                valid_pixels = in_map.valid_pixels

            elif nside_coverage_maps[index] < nside_coverage_out:
                # nside_coverage_maps[index] < nside_coverage_out
                # Output coverage is finer, which means we just need to know
                # the one coarse pix to read in here.

                in_map = _read_partial_sparsemap(fitses[index], cov_map_nfine_per_covs[index],
                                                 wmult, cov_index_maps[index],
                                                 np.right_shift(np.array([pix]), bit_shift_covs[index]),
                                                 sparse_stub.dtype, wide_mask_maxbits,
                                                 nside_coverage_maps[index], nside_sparse,
                                                 primary=primary, sentinel=sentinel)

                valid_pixels = in_map.valid_pixels
                valid_pixels_cov = sparse_map._cov_map.cov_pixels(valid_pixels)
                ok, = np.where(valid_pixels_cov == pix)
                if ok.size == 0:
                    # There is no valid data here
                    continue
                valid_pixels = valid_pixels[ok]
            else:
                # nside_coverage_maps[index] > nside_coverage_out
                # Output coverage is coarser, which means we need to know
                # the full range of coverage pixels to read.

                _pixels = (np.left_shift(pix, bit_shift_covs[index]) +
                           np.arange(2**bit_shift_covs[index], dtype=np.int32))

                in_map = _read_partial_sparsemap(fitses[index], cov_map_nfine_per_covs[index],
                                                 wmult, cov_index_maps[index],
                                                 _pixels,
                                                 sparse_stub.dtype, wide_mask_maxbits,
                                                 nside_coverage_maps[index], nside_sparse,
                                                 primary=primary, sentinel=sentinel)
                valid_pixels = in_map.valid_pixels

            if check_overlap:
                if np.any(sparse_map[valid_pixels] != sparse_map._sentinel):
                    if not sparse_map.is_integer_map or not or_overlap:
                        outfits.close()
                        raise RuntimeError("Map %s has pixels that were already set in coverage pixel %d" %
                                           (file_list[index], pix))
                    else:
                        non_sentinel = sparse_map[valid_pixels] != sparse_map._sentinel
                        # We need to separate between filled and not because if we choose
                        # a non-zero sentinel, the or operation with the sentinel can give
                        # strange results
                        valid_filled = valid_pixels[non_sentinel]
                        valid_empty = valid_pixels[~non_sentinel]
                        sparse_map[valid_filled] = in_map[valid_filled] | sparse_map[valid_filled]
                        if len(valid_empty) > 0:
                            sparse_map[valid_empty] = in_map[valid_empty]
                else:
                    sparse_map[valid_pixels] = in_map[valid_pixels]
            else:
                sparse_map[valid_pixels] = in_map[valid_pixels]

        # And if we are spooling to disk, do that now.
        if not in_memory:
            # Grab out just this coverage pixel from the temporary sparse_map
            # data vector.
            if sparse_map.is_wide_mask_map:
                new_data = sparse_map._sparse_map[cov_map.nfine_per_cov:, :].ravel()
            else:
                new_data = sparse_map._sparse_map[cov_map.nfine_per_cov:]
            outfits.append_extension('SPARSE', new_data)

    # Close all the fits files
    for fits in fitses:
        fits.close()

    if not in_memory:
        # Close the output fits file
        outfits.close()

        # And rename the file
        if clobber and os.path.isfile(outfile):
            os.unlink(outfile)

        os.rename(outfile_temp, outfile)
    else:
        # Output the in memory map to file
        sparse_map.write(outfile, clobber=clobber)


def _read_partial_sparsemap(fits, nfine_per_cov, wmult, cov_index_map_temp,
                            pixels, dtype, wide_mask_maxbits, nside_coverage,
                            nside_sparse,
                            sparse_map_temp_input=None, primary=None,
                            sentinel=None):
    """
    Read part of a sparse map from an open fits file.

    Parameters
    ----------
    fits : `HealSparseFits`
    nfine_per_cov : `int`
    wmult : `int`
    cov_index_map_temp : `np.ndarray`
    pixels : `np.ndarray`
    dtype : `np.dtype`
    wide_mask_maxbits : `int` or `None`
    nside_coverage : `int`
    nside_sparse : `int`
    sparse_map_temp_input : `np.ndarray`, optional
    primary : `str`
    sentinel : `int` or `float`

    Returns
    -------
    sparse_map : `HealSparseMap`
    """
    if len(pixels) == 1:
        # Only with 1 pixel can we use the cached input.
        # We also only read maps that have coverage
        if sparse_map_temp_input is None:
            sparse_map_temp = np.zeros(2*nfine_per_cov*wmult,
                                       dtype=dtype)
            row_range = [0, nfine_per_cov*wmult]
            sparse_map_temp[0: nfine_per_cov*wmult] = \
                fits.read_ext_data('SPARSE', row_range=row_range)
        else:
            sparse_map_temp = sparse_map_temp_input

        row_range = [cov_index_map_temp[pixels[0]]*wmult,
                     (cov_index_map_temp[pixels[0]] + nfine_per_cov)*wmult]
        sparse_map_temp[nfine_per_cov*wmult:
                        2*nfine_per_cov*wmult] = fits.read_ext_data('SPARSE', row_range=row_range)
    else:
        # We need to read multiple pixels -- separate code path to
        # check these pixels and loop over them
        cov_pix_temp, = np.where(cov_index_map_temp >= nfine_per_cov)
        sub = np.clip(np.searchsorted(cov_pix_temp, pixels), 0, cov_pix_temp.size - 1)
        ok, = np.where(cov_pix_temp[sub] == pixels)
        sub = np.sort(sub[ok])

        sparse_map_temp = np.zeros((sub.size + 1)*nfine_per_cov*wmult,
                                   dtype=dtype)
        row_range = [0, nfine_per_cov*wmult]
        sparse_map_temp[0: nfine_per_cov*wmult] = \
            fits.read_ext_data('SPARSE', row_range=row_range)

        for i, sub_pix in enumerate(cov_pix_temp[sub]):
            row_range = [cov_index_map_temp[sub_pix]*wmult,
                         (cov_index_map_temp[sub_pix] + nfine_per_cov)*wmult]
            sparse_map_temp[(i + 1)*nfine_per_cov*wmult:
                            (i + 2)*nfine_per_cov*wmult] = fits.read_ext_data('SPARSE', row_range=row_range)

    if wide_mask_maxbits is not None:
        sparse_map_temp = sparse_map_temp.reshape((sparse_map_temp.size // wmult,
                                                   wmult)).astype(WIDE_MASK)

    cov_map_temp = HealSparseCoverage.make_from_pixels(nside_coverage,
                                                       nside_sparse,
                                                       pixels)
    partial_map = HealSparseMap(cov_map=cov_map_temp,
                                sparse_map=sparse_map_temp,
                                nside_sparse=nside_sparse,
                                primary=primary,
                                sentinel=sentinel)
    return partial_map
