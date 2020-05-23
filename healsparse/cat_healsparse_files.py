import numpy as np
import healpy as hp
import os

from .healSparseMap import HealSparseMap
from .healSparseCoverage import HealSparseCoverage
from .fits_shim import HealSparseFits
from .utils import _compute_bitshift, WIDE_NBIT


def cat_healsparse_files(file_list, outfile, check_overlap=False, clobber=False,
                         in_memory=False, nside_coverage_out=None):
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
    """
    if os.path.isfile(outfile) and not clobber:
        raise RuntimeError("File %s already exists and clobber is False" % (outfile))

    # Read in all the coverage maps
    cov_mask_summary = None
    nside_sparse = None
    nside_coverage_maps = []
    bit_shift_covs = []
    for i, f in enumerate(file_list):
        cov_map = HealSparseCoverage.read(f)

        if cov_mask_summary is None:
            if nside_coverage_out is None:
                nside_coverage_out = cov_map.nside_coverage

            cov_mask_summary = np.zeros((len(f), hp.nside2npix(nside_coverage_out)),
                                        dtype=np.bool)
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
            sentinel = hp.UNSEEN

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

    if not in_memory:
        # When spooling to disk, we need a stub to write (with the final cov_map)
        # to append to.

        stub_map = HealSparseMap(cov_map=cov_map,
                                 sparse_map=sparse_stub, nside_sparse=nside_sparse,
                                 primary=primary, sentinel=sentinel)

        # And write this out to a temporary filename
        outfile_temp = outfile + '.incomplete'
        stub_map.write(outfile_temp, clobber=True)

        try:
            outfits = HealSparseFits(outfile_temp, mode='rw')
        except RuntimeError:
            raise RuntimeError("Running cat_healsparse_files with in_memory=False requires fitsio.")
    else:
        # When building in memory, we just need a blank map
        sparse_map = HealSparseMap.make_empty(nside_coverage_out, nside_sparse,
                                              sparse_stub.dtype, primary=primary,
                                              sentinel=sentinel, wide_mask_maxbits=wide_mask_maxbits)

    total_data = 0
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
                # Straightforward
                in_map = HealSparseMap.read(file_list[index], pixels=[pix])
                valid_pixels = in_map.valid_pixels
            elif nside_coverage_maps[index] < nside_coverage_out:
                # nside_coverage_maps[index] < nside_coverage_out
                # Output coverage is finer, which means we just need to know
                # the one coarse pix to read in here.

                pixels = np.right_shift(pix, bit_shift_covs[index])
                in_map = HealSparseMap.read(file_list[index], pixels=pixels)
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

                pixels = (np.left_shift(pix, bit_shift_covs[index]) +
                          np.arange(2**bit_shift_covs[index], dtype=np.int32))
                in_map = HealSparseMap.read(file_list[index], pixels=pixels)
                valid_pixels = in_map.valid_pixels

            if check_overlap:
                if np.any(sparse_map[valid_pixels] > sparse_map._sentinel):
                    outfits.close()
                    raise RuntimeError("Map %s has pixels that were already set in coverage pixel %d" %
                                       (file_list[index], pix))

            sparse_map[valid_pixels] = in_map[valid_pixels]

        # And if we are spooling to disk, do that now.
        if not in_memory:
            # Grab out just this coverage pixel from the temporary sparse_map
            # data vector.
            if sparse_map.is_wide_mask_map:
                new_data = sparse_map._sparse_map[cov_map.nfine_per_cov:, :].flatten()
            else:
                new_data = sparse_map._sparse_map[cov_map.nfine_per_cov:]
            outfits.append_extension('SPARSE', new_data)
            total_data += new_data.size

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
