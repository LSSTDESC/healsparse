import numpy as np
import hpgeom as hpg
import os
import warnings

from .healSparseMap import HealSparseMap
from .healSparseCoverage import HealSparseCoverage
from .fits_shim import HealSparseFits, use_rustfits
from .utils import _compute_bitshift


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
        Do operations in-memory (required unless rustfits is available).
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
        warnings.warn("or_overlap is True and check_overlap is False; will check overlap.")

    if not in_memory and not use_rustfits:
        raise RuntimeError("Spooling to disk (in_memory=False) requires rustfits.")

    # Get the combined coverage map and mapping from file to coverage pixels.
    cov_map, nside_coverages, cov_index_maps, cov_mask_summary, cov_bit_shifts = _combine_coverage_maps(
        file_list,
        nside_coverage_out,
    )
    cov_pixels, = np.nonzero(cov_map.coverage_mask)

    # Read in a pixel from the first map.
    map_temp = HealSparseMap.read(file_list[0], pixels=np.where(cov_index_maps[0] > 0)[0][0: 1])

    # Maybe this will work!
    # if map_temp.is_rec_array and not in_memory:
    #     raise RuntimeError("Spooling to disk (in_memory=False) is not supported with a recarray map.")

    if in_memory:
        # Create the empty map to fill.
        sparse_map = HealSparseMap.make_empty_like(
            map_temp,
            nside_coverage=cov_map.nside_coverage,
            cov_pixels=cov_pixels,
        )
    else:
        # Make an empty map.
        outfile_temp = outfile + ".incomplete"

        sparse_map_stub = HealSparseMap.make_empty_like(
            map_temp,
            nside_coverage=cov_map.nside_coverage,
        )

        # Hack the coverage map (do not try this at home).
        sparse_map_stub._cov_map = cov_map

        # Write out the stub (which includes the overflow data).
        sparse_map_stub.write(outfile_temp, clobber=True)

        # Open up a streaming fits object to append to.
        fits_stream = HealSparseFits(outfile_temp, mode="rw")

    # Work one coverage pixel at a time.
    for cov_pix in cov_pixels:
        # Which input files overlap this coverage pixel?
        u_cov_pix, = np.nonzero(cov_mask_summary[:, cov_pix])

        if not in_memory:
            # We need a holder for the data to stream.
            sparse_map = HealSparseMap.make_empty_like(
                sparse_map_stub,
                cov_pixels=[cov_pix],
            )

        for index in u_cov_pix:
            if nside_coverages[index] == nside_coverage_out:
                # Straightforward: matched coverage.
                in_map = HealSparseMap.read(file_list[index], pixels=[cov_pix])

                valid_pixels = in_map.valid_pixels
            elif nside_coverages[index] < nside_coverage_out:
                # Output coverage is finer, which means we just need to read
                # the one coarse pixel.
                in_map = HealSparseMap.read(
                    file_list[index],
                    pixels=np.right_shift([cov_pix], cov_bit_shifts[index]),
                )
                valid_pixels = in_map.valid_pixels
                valid_pixels_cov = cov_map.cov_pixels(valid_pixels)
                ok = (valid_pixels_cov == cov_pix)
                if ok.sum() == 0:
                    # No valid data here.
                    continue
                valid_pixels = valid_pixels[ok]
            else:
                # Output coverage is coarser, which means we need to know
                # the full range of coverage pixels to read.
                in_map = HealSparseMap.read(
                    file_list[index],
                    pixels=(
                        np.left_shift(cov_pix, cov_bit_shifts[index]) +
                        np.arange(2**cov_bit_shifts[index], dtype=np.int32)
                    ),
                )
                valid_pixels = in_map.valid_pixels

            if check_overlap:
                if np.any(sparse_map[valid_pixels] != sparse_map.sentinel):
                    if not sparse_map.is_integer_map or not or_overlap:
                        raise RuntimeError(f"Map {file_list[index]} has pixels that were already set.")
                    else:
                        non_sentinel = sparse_map[valid_pixels] != sparse_map.sentinel
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

        if not in_memory:
            # Stream coverage pixel data to disk.
            if sparse_map.is_wide_mask_map:
                fits_stream.append_extension(
                    "SPARSE",
                    sparse_map._sparse_map[cov_map.nfine_per_cov:, :].ravel(),
                )
            elif sparse_map.is_bit_packed_map:
                fits_stream.append_extension(
                    "SPARSE",
                    sparse_map._sparse_map.data_array[cov_map.nfine_per_cov // 8:],
                )
            else:
                fits_stream.append_extension("SPARSE", sparse_map._sparse_map[cov_map.nfine_per_cov:])

    if in_memory:
        sparse_map.write(outfile, clobber=clobber)
    else:
        # Close the output fits file.
        fits_stream.close()

        # Rename the file
        if clobber and os.path.isfile(outfile):
            os.unlink(outfile)

        os.rename(outfile_temp, outfile)


def _combine_coverage_maps(file_list, nside_coverage_out):
    """Combine coverage maps.

    Parameters
    ----------
    file_list : `list` [`str`]
    nside_coverage_out : `int`

    Returns
    -------
    cov_map : `healsparse.HealSparseCoverage`
        The combined coverage map.
    nside_coverage_maps : `list` [`int`]
        List of input coverage map nsides.
    cov_index_maps : `list` [`np.ndarray`]
        List of input boolean coverage masks.
    cov_mask_summary : `np.ndarray`
        Summary of input coverage masks, converted to output nside_coverage.
    bit_shift_covs : `list` [`int`]
        List of bit-shift values to convert to output coverage nside.
    """
    cov_mask_summary = None
    nside_sparse = None
    nside_coverage_maps = []
    bit_shift_covs = []
    cov_index_maps = []
    cov_map_nfine_per_covs = []

    for i, f in enumerate(file_list):
        cov_map = HealSparseCoverage.read(f)

        cov_index_map = cov_map[:] + np.arange(
            hpg.nside_to_npixel(cov_map.nside_coverage),
            dtype=np.int64,
        )*cov_map.nfine_per_cov
        cov_index_maps.append(cov_index_map)
        cov_map_nfine_per_covs.append(cov_map.nfine_per_cov)

        if cov_mask_summary is None:
            if nside_coverage_out is None:
                nside_coverage_out = cov_map.nside_coverage

            cov_mask_summary = np.zeros(
                (len(file_list), hpg.nside_to_npixel(nside_coverage_out)),
                dtype=np.bool_,
            )
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

    return cov_map, nside_coverage_maps, cov_index_maps, cov_mask_summary, bit_shift_covs
