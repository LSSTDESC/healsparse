import numpy as np
import os
import warnings

from .healSparseMap import HealSparseMap


def cat_healsparse_files(file_list, outfile, check_overlap=False, clobber=False,
                         in_memory=True, nside_coverage_out=None, or_overlap=False):
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
        Spool-to-disk is no longer supported.
    nside_coverage_out : `int`, optional
        Output map with specific nside_coverage.  Default is nside_coverage
        of first map in file_list.
    or_overlap: `bool`, optional
        If True compute the `or` overlap of two integer maps when concatenating.
    """
    if os.path.isfile(outfile) and not clobber:
        raise RuntimeError("File %s already exists and clobber is False" % (outfile))

    if not in_memory:
        warnings.warn("cat_healsparse_files always does in-memory operations.")

    if or_overlap and not check_overlap:
        check_overlap = True
        warnings.warn("or_overlap is True and check_overlap is False; will check overlap.")

    sparse_map = None

    for f in file_list:
        in_map = HealSparseMap.read(f)

        if sparse_map is None:
            sparse_map = HealSparseMap.make_empty_like(in_map, nside_coverage=nside_coverage_out)

        for valid_pixels in in_map.iter_valid_pixels_by_covpix():
            if check_overlap:
                if np.any(sparse_map[valid_pixels] != sparse_map.sentinel):
                    if not sparse_map.is_integer_map or not or_overlap:
                        raise RuntimeError(f"Map {f} has pixels that were already set.")
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

    # Defragment the map.
    if sparse_map.is_fragmented:
        sparse_map.defragment(in_place=True)

    # Output the in memory map to file
    sparse_map.write(outfile, clobber=clobber)
