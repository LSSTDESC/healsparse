from __future__ import division, absolute_import, print_function

import numpy as np
import healpy as hp

from .healSparseMap import HealSparseMap
from .healSparseCoverage import HealSparseCoverage


def sum_union(map_list):
    """
    Sum a list of HealSparseMaps as a union.  Empty values will be treated as
    0s in the summation, and the output map will have a union of all the input
    map pixels.

    Parameters
    ----------
    map_list : `list` of `HealSparseMap`
       Input list of maps to sum

    Returns
    -------
    result : `HealSparseMap`
       Summation of maps
    """

    return _apply_operation(map_list, np.add, 0, union=True)


def sum_intersection(map_list):
    """
    Sum a list of HealSparseMaps as an intersection.  Only pixels that are valid
    in all the input maps will have valid values in the output.

    Parameters
    ----------
    map_list : `list` of `HealSparseMap`
       Input list of maps to sum

    Returns
    -------
    result : `HealSparseMap`
       Summation of maps
    """

    return _apply_operation(map_list, np.add, 0, union=False)


def product_union(map_list):
    """
    Compute the product of a list of HealSparseMaps as a union.  Empty values
    will be treated as 1s in the product, and the output map will have a
    union of all the input map pixels.

    Parameters
    ----------
    map_list : `list` of `HealSparseMap`
       Input list of maps to take the product

    Returns
    -------
    result : `HealSparseMap`
       Product of maps
    """

    if map_list[0].is_integer_map:
        value = 1
    else:
        value = 1.0

    return _apply_operation(map_list, np.multiply, value, union=True)


def product_intersection(map_list):
    """
    Compute the product of a list of HealSparseMaps as an intersection.  Only
    pixels that are valid in all the input maps will have valid values in the
    output.

    Parameters
    ----------
    map_list : `list` of `HealSparseMap`
       Input list of maps to take the product

    Returns
    -------
    result : `HealSparseMap`
       Product of maps
    """

    if map_list[0].is_integer_map:
        value = 1
    else:
        value = 1.0

    return _apply_operation(map_list, np.multiply, value, union=False)


def or_union(map_list):
    """
    Bitwise or a list of HealSparseMaps as a union.  Empty values will be
    treated as 0s in the bitwise or, and the output map will have a union of all
    the input map pixels.  Only works in integer maps.

    Parameters
    ----------
    map_list : `list` of `HealSparseMap`
       Input list of maps to bitwise or

    Returns
    -------
    result : `HealSparseMap`
       Bitwise or of maps
    """

    return _apply_operation(map_list, np.bitwise_or, 0, union=True, int_only=True)


def or_intersection(map_list):
    """
    Bitwise or a list of HealSparseMaps as an intersection.  Only pixels that
    are valid in all the input maps will have valid values in the output.
    Only works on integer maps.

    Parameters
    ----------
    map_list : `list` of `HealSparseMap`
       Input list of maps to bitwise or

    Returns
    -------
    result : `HealSparseMap`
       Bitwise or of maps
    """

    return _apply_operation(map_list, np.bitwise_or, 0, union=False, int_only=True)


def and_union(map_list):
    """
    Bitwise and a list of HealSparseMaps as a union.  Empty values will be
    treated as 0s in the bitwise and, and the output map will have a union of all
    the input map pixels.  Only works in integer maps.

    Parameters
    ----------
    map_list : `list` of `HealSparseMap`
       Input list of maps to bitwise and

    Returns
    -------
    result : `HealSparseMap`
       Bitwise and of maps
    """
    filler = map_list[0]._sparse_map.dtype.type(-1)

    return _apply_operation(map_list, np.bitwise_and, filler, union=True, int_only=True)


def and_intersection(map_list):
    """
    Bitwise or a list of HealSparseMaps as an intersection.  Only pixels that
    are valid in all the input maps will have valid values in the output.
    Only works on integer maps.

    Parameters
    ----------
    map_list : `list` of `HealSparseMap`
       Input list of maps to bitwise and

    Returns
    -------
    result : `HealSparseMap`
       Bitwise and of maps
    """
    filler = map_list[0]._sparse_map.dtype.type(-1)

    return _apply_operation(map_list, np.bitwise_and, filler, union=False, int_only=True)


def xor_union(map_list):
    """
    Bitwise xor a list of HealSparseMaps as a union.  Empty values will be
    treated as 0s in the bitwise or, and the output map will have a union of all
    the input map pixels.  Only works in integer maps.

    Parameters
    ----------
    map_list : `list` of `HealSparseMap`
       Input list of maps to bitwise xor

    Returns
    -------
    result : `HealSparseMap`
       Bitwise xor of maps
    """

    return _apply_operation(map_list, np.bitwise_xor, 0, union=True, int_only=True)


def xor_intersection(map_list):
    """
    Bitwise xor a list of HealSparseMaps as an intersection.  Only pixels that
    are valid in all the input maps will have valid values in the output.
    Only works on integer maps.

    Parameters
    ----------
    map_list : `list` of `HealSparseMap`
       Input list of maps to bitwise xor

    Returns
    -------
    result : `HealSparseMap`
       Bitwise xor of maps
    """

    return _apply_operation(map_list, np.bitwise_xor, 0, union=False, int_only=True)


def max_intersection(map_list):
    """
    Element-wise maximum of the intersection of a list of the HealSparseMaps.

    Parameters
    ----------
    map_list : `list` of `HealSparseMap`
        Input list of maps to compute the maximum of

    Returns
    -------
    result : `HealSparseMap`
        Element-wise maximum of maps
    """

    return _apply_operation(map_list, np.fmax, 0, union=False, int_only=False)


def min_intersection(map_list):
    """
    Element-wise minimum of the intersection of a list of HealSparseMaps.

    Parameters
    ----------
    map_list : `list` of `HealSparseMap`
        Input list of maps to compute the minimum of

    Returns
    -------
    result : `HealSparseMap`
        Element-wise minimum of maps
    """

    return _apply_operation(map_list, np.fmin, -hp.UNSEEN, union=False, int_only=False)


def max_union(map_list):
    """
    Element-wise maximum of the union of a list of HealSparseMaps.

    Parameters
    ----------
    map_list : `list` of `HealSparseMap`
        Input list of maps to compute the maximum of

    Returns
    -------
    result : `HealSparseMap`
        Element-wise maximum of maps
    """

    return _apply_operation(map_list, np.fmax, 0, union=True, int_only=False)


def min_union(map_list):
    """
    Element-wise minimum of the union of a list of HealSparseMaps.

    Parameters
    ----------
    map_list : `list` of `HealSparseMap`
        Input list of maps to compute the minimum of

    Returns
    -------
    result : `HealSparseMap`
        Element-wise minimum of maps
    """

    return _apply_operation(map_list, np.fmin, -hp.UNSEEN, union=True, int_only=False)


def ufunc_intersection(map_list, func, filler_value=0):
    """
    Apply numpy ufunc to the intersection of a list of HealSparseMaps.

    Parameters
    ----------
    map_list : `list` of `HealSparseMap`
        Input list of maps where the operation is applied
    func : `np.ufunc`
        Numpy universal function to apply
    filler_value : `int` or `float`
        Starting value
    Returns
    -------
    result : `HealSparseMap`
        Resulting map
    """

    return _apply_operation(map_list, func, filler_value, union=False, int_only=False)


def ufunc_union(map_list, func, filler_value=0):
    """
    Apply numpy ufunc to the union of a list of HealSparseMaps.

    Parameters
    ----------
    map_list : `list` of `HealSparseMaps`
        Input list of maps where the operation is applied
    func : `np.ufunc`
        Numpy universal function to apply
    filler_value : `int` or `float`
        Starting value and filler for the union

    Returns
    -------
    result : `HealSparseMap`
        Resulting map
    """

    return _apply_operation(map_list, func, filler_value, union=True, int_only=False)


def _apply_operation(map_list, func, filler_value, union=False, int_only=False):
    """
    Apply a generic arithmetic function.

    Cannot be used with recarray maps

    Parameters
    ----------
    map_list : `list` of `HealSparseMap`
       Input list of maps to perform the operation on.
    func : `np.ufunc`
       Numpy universal function to apply
    filler_value : `int` or `float`
       Starting value and filler when union is True
    union : `bool`, optional
       Use union mode instead of intersection.  Default is False.
    int_only : `bool`, optional
       Check that input maps are integer types.  Default is False.

    Returns
    -------
    result : `HealSparseMap`
       Resulting map
    """

    name = func.__str__()

    if len(map_list) < 2:
        raise RuntimeError("Must supply at least 2 maps to apply %s" % (name))

    nside_coverage = None
    for m in map_list:
        if not isinstance(m, HealSparseMap):
            raise NotImplementedError("Can only apply %s to HealSparseMaps" % (name))
        if m.is_rec_array:
            raise NotImplementedError("Cannot apply %s to recarray maps" % (name))
        if int_only:
            if not m.is_integer_map:
                raise ValueError("Can only apply %s to integer maps" % (name))

        if nside_coverage is None:
            nside_coverage = m.nside_coverage
            nside_sparse = m._nside_sparse
            dtype = m._sparse_map.dtype
            sentinel = m._sentinel
            is_wide_mask = m._is_wide_mask
            wide_mask_width = m._wide_mask_width
        else:
            if (nside_coverage != m.nside_coverage or nside_sparse != m._nside_sparse):
                raise RuntimeError("Cannot apply %s to maps with different coverage or map nsides" % (name))
            if (is_wide_mask != m._is_wide_mask or wide_mask_width != m._wide_mask_width):
                raise RuntimeError("Can only apply %s to wide_mask maps with same width" % (name))

    combined_cov_mask = map_list[0].coverage_mask

    if union:
        # Union mode
        for m in map_list[1:]:
            combined_cov_mask |= m.coverage_mask
    else:
        # Intersection mode
        for m in map_list[1:]:
            combined_cov_mask &= m.coverage_mask

    cov_pix, = np.where(combined_cov_mask)

    if cov_pix.size == 0:
        # No coverage ... the result is an empty map
        return HealSparseMap.make_empty_like(map_list[0])

    # Initialize the combined map, we know the size
    cov_map = HealSparseCoverage.make_from_pixels(nside_coverage,
                                                  nside_sparse,
                                                  cov_pix)
    if is_wide_mask:
        combined_sparse_map = np.zeros(((cov_pix.size + 1)*cov_map.nfine_per_cov,
                                        wide_mask_width), dtype=dtype) + filler_value
    else:
        combined_sparse_map = np.zeros((cov_pix.size + 1)*cov_map.nfine_per_cov, dtype=dtype) + filler_value

    if union:
        combined_sparse_map_touched = np.zeros(len(combined_sparse_map), dtype=np.bool)
    else:
        combined_sparse_map_ntouch = np.zeros(len(combined_sparse_map), dtype=np.int32)
    for m in map_list:
        m_valid_pixels = m.valid_pixels

        ipnest_cov = cov_map.cov_pixels(m_valid_pixels)

        combined_pixel_index = m_valid_pixels + cov_map[ipnest_cov]

        if is_wide_mask:
            values = m.get_values_pix(m_valid_pixels)
            for i in range(wide_mask_width):
                combined_sparse_map[combined_pixel_index, i] = func(
                    combined_sparse_map[combined_pixel_index, i],
                    values[:, i])
        else:
            combined_sparse_map[combined_pixel_index] = func(
                combined_sparse_map[combined_pixel_index],
                m.get_values_pix(m_valid_pixels))
        if union:
            combined_sparse_map_touched[combined_pixel_index] = True
        else:
            combined_sparse_map_ntouch[combined_pixel_index] += 1

    # And when we're done, those that are untouched should be set to sentinel
    if union:
        # In union, replace untouched fillers with sentinel
        if filler_value != sentinel:
            combined_sparse_map[~combined_sparse_map_touched] = sentinel
    else:
        # In intersection, all pixels that weren't touched all the time
        # should be set to the sentinel
        combined_sparse_map[combined_sparse_map_ntouch != len(map_list)] = sentinel

    # And set the overflow bins to the sentinel
    combined_sparse_map[0: cov_map.nfine_per_cov] = sentinel

    return HealSparseMap(cov_map=cov_map, sparse_map=combined_sparse_map,
                         nside_sparse=nside_sparse, sentinel=sentinel)
