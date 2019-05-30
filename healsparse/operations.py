from __future__ import division, absolute_import, print_function

import numpy as np
import healpy as hp

from .utils import reduce_array, checkSentinel
from .healSparseMap import HealSparseMap

def sumUnion(mapList):
    """
    Sum a list of HealSparseMaps as a union.  Empty values will be treated as
    0s in the summation, and the output map will have a union of all the input
    map pixels.

    Parameters
    ----------
    mapList: `list` of `HealSparseMap`
       Input list of maps to sum

    Returns
    -------
    result: `HealSparseMap`
       Summation of maps
    """

    return _applyOperation(mapList, np.add, 0, union=True)

def sumIntersection(mapList):
    """
    Sum a list of HealSparseMaps as an intersection.  Only pixels that are valid
    in all the input maps will have valid values in the output.

    Parameters
    ----------
    mapList: `list` of `HealSparseMap`
       Input list of maps to sum

    Returns
    -------
    result: `HealSparseMap`
       Summation of maps
    """

    return _applyOperation(mapList, np.add, 0, union=False)

def productUnion(mapList):
    """
    Compute the product of a list of HealSparseMaps as a union.  Empty values
    will be treated as 1s in the product, and the output map will have a
    union of all the input map pixels.

    Parameters
    ----------
    mapList: `list` of `HealSparseMap`
       Input list of maps to take the product

    Returns
    -------
    result: `HealSparseMap`
       Product of maps
    """

    return _applyOperation(mapList, np.multiply, 1.0, union=True)

def productIntersection(mapList):
    """
    Compute the product of a list of HealSparseMaps as an intersection.  Only
    pixels that are valid in all the input maps will have valid values in the
    output.

    Parameters
    ----------
    mapList: `list` of `HealSparseMap`
       Input list of maps to take the product

    Returns
    -------
    result: `HealSparseMap`
       Product of maps
    """

    return _applyOperation(mapList, np.multiply, 1.0, union=False)

def orUnion(mapList):
    """
    Bitwise or a list of HealSparseMaps as a union.  Empty values will be
    treated as 0s in the bitwise or, and the output map will have a union of all
    the input map pixels.  Only works in integer maps.

    Parameters
    ----------
    mapList: `list` of `HealSparseMap`
       Input list of maps to bitwise or

    Returns
    -------
    result: `HealSparseMap`
       Bitwise or of maps
    """

    return _applyOperation(mapList, np.bitwise_or, 0, union=True, intOnly=True)

def orIntersection(mapList):
    """
    Bitwise or a list of HealSparseMaps as an intersection.  Only pixels that
    are valid in all the input maps will have valid values in the output.
    Only works on integer maps.

    Parameters
    ----------
    mapList: `list` of `HealSparseMap`
       Input list of maps to bitwise or

    Returns
    -------
    result: `HealSparseMap`
       Bitwise or of maps
    """

    return _applyOperation(mapList, np.bitwise_or, 0, union=False, intOnly=True)

def andUnion(mapList):
    """
    Bitwise and a list of HealSparseMaps as a union.  Empty values will be
    treated as 0s in the bitwise and, and the output map will have a union of all
    the input map pixels.  Only works in integer maps.

    Parameters
    ----------
    mapList: `list` of `HealSparseMap`
       Input list of maps to bitwise and

    Returns
    -------
    result: `HealSparseMap`
       Bitwise and of maps
    """

    return _applyOperation(mapList, np.bitwise_and, -1, union=True, intOnly=True)

def andIntersection(mapList):
    """
    Bitwise or a list of HealSparseMaps as an intersection.  Only pixels that
    are valid in all the input maps will have valid values in the output.
    Only works on integer maps.

    Parameters
    ----------
    mapList: `list` of `HealSparseMap`
       Input list of maps to bitwise and

    Returns
    -------
    result: `HealSparseMap`
       Bitwise and of maps
    """

    return _applyOperation(mapList, np.bitwise_and, -1, union=False, intOnly=True)

def xorUnion(mapList):
    """
    Bitwise xor a list of HealSparseMaps as a union.  Empty values will be
    treated as 0s in the bitwise or, and the output map will have a union of all
    the input map pixels.  Only works in integer maps.

    Parameters
    ----------
    mapList: `list` of `HealSparseMap`
       Input list of maps to bitwise xor

    Returns
    -------
    result: `HealSparseMap`
       Bitwise xor of maps
    """

    return _applyOperation(mapList, np.bitwise_xor, 0, union=True, intOnly=True)

def xorIntersection(mapList):
    """
    Bitwise xor a list of HealSparseMaps as an intersection.  Only pixels that
    are valid in all the input maps will have valid values in the output.
    Only works on integer maps.

    Parameters
    ----------
    mapList: `list` of `HealSparseMap`
       Input list of maps to bitwise xor

    Returns
    -------
    result: `HealSparseMap`
       Bitwise xor of maps
    """

    return _applyOperation(mapList, np.bitwise_xor, 0, union=False, intOnly=True)

def _applyOperation(mapList, func, fillerValue, union=False, intOnly=False):
    """
    Apply a generic arithmetic function.

    Cannot be used with recarray maps

    Parameters
    ----------
    mapList: `list` of `HealSparseMap`
       Input list of maps to perform the operation on.
    func: `np.ufunc`
       Numpy universal function to apply
    fillerValue: `int` or `float`
       Starting value and filler when union is True
    union: `bool`, optional
       Use union mode instead of intersection.  Default is False.
    intOnly: `bool`, optional
       Check that input maps are integer types.  Default is False.

    Returns
    -------
    result: `HealSparseMap`
       Resulting map
    """

    name = func.__str__()

    if len(mapList) < 2:
        raise RuntimeError("Must supply at least 2 maps to apply %s" % (name))

    nsideCoverage = None
    for m in mapList:
        if not isinstance(m, HealSparseMap):
            raise NotImplementedError("Can only apply %s to HealSparseMaps" % (name))
        if m._isRecArray:
            raise NotImplementedError("Cannot apply %s to recarray maps" % (name))
        if intOnly:
            if not issubclass(m._sparseMap.dtype.type, np.integer):
                raise ValueError("Can only apply %s to integer maps" % (name))

        if nsideCoverage is None:
            nsideCoverage = m._nsideCoverage
            nsideSparse = m._nsideSparse
            bitShift = m._bitShift
            dtype = m._sparseMap.dtype
            sentinel = m._sentinel
        else:
            if (nsideCoverage != m._nsideCoverage or
                nsideSparse != m._nsideSparse):
                raise RuntimeError("Cannot apply %s to maps with different coverage or map nsides" % (name))

    combinedCovMask = mapList[0].coverageMask

    if union:
        # Union mode
        for m in mapList[1: ]:
            combinedCovMask |= m.coverageMask
    else:
        # Intersection mode
        for m in mapList[1: ]:
            combinedCovMask &= m.coverageMask

    covPix, = np.where(combinedCovMask)

    if covPix.size == 0:
        # No coverage ... the result is an empty map
        return HealSparseMap.makeEmpty(nsideCoverage, nsideSparse, dtype)

    # Initialize the combined map, we know the size
    nFinePerCov = 2**bitShift
    covIndexMap = np.zeros(hp.nside2npix(nsideCoverage), dtype=np.int64)
    covIndexMap[covPix] = np.arange(1, covPix.size + 1) * nFinePerCov
    covIndexMap[:] -= np.arange(hp.nside2npix(nsideCoverage), dtype=np.int64) * nFinePerCov
    combinedSparseMap = np.zeros((covPix.size + 1) * nFinePerCov, dtype=dtype) + sentinel

    # Determine which individual pixels might be in the combined map
    allPixels = None
    for m in mapList:
        pixels = np.arange(m._sparseMap.size)
        covIndex = np.right_shift(pixels, bitShift)
        pixels -= m._covIndexMap[m.coverageMask][covIndex - 1]

        if allPixels is None:
            allPixels = pixels.copy()
        else:
            allPixels = np.concatenate((allPixels, pixels))

    allPixels = np.unique(allPixels)
    covIndex = np.right_shift(allPixels, bitShift)

    m = mapList[0]
    gd = (m._sparseMap[allPixels + m._covIndexMap[covIndex]] > m._sentinel)
    for m in mapList[1: ]:
        if union:
            # Union mode
            gd |= (m._sparseMap[allPixels + m._covIndexMap[covIndex]] > m._sentinel)
        else:
            # Intersection mode
            gd &= (m._sparseMap[allPixels + m._covIndexMap[covIndex]] > m._sentinel)

    allPixels = allPixels[gd]
    covIndex = covIndex[gd]

    combinedSparseMap[allPixels + covIndexMap[covIndex]] = fillerValue

    for m in mapList:
        if union:
            # Union mode requires tracking bad pixels and replacing them
            # Note that this is redundant with above.  I don't know if it's better
            # to cache and take a possible memory hit, or recompute here.
            gd = (m._sparseMap[allPixels + m._covIndexMap[covIndex]] > m._sentinel)
            combinedSparseMap[allPixels[gd] + covIndexMap[covIndex[gd]]] = func(combinedSparseMap[allPixels[gd] + covIndexMap[covIndex[gd]]],
                                                                                m._sparseMap[allPixels[gd] + m._covIndexMap[covIndex[gd]]])
        else:
            # Intersection mode we only have good pixels
            combinedSparseMap[allPixels + covIndexMap[covIndex]] = func(combinedSparseMap[allPixels + covIndexMap[covIndex]],
                                                                        m._sparseMap[allPixels + m._covIndexMap[covIndex]])

    return HealSparseMap(covIndexMap=covIndexMap, sparseMap=combinedSparseMap, nsideSparse=nsideSparse, sentinel=sentinel)

