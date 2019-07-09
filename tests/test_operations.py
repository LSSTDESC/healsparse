from __future__ import division, absolute_import, print_function

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
from numpy import random

import healsparse

class OperationsTestCase(unittest.TestCase):
    def test_sum(self):
        """
        Test map addition.
        """

        random.seed(seed=12345)

        nsideCoverage = 32
        nsideMap = 64

        # Test adding two or three maps

        sparseMap1 = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, np.float64)
        pixel1 = np.arange(4000, 20000)
        pixel1 = np.delete(pixel1, 15000)
        values1 = np.random.random(size=pixel1.size)
        sparseMap1.updateValues(pixel1, values1)
        hpmap1 = sparseMap1.generateHealpixMap()

        sparseMap2 = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, np.float64)
        pixel2 = np.arange(15000, 25000)
        values2 = np.random.random(size=pixel2.size)
        sparseMap2.updateValues(pixel2, values2)
        hpmap2 = sparseMap2.generateHealpixMap()

        sparseMap3 = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, np.float64)
        pixel3 = np.arange(16000, 25000)
        values3 = np.random.random(size=pixel3.size)
        sparseMap3.updateValues(pixel3, values3)
        hpmap3 = sparseMap3.generateHealpixMap()

        # Intersection addition

        # sum 2
        addedMapIntersection = healsparse.sumIntersection([sparseMap1, sparseMap2])

        gd, = np.where((hpmap1 > hp.UNSEEN) & (hpmap2 > hp.UNSEEN))
        hpmapSumIntersection = np.zeros_like(hpmap1) + hp.UNSEEN
        hpmapSumIntersection[gd] = hpmap1[gd] + hpmap2[gd]

        testing.assert_almost_equal(hpmapSumIntersection, addedMapIntersection.generateHealpixMap())

        # sum 3
        addedMapIntersection = healsparse.sumIntersection([sparseMap1, sparseMap2, sparseMap3])

        gd, = np.where((hpmap1 > hp.UNSEEN) & (hpmap2 > hp.UNSEEN) & (hpmap3 > hp.UNSEEN))
        hpmapSumIntersection = np.zeros_like(hpmap1) + hp.UNSEEN
        hpmapSumIntersection[gd] = hpmap1[gd] + hpmap2[gd] + hpmap3[gd]

        testing.assert_almost_equal(hpmapSumIntersection, addedMapIntersection.generateHealpixMap())

        # Union addition

        # sum 2
        addedMapUnion = healsparse.sumUnion([sparseMap1, sparseMap2])

        gd, = np.where((hpmap1 > hp.UNSEEN) | (hpmap2 > hp.UNSEEN))
        hpmapSumUnion = np.zeros_like(hpmap1) + hp.UNSEEN
        # This hack works because we don't have summands going below zero...
        hpmapSumUnion[gd] = np.clip(hpmap1[gd], 0.0, None) + np.clip(hpmap2[gd], 0.0, None)

        testing.assert_almost_equal(hpmapSumUnion, addedMapUnion.generateHealpixMap())

        # sum 3
        addedMapUnion = healsparse.sumUnion([sparseMap1, sparseMap2, sparseMap3])

        gd, = np.where((hpmap1 > hp.UNSEEN) | (hpmap2 > hp.UNSEEN) | (hpmap3 > hp.UNSEEN))
        hpmapSumUnion = np.zeros_like(hpmap1) + hp.UNSEEN
        # This hack works because we don't have summands going below zero...
        hpmapSumUnion[gd] = np.clip(hpmap1[gd], 0.0, None) + np.clip(hpmap2[gd], 0.0, None) + np.clip(hpmap3[gd], 0.0, None)

        testing.assert_almost_equal(hpmapSumUnion, addedMapUnion.generateHealpixMap())

        # Test adding an int constant to a map

        addedMap = sparseMap1 + 2

        hpmapAdd2 = np.zeros_like(hpmap1) + hp.UNSEEN
        gd, = np.where(hpmap1 > hp.UNSEEN)
        hpmapAdd2[gd] = hpmap1[gd] + 2

        testing.assert_almost_equal(hpmapAdd2, addedMap.generateHealpixMap())

        # Test adding a float constant to a map

        addedMap = sparseMap1 + 2.0

        hpmapAdd2 = np.zeros_like(hpmap1) + hp.UNSEEN
        gd, = np.where(hpmap1 > hp.UNSEEN)
        hpmapAdd2[gd] = hpmap1[gd] + 2.0

        testing.assert_almost_equal(hpmapAdd2, addedMap.generateHealpixMap())

        # Test adding a float constant to a map, in place

        sparseMap1 += 2.0

        testing.assert_almost_equal(hpmapAdd2, sparseMap1.generateHealpixMap())

    def test_product(self):
        """
        Test map products.
        """

        random.seed(seed=12345)

        nsideCoverage = 32
        nsideMap = 64

        # Test adding two or three maps

        sparseMap1 = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, np.float64)
        pixel1 = np.arange(4000, 20000)
        pixel1 = np.delete(pixel1, 15000)
        values1 = np.random.random(size=pixel1.size)
        sparseMap1.updateValues(pixel1, values1)
        hpmap1 = sparseMap1.generateHealpixMap()

        sparseMap2 = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, np.float64)
        pixel2 = np.arange(15000, 25000)
        values2 = np.random.random(size=pixel2.size)
        sparseMap2.updateValues(pixel2, values2)
        hpmap2 = sparseMap2.generateHealpixMap()

        sparseMap3 = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, np.float64)
        pixel3 = np.arange(16000, 25000)
        values3 = np.random.random(size=pixel3.size)
        sparseMap3.updateValues(pixel3, values3)
        hpmap3 = sparseMap3.generateHealpixMap()

        # Intersection product

        # product of 2
        productMapIntersection = healsparse.productIntersection([sparseMap1, sparseMap2])

        gd, = np.where((hpmap1 > hp.UNSEEN) & (hpmap2 > hp.UNSEEN))
        hpmapProductIntersection = np.zeros_like(hpmap1) + hp.UNSEEN
        hpmapProductIntersection[gd] = hpmap1[gd] * hpmap2[gd]

        testing.assert_almost_equal(hpmapProductIntersection, productMapIntersection.generateHealpixMap())

        # product of 3
        productMapIntersection = healsparse.productIntersection([sparseMap1, sparseMap2, sparseMap3])

        gd, = np.where((hpmap1 > hp.UNSEEN) & (hpmap2 > hp.UNSEEN) & (hpmap3 > hp.UNSEEN))
        hpmapProductIntersection = np.zeros_like(hpmap1) + hp.UNSEEN
        hpmapProductIntersection[gd] = hpmap1[gd] * hpmap2[gd] * hpmap3[gd]

        testing.assert_almost_equal(hpmapProductIntersection, productMapIntersection.generateHealpixMap())

        # Union product

        # product of 2
        productMapUnion = healsparse.productUnion([sparseMap1, sparseMap2])

        gd, = np.where((hpmap1 > hp.UNSEEN) | (hpmap2 > hp.UNSEEN))
        hpmapProductUnion = np.zeros_like(hpmap1) + hp.UNSEEN

        hpmapProductUnion[gd] = 1.0
        gd1, = np.where(hpmap1[gd] > hp.UNSEEN)
        hpmapProductUnion[gd[gd1]] *= hpmap1[gd[gd1]]
        gd2, = np.where(hpmap2[gd] > hp.UNSEEN)
        hpmapProductUnion[gd[gd2]] *= hpmap2[gd[gd2]]

        testing.assert_almost_equal(hpmapProductUnion, productMapUnion.generateHealpixMap())

        # product 3
        productMapUnion = healsparse.productUnion([sparseMap1, sparseMap2, sparseMap3])

        gd, = np.where((hpmap1 > hp.UNSEEN) | (hpmap2 > hp.UNSEEN) | (hpmap3 > hp.UNSEEN))
        hpmapProductUnion = np.zeros_like(hpmap1) + hp.UNSEEN

        hpmapProductUnion[gd] = 1.0
        gd1, = np.where(hpmap1[gd] > hp.UNSEEN)
        hpmapProductUnion[gd[gd1]] *= hpmap1[gd[gd1]]
        gd2, = np.where(hpmap2[gd] > hp.UNSEEN)
        hpmapProductUnion[gd[gd2]] *= hpmap2[gd[gd2]]
        gd3, = np.where(hpmap3[gd] > hp.UNSEEN)
        hpmapProductUnion[gd[gd3]] *= hpmap3[gd[gd3]]

        testing.assert_almost_equal(hpmapProductUnion, productMapUnion.generateHealpixMap())

        # Test multiplying an int constant to a map

        multMap = sparseMap1 * 2

        hpmapProduct2 = np.zeros_like(hpmap1) + hp.UNSEEN
        gd, = np.where(hpmap1 > hp.UNSEEN)
        hpmapProduct2[gd] = hpmap1[gd] * 2

        testing.assert_almost_equal(hpmapProduct2, multMap.generateHealpixMap())

        # Test multiplying a float constant to a map

        multMap = sparseMap1 * 2.0

        hpmapProduct2 = np.zeros_like(hpmap1) + hp.UNSEEN
        gd, = np.where(hpmap1 > hp.UNSEEN)
        hpmapProduct2[gd] = hpmap1[gd] * 2.0

        testing.assert_almost_equal(hpmapProduct2, multMap.generateHealpixMap())

        # Test adding a float constant to a map, in place

        sparseMap1 *= 2.0

        testing.assert_almost_equal(hpmapProduct2, sparseMap1.generateHealpixMap())

    def test_product_integer(self):
        """
        Test map products.
        """

        random.seed(seed=12345)

        nsideCoverage = 32
        nsideMap = 64
        sentinel = 0
        maxval = 100

        # Test adding two or three maps

        sparseMap1 = healsparse.HealSparseMap.makeEmpty(
            nsideCoverage,
            nsideMap,
            np.int64,
            sentinel=sentinel,
        )
        pixel1 = np.arange(4000, 20000)
        pixel1 = np.delete(pixel1, 15000)
        values1 = random.randint(low=1, high=maxval, size=pixel1.size)
        sparseMap1.updateValues(pixel1, values1)

        hpmap1 = np.zeros(hp.nside2npix(nsideMap), dtype=np.int64)
        vpix = sparseMap1.validPixels
        hpmap1[vpix] = sparseMap1.getValuePixel(vpix)

        sparseMap2 = healsparse.HealSparseMap.makeEmpty(
            nsideCoverage,
            nsideMap,
            np.int64,
            sentinel=sentinel,
        )
        pixel2 = np.arange(15000, 25000)
        values2 = random.randint(low=1, high=maxval, size=pixel2.size)
        sparseMap2.updateValues(pixel2, values2)

        hpmap2 = np.zeros(hp.nside2npix(nsideMap), dtype=np.int64)
        vpix = sparseMap2.validPixels
        hpmap2[vpix] = sparseMap2.getValuePixel(vpix)

        sparseMap3 = healsparse.HealSparseMap.makeEmpty(
            nsideCoverage,
            nsideMap,
            np.int64,
            sentinel=sentinel,
        )
        pixel3 = np.arange(16000, 25000)
        values3 = random.randint(low=1, high=maxval, size=pixel3.size)
        sparseMap3.updateValues(pixel3, values3)

        hpmap3 = np.zeros(hp.nside2npix(nsideMap), dtype=np.int64)
        vpix = sparseMap3.validPixels
        hpmap3[vpix] = sparseMap3.getValuePixel(vpix)

        # Intersection product

        # product of 2
        productMapIntersection = healsparse.productIntersection([sparseMap1, sparseMap2])

        gd, = np.where((hpmap1 > sentinel) & (hpmap2 > sentinel))
        hpmapProductIntersection = np.zeros_like(hpmap1)
        hpmapProductIntersection[gd] = hpmap1[gd] * hpmap2[gd]

        pmap = np.zeros(hp.nside2npix(nsideMap), dtype=np.int64)
        vpix = productMapIntersection.validPixels
        pmap[vpix] = productMapIntersection.getValuePixel(vpix)

        testing.assert_equal(hpmapProductIntersection, pmap)

        # product of 3
        productMapIntersection = healsparse.productIntersection([sparseMap1, sparseMap2, sparseMap3])

        gd, = np.where((hpmap1 > sentinel) & (hpmap2 > sentinel) & (hpmap3 > sentinel))
        hpmapProductIntersection = np.zeros_like(hpmap1)
        hpmapProductIntersection[gd] = hpmap1[gd] * hpmap2[gd] * hpmap3[gd]

        pmap = np.zeros(hp.nside2npix(nsideMap), dtype=np.int64)
        vpix = productMapIntersection.validPixels
        pmap[vpix] = productMapIntersection.getValuePixel(vpix)

        testing.assert_equal(hpmapProductIntersection, pmap)

        # Union product

        # product of 2
        productMapUnion = healsparse.productUnion([sparseMap1, sparseMap2])

        gd, = np.where((hpmap1 > sentinel) | (hpmap2 > sentinel))
        hpmapProductUnion = np.zeros_like(hpmap1)

        hpmapProductUnion[gd] = 1
        gd1, = np.where(hpmap1[gd] > sentinel)
        hpmapProductUnion[gd[gd1]] *= hpmap1[gd[gd1]]
        gd2, = np.where(hpmap2[gd] > sentinel)
        hpmapProductUnion[gd[gd2]] *= hpmap2[gd[gd2]]

        pmap = np.zeros(hp.nside2npix(nsideMap), dtype=np.int64)
        vpix = productMapUnion.validPixels
        pmap[vpix] = productMapUnion.getValuePixel(vpix)

        testing.assert_equal(hpmapProductUnion, pmap)

        # product 3
        productMapUnion = healsparse.productUnion([sparseMap1, sparseMap2, sparseMap3])

        gd, = np.where((hpmap1 > sentinel) | (hpmap2 > sentinel) | (hpmap3 > sentinel))
        hpmapProductUnion = np.zeros_like(hpmap1)

        hpmapProductUnion[gd] = 1
        gd1, = np.where(hpmap1[gd] > sentinel)
        hpmapProductUnion[gd[gd1]] *= hpmap1[gd[gd1]]
        gd2, = np.where(hpmap2[gd] > sentinel)
        hpmapProductUnion[gd[gd2]] *= hpmap2[gd[gd2]]
        gd3, = np.where(hpmap3[gd] > sentinel)
        hpmapProductUnion[gd[gd3]] *= hpmap3[gd[gd3]]

        pmap = np.zeros(hp.nside2npix(nsideMap), dtype=np.int64)
        vpix = productMapUnion.validPixels
        pmap[vpix] = productMapUnion.getValuePixel(vpix)

        testing.assert_equal(hpmapProductUnion, pmap)

        # Test multiplying an int constant to a map

        multMap = sparseMap1 * 2

        hpmapProduct2 = np.zeros_like(hpmap1)
        gd, = np.where(hpmap1 > sentinel)
        hpmapProduct2[gd] = hpmap1[gd] * 2

        pmap = np.zeros(hp.nside2npix(nsideMap), dtype=np.int64)
        vpix = multMap.validPixels
        pmap[vpix] = multMap.getValuePixel(vpix)

        testing.assert_equal(hpmapProduct2, pmap)


    def test_or(self):
        """
        Test map bitwise or.
        """

        random.seed(seed=12345)

        nsideCoverage = 32
        nsideMap = 64

        sparseMap1 = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, np.int64)
        pixel1 = np.arange(4000, 20000)
        pixel1 = np.delete(pixel1, 15000)
        # Get a random list of integers
        values1 = np.random.poisson(size=pixel1.size, lam=2)
        sparseMap1.updateValues(pixel1, values1)
        hpmap1 = sparseMap1.generateHealpixMap()

        sparseMap2 = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, np.int64)
        pixel2 = np.arange(15000, 25000)
        values2 = np.random.poisson(size=pixel2.size, lam=2)
        sparseMap2.updateValues(pixel2, values2)
        hpmap2 = sparseMap2.generateHealpixMap()

        sparseMap3 = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, np.int64)
        pixel3 = np.arange(16000, 25000)
        values3 = np.random.poisson(size=pixel3.size, lam=2)
        sparseMap3.updateValues(pixel3, values3)
        hpmap3 = sparseMap3.generateHealpixMap()

        # Intersection or

        # or 2
        orMapIntersection = healsparse.orIntersection([sparseMap1, sparseMap2])

        gd, = np.where((hpmap1 > hp.UNSEEN) & (hpmap2 > hp.UNSEEN))
        hpmapOrIntersection = np.zeros_like(hpmap1) + hp.UNSEEN
        hpmapOrIntersection[gd] = hpmap1[gd].astype(np.int64) | hpmap2[gd].astype(np.int64)

        testing.assert_almost_equal(hpmapOrIntersection, orMapIntersection.generateHealpixMap())

        # or 3
        orMapIntersection = healsparse.orIntersection([sparseMap1, sparseMap2, sparseMap3])

        gd, = np.where((hpmap1 > hp.UNSEEN) & (hpmap2 > hp.UNSEEN) & (hpmap3 > hp.UNSEEN))
        hpmapOrIntersection = np.zeros_like(hpmap1) + hp.UNSEEN
        hpmapOrIntersection[gd] = hpmap1[gd].astype(np.int64) | hpmap2[gd].astype(np.int64) | hpmap3[gd].astype(np.int64)

        testing.assert_almost_equal(hpmapOrIntersection, orMapIntersection.generateHealpixMap())

        # Union or

        # or 2
        orMapUnion = healsparse.orUnion([sparseMap1, sparseMap2])

        gd, = np.where((hpmap1 > hp.UNSEEN) | (hpmap2 > hp.UNSEEN))
        hpmapOrUnion = np.zeros_like(hpmap1) + hp.UNSEEN
        hpmapOrUnion[gd] = np.clip(hpmap1[gd], 0.0, None).astype(np.int64) | np.clip(hpmap2[gd], 0.0, None).astype(np.int64)

        testing.assert_almost_equal(hpmapOrUnion, orMapUnion.generateHealpixMap())

        # or 3
        orMapUnion = healsparse.orUnion([sparseMap1, sparseMap2, sparseMap3])

        gd, = np.where((hpmap1 > hp.UNSEEN) | (hpmap2 > hp.UNSEEN) | (hpmap3 > hp.UNSEEN))
        hpmapOrUnion = np.zeros_like(hpmap1) + hp.UNSEEN
        hpmapOrUnion[gd] = np.clip(hpmap1[gd], 0.0, None).astype(np.int64) | np.clip(hpmap2[gd], 0.0, None).astype(np.int64) | np.clip(hpmap3[gd], 0.0, None).astype(np.int64)

        testing.assert_almost_equal(hpmapOrUnion, orMapUnion.generateHealpixMap())

        # Test orring an int constant to a map

        orMap = sparseMap1 | 2

        hpmapOr2 = np.zeros_like(hpmap1) + hp.UNSEEN
        gd, = np.where(hpmap1 > hp.UNSEEN)
        hpmapOr2[gd] = hpmap1[gd].astype(np.int64) | 2
        testing.assert_almost_equal(hpmapOr2, orMap.generateHealpixMap())

        # Test orring an int constant to a map, in place

        sparseMap1 |= 2

        testing.assert_almost_equal(hpmapOr2, sparseMap1.generateHealpixMap())

    def test_and(self):
        """
        Test map bitwise and.
        """

        random.seed(seed=12345)

        nsideCoverage = 32
        nsideMap = 64

        sparseMap1 = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, np.int64)
        pixel1 = np.arange(4000, 20000)
        pixel1 = np.delete(pixel1, 15000)
        # Get a random list of integers
        values1 = np.random.poisson(size=pixel1.size, lam=2)
        sparseMap1.updateValues(pixel1, values1)
        hpmap1 = sparseMap1.generateHealpixMap()

        sparseMap2 = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, np.int64)
        pixel2 = np.arange(15000, 25000)
        values2 = np.random.poisson(size=pixel2.size, lam=2)
        sparseMap2.updateValues(pixel2, values2)
        hpmap2 = sparseMap2.generateHealpixMap()

        sparseMap3 = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, np.int64)
        pixel3 = np.arange(16000, 25000)
        values3 = np.random.poisson(size=pixel3.size, lam=2)
        sparseMap3.updateValues(pixel3, values3)
        hpmap3 = sparseMap3.generateHealpixMap()

        # Intersection and

        # and 2
        andMapIntersection = healsparse.andIntersection([sparseMap1, sparseMap2])

        gd, = np.where((hpmap1 > hp.UNSEEN) & (hpmap2 > hp.UNSEEN))
        hpmapAndIntersection = np.zeros_like(hpmap1) + hp.UNSEEN
        hpmapAndIntersection[gd] = hpmap1[gd].astype(np.int64) & hpmap2[gd].astype(np.int64)

        testing.assert_almost_equal(hpmapAndIntersection, andMapIntersection.generateHealpixMap())

        # and 3
        andMapIntersection = healsparse.andIntersection([sparseMap1, sparseMap2, sparseMap3])

        gd, = np.where((hpmap1 > hp.UNSEEN) & (hpmap2 > hp.UNSEEN) & (hpmap3 > hp.UNSEEN))
        hpmapAndIntersection = np.zeros_like(hpmap1) + hp.UNSEEN
        hpmapAndIntersection[gd] = hpmap1[gd].astype(np.int64) & hpmap2[gd].astype(np.int64) & hpmap3[gd].astype(np.int64)

        testing.assert_almost_equal(hpmapAndIntersection, andMapIntersection.generateHealpixMap())

        # Union and

        # and 2
        andMapUnion = healsparse.andUnion([sparseMap1, sparseMap2])

        gd, = np.where((hpmap1 > hp.UNSEEN) | (hpmap2 > hp.UNSEEN))
        hpmapAndUnion = np.zeros_like(hpmap1) + hp.UNSEEN

        hpmapAndUnion[gd] = -1.0
        gd1, = np.where(hpmap1[gd] > hp.UNSEEN)
        hpmapAndUnion[gd[gd1]] = hpmapAndUnion[gd[gd1]].astype(np.int64) & hpmap1[gd[gd1]].astype(np.int64)
        gd2, = np.where(hpmap2[gd] > hp.UNSEEN)
        hpmapAndUnion[gd[gd2]] = hpmapAndUnion[gd[gd2]].astype(np.int64) & hpmap2[gd[gd2]].astype(np.int64)

        testing.assert_almost_equal(hpmapAndUnion, andMapUnion.generateHealpixMap())

        # and 3
        andMapUnion = healsparse.andUnion([sparseMap1, sparseMap2, sparseMap3])

        gd, = np.where((hpmap1 > hp.UNSEEN) | (hpmap2 > hp.UNSEEN) | (hpmap3 > hp.UNSEEN))
        hpmapAndUnion = np.zeros_like(hpmap1) + hp.UNSEEN

        hpmapAndUnion[gd] = -1.0
        gd1, = np.where(hpmap1[gd] > hp.UNSEEN)
        hpmapAndUnion[gd[gd1]] = hpmapAndUnion[gd[gd1]].astype(np.int64) & hpmap1[gd[gd1]].astype(np.int64)
        gd2, = np.where(hpmap2[gd] > hp.UNSEEN)
        hpmapAndUnion[gd[gd2]] = hpmapAndUnion[gd[gd2]].astype(np.int64) & hpmap2[gd[gd2]].astype(np.int64)
        gd3, = np.where(hpmap3[gd] > hp.UNSEEN)
        hpmapAndUnion[gd[gd3]] = hpmapAndUnion[gd[gd3]].astype(np.int64) & hpmap3[gd[gd3]].astype(np.int64)

        testing.assert_almost_equal(hpmapAndUnion, andMapUnion.generateHealpixMap())

        # Test anding an int constant to a map

        andMap = sparseMap1 & 2

        hpmapAnd2 = np.zeros_like(hpmap1) + hp.UNSEEN
        gd, = np.where(hpmap1 > hp.UNSEEN)
        hpmapAnd2[gd] = hpmap1[gd].astype(np.int64) & 2
        testing.assert_almost_equal(hpmapAnd2, andMap.generateHealpixMap())

        # Test anding an int constant to a map, in place

        sparseMap1 &= 2

        testing.assert_almost_equal(hpmapAnd2, sparseMap1.generateHealpixMap())

    def test_xor(self):
        """
        Test map bitwise xor.
        """
        random.seed(seed=12345)

        nsideCoverage = 32
        nsideMap = 64

        sparseMap1 = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, np.int64)
        pixel1 = np.arange(4000, 20000)
        pixel1 = np.delete(pixel1, 15000)
        # Get a random list of integers
        values1 = np.random.poisson(size=pixel1.size, lam=2)
        sparseMap1.updateValues(pixel1, values1)
        hpmap1 = sparseMap1.generateHealpixMap()

        sparseMap2 = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, np.int64)
        pixel2 = np.arange(15000, 25000)
        values2 = np.random.poisson(size=pixel2.size, lam=2)
        sparseMap2.updateValues(pixel2, values2)
        hpmap2 = sparseMap2.generateHealpixMap()

        sparseMap3 = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, np.int64)
        pixel3 = np.arange(16000, 25000)
        values3 = np.random.poisson(size=pixel3.size, lam=2)
        sparseMap3.updateValues(pixel3, values3)
        hpmap3 = sparseMap3.generateHealpixMap()

        # Intersection xor

        # xor 2
        xorMapIntersection = healsparse.xorIntersection([sparseMap1, sparseMap2])

        gd, = np.where((hpmap1 > hp.UNSEEN) & (hpmap2 > hp.UNSEEN))
        hpmapXorIntersection = np.zeros_like(hpmap1) + hp.UNSEEN
        hpmapXorIntersection[gd] = hpmap1[gd].astype(np.int64) ^ hpmap2[gd].astype(np.int64)

        testing.assert_almost_equal(hpmapXorIntersection, xorMapIntersection.generateHealpixMap())

        # xor 3
        xorMapIntersection = healsparse.xorIntersection([sparseMap1, sparseMap2, sparseMap3])

        gd, = np.where((hpmap1 > hp.UNSEEN) & (hpmap2 > hp.UNSEEN) & (hpmap3 > hp.UNSEEN))
        hpmapXorIntersection = np.zeros_like(hpmap1) + hp.UNSEEN
        hpmapXorIntersection[gd] = hpmap1[gd].astype(np.int64) ^ hpmap2[gd].astype(np.int64) ^ hpmap3[gd].astype(np.int64)

        testing.assert_almost_equal(hpmapXorIntersection, xorMapIntersection.generateHealpixMap())

        # Union xor

        # xor 2
        xorMapUnion = healsparse.xorUnion([sparseMap1, sparseMap2])

        gd, = np.where((hpmap1 > hp.UNSEEN) | (hpmap2 > hp.UNSEEN))
        hpmapXorUnion = np.zeros_like(hpmap1) + hp.UNSEEN

        hpmapXorUnion[gd] = 0.0
        gd1, = np.where(hpmap1[gd] > hp.UNSEEN)
        hpmapXorUnion[gd[gd1]] = hpmapXorUnion[gd[gd1]].astype(np.int64) ^ hpmap1[gd[gd1]].astype(np.int64)
        gd2, = np.where(hpmap2[gd] > hp.UNSEEN)
        hpmapXorUnion[gd[gd2]] = hpmapXorUnion[gd[gd2]].astype(np.int64) ^ hpmap2[gd[gd2]].astype(np.int64)

        testing.assert_almost_equal(hpmapXorUnion, xorMapUnion.generateHealpixMap())

        # xor 3
        xorMapUnion = healsparse.xorUnion([sparseMap1, sparseMap2, sparseMap3])

        gd, = np.where((hpmap1 > hp.UNSEEN) | (hpmap2 > hp.UNSEEN) | (hpmap3 > hp.UNSEEN))
        hpmapXorUnion = np.zeros_like(hpmap1) + hp.UNSEEN

        hpmapXorUnion[gd] = 0.0
        gd1, = np.where(hpmap1[gd] > hp.UNSEEN)
        hpmapXorUnion[gd[gd1]] = hpmapXorUnion[gd[gd1]].astype(np.int64) ^ hpmap1[gd[gd1]].astype(np.int64)
        gd2, = np.where(hpmap2[gd] > hp.UNSEEN)
        hpmapXorUnion[gd[gd2]] = hpmapXorUnion[gd[gd2]].astype(np.int64) ^ hpmap2[gd[gd2]].astype(np.int64)
        gd3, = np.where(hpmap3[gd] > hp.UNSEEN)
        hpmapXorUnion[gd[gd3]] = hpmapXorUnion[gd[gd3]].astype(np.int64) ^ hpmap3[gd[gd3]].astype(np.int64)

        testing.assert_almost_equal(hpmapXorUnion, xorMapUnion.generateHealpixMap())

        # Test xorring an int constant to a map

        xorMap = sparseMap1 ^ 2

        hpmapXor2 = np.zeros_like(hpmap1) + hp.UNSEEN
        gd, = np.where(hpmap1 > hp.UNSEEN)
        hpmapXor2[gd] = hpmap1[gd].astype(np.int64) ^ 2
        testing.assert_almost_equal(hpmapXor2, xorMap.generateHealpixMap())

        # Test xorring an int constant to a map, in place

        sparseMap1 ^= 2

        testing.assert_almost_equal(hpmapXor2, sparseMap1.generateHealpixMap())

    def test_miscellaneous_operations(self):
        """
        Test miscellaneous constant operations.
        """
        random.seed(seed=12345)

        nsideCoverage = 32
        nsideMap = 64

        sparseMap1 = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, np.float64)
        pixel1 = np.arange(4000, 20000)
        pixel1 = np.delete(pixel1, 15000)
        values1 = np.random.random(size=pixel1.size)
        sparseMap1.updateValues(pixel1, values1)
        hpmap1 = sparseMap1.generateHealpixMap()

        # subtraction
        testMap = sparseMap1 - 2.0

        hpmapTest = np.zeros_like(hpmap1) + hp.UNSEEN
        gd, = np.where(hpmap1 > hp.UNSEEN)
        hpmapTest[gd] = hpmap1[gd] - 2.0

        testing.assert_almost_equal(hpmapTest, testMap.generateHealpixMap())

        testMap = sparseMap1.copy()
        testMap -= 2.0
        testing.assert_almost_equal(hpmapTest, testMap.generateHealpixMap())

        # division
        testMap = sparseMap1 / 2.0

        hpmapTest = np.zeros_like(hpmap1) + hp.UNSEEN
        gd, = np.where(hpmap1 > hp.UNSEEN)
        hpmapTest[gd] = hpmap1[gd] / 2.0

        testing.assert_almost_equal(hpmapTest, testMap.generateHealpixMap())

        testMap = sparseMap1.copy()
        testMap /= 2.0
        testing.assert_almost_equal(hpmapTest, testMap.generateHealpixMap())

        # power
        testMap = sparseMap1 ** 2.0

        hpmapTest = np.zeros_like(hpmap1) + hp.UNSEEN
        gd, = np.where(hpmap1 > hp.UNSEEN)
        hpmapTest[gd] = hpmap1[gd] ** 2.0

        testing.assert_almost_equal(hpmapTest, testMap.generateHealpixMap())

        testMap = sparseMap1.copy()
        testMap **= 2.0
        testing.assert_almost_equal(hpmapTest, testMap.generateHealpixMap())


if __name__=='__main__':
    unittest.main()


