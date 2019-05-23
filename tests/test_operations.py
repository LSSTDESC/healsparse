from __future__ import division, absolute_import, print_function

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
from numpy import random

import healsparse

class OperationsTestCase(unittest.TestCase):
    def test_addition(self):
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

        # Test adding a float constant to a map

        addedMap = sparseMap1 + 2.0

        hpmapAdd2 = np.zeros_like(hpmap1) + hp.UNSEEN
        gd, = np.where(hpmap1 > hp.UNSEEN)
        hpmapAdd2[gd] = hpmap1[gd] + 2.0

        testing.assert_almost_equal(hpmapAdd2, addedMap.generateHealpixMap())

        # Test adding an int constant to a map

        addedMap = sparseMap1 + 2

        hpmapAdd2 = np.zeros_like(hpmap1) + hp.UNSEEN
        gd, = np.where(hpmap1 > hp.UNSEEN)
        hpmapAdd2[gd] = hpmap1[gd] + 2

        testing.assert_almost_equal(hpmapAdd2, addedMap.generateHealpixMap())

        # Test adding a float constant to a map, in place

        sparseMap1 += 2.0
        hpmapAdd2 = np.zeros_like(hpmap1) + hp.UNSEEN
        gd, = np.where(hpmap1 > hp.UNSEEN)
        hpmapAdd2[gd] = hpmap1[gd] + 2.0

        testing.assert_almost_equal(hpmapAdd2, sparseMap1.generateHealpixMap())


if __name__=='__main__':
    unittest.main()


