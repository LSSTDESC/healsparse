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

        # Test adding two maps

        sparseMap = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, np.float64)
        pixel = np.arange(4000, 20000)
        values = np.random.random(size=pixel.size)
        sparseMap.updateValues(pixel, values)

        sparseMap2 = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, np.float64)
        pixel2 = np.arange(15000, 25000)
        values2 = np.random.random(size=pixel2.size)
        sparseMap2.updateValues(pixel2, values2)

        addedMap = sparseMap + sparseMap2

        hpmap1 = sparseMap.generateHealpixMap()
        hpmap2 = sparseMap2.generateHealpixMap()
        gd, = np.where((hpmap1 > hp.UNSEEN) & (hpmap2 > hp.UNSEEN))
        hpmap3 = np.zeros_like(hpmap1) + hp.UNSEEN
        hpmap3[gd] = hpmap1[gd] + hpmap2[gd]

        addedHpMap = addedMap.generateHealpixMap()

        testing.assert_almost_equal(hpmap3, addedHpMap)

        # Test adding a float constant to a map

        addedMap = sparseMap + 2.0

        hpmap3 = np.zeros_like(hpmap1) + hp.UNSEEN
        gd, = np.where(hpmap1 > hp.UNSEEN)
        hpmap3[gd] = hpmap1[gd] + 2.0

        addedHpMap = addedMap.generateHealpixMap()
        testing.assert_almost_equal(hpmap3, addedHpMap)

        # Test adding an int constant to a map

        addedMap = sparseMap + 2

        hpmap3 = np.zeros_like(hpmap1) + hp.UNSEEN
        gd, = np.where(hpmap1 > hp.UNSEEN)
        hpmap3[gd] = hpmap1[gd] + 2

        addedHpMap = addedMap.generateHealpixMap()
        testing.assert_almost_equal(hpmap3, addedHpMap)


if __name__=='__main__':
    unittest.main()


