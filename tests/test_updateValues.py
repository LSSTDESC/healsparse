from __future__ import division, absolute_import, print_function

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
from numpy import random
import os
import healsparse

class UpdateValuesTestCase(unittest.TestCase):
    def test_updateValues_inorder(self):
        """
        Test doing updateValues, in coarse pixel order.
        """

        nsideCoverage = 32
        nsideMap = 64
        dtype = np.float64

        sparseMap = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, dtype)

        nFinePerCov = 2**sparseMap._bitShift

        testPix = np.arange(nFinePerCov) + nFinePerCov * 10
        testValues = np.zeros(nFinePerCov)

        sparseMap.updateValues(testPix, testValues)
        testing.assert_almost_equal(sparseMap.getValuePixel(testPix), testValues)

        validPixels = sparseMap.validPixels
        testing.assert_equal(validPixels, testPix)

        testPix2 = np.arange(nFinePerCov) + nFinePerCov * 16
        testValues2 = np.zeros(nFinePerCov) + 100

        sparseMap.updateValues(testPix2, testValues2)
        testing.assert_almost_equal(sparseMap.getValuePixel(testPix), testValues)
        testing.assert_almost_equal(sparseMap.getValuePixel(testPix2), testValues2)

        validPixels = sparseMap.validPixels
        testing.assert_equal(np.sort(validPixels), np.sort(np.concatenate((testPix, testPix2))))

    def test_updateValues_outoforder(self):
        """
        Test doing updateValues, out of order.
        """

        nsideCoverage = 32
        nsideMap = 64
        dtype = np.float64

        sparseMap = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, dtype)

        nFinePerCov = 2**sparseMap._bitShift

        testPix = np.arange(nFinePerCov) + nFinePerCov * 16
        testValues = np.zeros(nFinePerCov)

        sparseMap.updateValues(testPix, testValues)
        testing.assert_almost_equal(sparseMap.getValuePixel(testPix), testValues)

        validPixels = sparseMap.validPixels
        testing.assert_equal(validPixels, testPix)

        testPix2 = np.arange(nFinePerCov) + nFinePerCov * 10
        testValues2 = np.zeros(nFinePerCov) + 100

        sparseMap.updateValues(testPix2, testValues2)
        testing.assert_almost_equal(sparseMap.getValuePixel(testPix), testValues)
        testing.assert_almost_equal(sparseMap.getValuePixel(testPix2), testValues2)

        validPixels = sparseMap.validPixels
        testing.assert_equal(np.sort(validPixels), np.sort(np.concatenate((testPix, testPix2))))


if __name__=='__main__':
    unittest.main()
