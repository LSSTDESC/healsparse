from __future__ import division, absolute_import, print_function

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
from numpy import random

import healsparse

class LookupTestCase(unittest.TestCase):
    def test_lookup(self):
        """
        Test lookup functionality
        """

        np.random.seed(12345)

        nsideCoverage = 32
        nsideMap = 1024

        fullMap = np.zeros(hp.nside2npix(nsideMap)) + hp.UNSEEN
        fullMap[0: 200000] = np.random.random(size=200000)

        sparseMap = healsparse.HealSparseMap(healpixMap=fullMap, nsideCoverage=nsideCoverage)

        nRand = 100000
        ra = np.random.random(nRand) * 360.0
        dec = np.random.random(nRand) * 180.0 - 90.0

        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        ipnest = hp.ang2pix(nsideMap, theta, phi, nest=True)

        testValues = fullMap[ipnest]

        # Test the pixel lookup
        compValues = sparseMap.getValuePixel(ipnest)
        testing.assert_almost_equal(compValues, testValues)

        # Test pixel lookup (ring)
        ipring = hp.nest2ring(nsideMap, ipnest)
        compValues = sparseMap.getValuePixel(ipring, nest=False)
        testing.assert_almost_equal(compValues, testValues)

        # Test the theta/phi lookup
        compValues = sparseMap.getValueThetaPhi(theta, phi)
        testing.assert_almost_equal(compValues, testValues)

        # Test the ra/dec lookup
        compValues = sparseMap.getValueRaDec(ra, dec)
        testing.assert_almost_equal(compValues, testValues)

        # Test the list of valid pixels
        validPixels = sparseMap.validPixels
        testing.assert_equal(validPixels, np.where(fullMap > hp.UNSEEN)[0])

if __name__=='__main__':
    unittest.main()
