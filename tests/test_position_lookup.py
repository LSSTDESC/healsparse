from __future__ import division, absolute_import, print_function

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
from numpy import random
import os
import matplotlib.pyplot as plt
import healsparse

class PositionLookupTestCase(unittest.TestCase):
    def test_PositionLookup(self):
        """
        Test the conversion from ra, dec to healsparse index and viceversa
        """
        random.seed(seed=12345)

        nsideCoverage = 32
        nsideMap = 64

        nRand = 1000
        ra = np.random.random(nRand) * 360.0
        dec = np.random.random(nRand) * 180.0 - 90.0
        value = np.random.random(nRand)
        healpixMap = np.zeros(hp.nside2npix(nsideMap))+hp.UNSEEN
        healpixMap[hp.ang2pix(nsideMap, np.radians(90-dec), np.radians(ra), nest=True)] = value
        sparseMap = healsparse.HealSparseMap(nsideCoverage=nsideCoverage, healpixMap=healpixMap)
        # Get the populated pixel numbers in the sparseMap
        pop_idx, = np.where(sparseMap._sparseMap > hp.UNSEEN)
        # Get the ra, dec of the center of the pixel
        ra_sp, dec_sp = sparseMap.HSPixel2RaDec(pop_idx)
        # Get the indices corresponding to those positions
        idx_rec = sparseMap.RaDec2HSPixel(ra_sp, dec_sp)
        # Check that the indices are the same and close the loop
        testing.assert_almost_equal(pop_idx, idx_rec)
if __name__=='__main__':
    unittest.main()
