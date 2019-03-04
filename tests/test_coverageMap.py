from __future__ import division, absolute_import, print_function

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
from numpy import random
import healsparse

class CoverageMapTestCase(unittest.TestCase):
    def test_coverageMap(self):
        """
        Test coverageMap functionality
        """

        nsideCoverage = 16
        nsideMap = 512
        non_masked_px = 10 # Number of non-masked pixels in the coverage map resolution
        nfine = (nsideMap//nsideCoverage)**2
        fullMap = np.zeros(hp.nside2npix(nsideMap)) + hp.UNSEEN
        fullMap[0:non_masked_px*nfine] = 1+np.random.random(size=non_masked_px*nfine)       
        
        # Generate sparse map

        sparseMap = healsparse.HealSparseMap(healpixMap=fullMap, nsideCoverage=nsideCoverage)
        
        # Build the "original" coverage map

        covMap_orig = np.zeros(hp.nside2npix(nsideCoverage), dtype=np.double)
        idx_cov = np.right_shift(np.arange(0,non_masked_px*nfine), sparseMap._bitShift)
        unique_idx_cov = np.unique(idx_cov)
        idx_counts = np.bincount(idx_cov, minlength=hp.nside2npix(nsideCoverage)).astype(float) 
        covMap_orig[unique_idx_cov] = idx_counts[unique_idx_cov]/nfine 
        
        # Get the built coverage map

        covMap = sparseMap.coverageMap


        # Test the coverage map generation and lookup
        
        testing.assert_equal(covMap_orig, covMap)

if __name__=='__main__':
    unittest.main()
