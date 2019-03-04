from __future__ import division, absolute_import, print_function

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
from numpy import random
import healsparse

class CoverageMaskTestCase(unittest.TestCase):
    def test_coverageMask(self):
        """
        Test coverageMask functionality
        """

        nsideCoverage = 8
        nsideMap = 64
        non_masked_px = 10 # Number of non-masked pixels in the coverage map resolution
        nfine = (nsideMap//nsideCoverage)**2
        fullMap = np.zeros(hp.nside2npix(nsideMap)) + hp.UNSEEN
        fullMap[0:non_masked_px*nfine] = 1+np.random.random(size=non_masked_px*nfine)       
        
        # Generate sparse map

        sparseMap = healsparse.HealSparseMap(healpixMap=fullMap, nsideCoverage=nsideCoverage)
        
        # Build the "original" coverage mask

        covMask_orig = np.zeros(hp.nside2npix(nsideCoverage), dtype=np.bool)
        idx_cov = np.unique(np.right_shift(np.arange(0,non_masked_px*nfine), sparseMap._bitShift)) 
        covMask_orig[idx_cov] = 1
        
        # Get the built coverage mask

        covMask = sparseMap.coverageMask


        # Test the mask generation and lookup
        
        testing.assert_equal(covMask_orig, covMask)

if __name__=='__main__':
    unittest.main()
