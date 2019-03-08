from __future__ import division, absolute_import, print_function

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
from numpy import random
import healsparse

class DegradeMapTestCase(unittest.TestCase):
    def test_degradeMap(self):
        """
        Test HealSparse.degrade functionality
        """
        random.seed(12345)
        nsideCoverage = 32
        nsideMap = 1024
        nsideNew = 256
        fullMap = random.random(hp.nside2npix(nsideMap))

        # Generate sparse map

        sparseMap = healsparse.HealSparseMap(healpixMap=fullMap, nsideCoverage=nsideCoverage)
        
        # Degrade original HEALPix map

        deg_map = hp.ud_grade(fullMap, nside_out=nsideNew, order_in='NESTED', order_out='NESTED')

        # Degrade sparse map and compare to original

        sparseMap = sparseMap.degrade(sparseMap, nside_out=nsideNew)

        # Test the coverage map generation and lookup

        testing.assert_equal(deg_map, sparseMap._sparseMap[2**sparseMap._bitShift:])

if __name__=='__main__':
    unittest.main()
