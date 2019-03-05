from __future__ import division, absolute_import, print_function

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
from numpy import random
import tempfile
import shutil
import os

import healsparse

class IoTestCase(unittest.TestCase):
    def test_writeread(self):
        """
        Test i/o functionality
        """

        random.seed(seed=12345)

        nsideCoverage = 32
        nsideMap = 64

        nRand = 1000
        ra = np.random.random(nRand) * 360.0
        dec = np.random.random(nRand) * 180.0 - 90.0

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        # Generate a random map

        fullMap = np.zeros(hp.nside2npix(nsideMap)) + hp.UNSEEN
        fullMap[0:20000] = np.random.random(size=20000)

        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        ipnest = hp.ang2pix(nsideMap, theta, phi, nest=True)

        testValues = fullMap[ipnest]

        # Save it with healpy in ring

        fullMapRing = hp.reorder(fullMap, n2r=True)
        hp.write_map(os.path.join(self.test_dir, 'healpix_map_ring.fits'), fullMapRing)

        # Read it with healsparse
        # TODO Test that we raise an exception when nsideCoverage isn't set

        sparseMap = healsparse.HealSparseMap.read(os.path.join(self.test_dir, 'healpix_map_ring.fits'), nsideCoverage=nsideCoverage)

        # Check that we can do a basic lookup
        testing.assert_almost_equal(sparseMap.getValuePixel(ipnest), testValues)

        # Save map to healpy in nest
        hp.write_map(os.path.join(self.test_dir, 'healpix_map_nest.fits'), fullMap)

        # Read it with healsparse
        sparseMap = healsparse.HealSparseMap.read(os.path.join(self.test_dir, 'healpix_map_ring.fits'), nsideCoverage=nsideCoverage)

        # Check that we can do a basic lookup
        testing.assert_almost_equal(sparseMap.getValuePixel(ipnest), testValues)

        # Write it to healsparse format
        sparseMap.write(os.path.join(self.test_dir, 'healsparse_map.fits'))

        # Read in healsparse format (full map)
        sparseMap = healsparse.HealSparseMap.read(os.path.join(self.test_dir, 'healsparse_map.fits'))

        # Check that we can do a basic lookup
        testing.assert_almost_equal(sparseMap.getValuePixel(ipnest), testValues)

        # Try to read in healsparse format, non-unique pixels
        self.assertRaises(RuntimeError, healsparse.HealSparseMap.read, os.path.join(self.test_dir, 'healsparse_map.fits'), pixels=[0, 0])

        # Read in healsparse format (two pixels)
        sparseMapSmall = healsparse.HealSparseMap.read(os.path.join(self.test_dir, 'healsparse_map.fits'), pixels=[0, 1])

        # Test the coverage map only has two pixels
        covMask = sparseMapSmall.coverageMask
        self.assertEqual(covMask.sum(), 2)

        # Test lookup of values in those two pixels
        ipnestCov = np.right_shift(ipnest, sparseMapSmall._bitShift)
        outsideSmall, = np.where(ipnestCov > 1)
        testValues2 = testValues.copy()
        testValues2[outsideSmall] = hp.UNSEEN

        testing.assert_almost_equal(sparseMapSmall.getValuePixel(ipnest), testValues2)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)

if __name__=='__main__':
    unittest.main()
