from __future__ import division, absolute_import, print_function

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
from numpy import random
import tempfile
import shutil
import os
import fitsio

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
        fullMap[0: 20000] = np.random.random(size=20000)

        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        ipnest = hp.ang2pix(nsideMap, theta, phi, nest=True)

        testValues = fullMap[ipnest]

        # Save it with healpy in ring

        fullMapRing = hp.reorder(fullMap, n2r=True)
        hp.write_map(os.path.join(self.test_dir, 'healpix_map_ring.fits'), fullMapRing, dtype=np.float64)

        # Read it with healsparse
        # TODO Test that we raise an exception when nsideCoverage isn't set

        sparseMap = healsparse.HealSparseMap.read(os.path.join(self.test_dir, 'healpix_map_ring.fits'), nsideCoverage=nsideCoverage)

        # Check that we can do a basic lookup
        testing.assert_almost_equal(sparseMap.getValuePixel(ipnest), testValues)

        # Save map to healpy in nest
        hp.write_map(os.path.join(self.test_dir, 'healpix_map_nest.fits'), fullMap, dtype=np.float64, nest=True)

        # Read it with healsparse
        sparseMap = healsparse.HealSparseMap.read(os.path.join(self.test_dir, 'healpix_map_nest.fits'), nsideCoverage=nsideCoverage)

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

    def test_readOutOfOrder(self):
        """
        Test reading maps that have been written with out-of-order pixels
        """

        random.seed(seed=12345)

        nsideCoverage = 32
        nsideMap = 64

        nRand = 1000
        ra = np.random.random(nRand) * 360.0
        dec = np.random.random(nRand) * 180.0 - 90.0

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        # Create an empty map
        sparseMap = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, np.float64)

        # Fill it out of order
        pixel = np.arange(4000, 20000)
        values = np.random.random(pixel.size)
        sparseMap.updateValues(pixel, values)
        pixel2 = np.arange(1000)
        values2 = np.random.random(pixel2.size)
        sparseMap.updateValues(pixel2, values2)

        sparseMap.write(os.path.join(self.test_dir, 'healsparse_map_outoforder.fits'))

        # And read it in...
        sparseMap = healsparse.HealSparseMap.read(os.path.join(self.test_dir, 'healsparse_map_outoforder.fits'))

        # Test some values
        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        ipnest = hp.ang2pix(nsideMap, theta, phi)
        testMap = np.zeros(hp.nside2npix(nsideMap)) + hp.UNSEEN
        testMap[pixel] = values
        testMap[pixel2] = values2

        testing.assert_almost_equal(sparseMap.getValuePixel(ipnest), testMap[ipnest])

        # Read in the first two and the Nth pixel

        # These pixels are chosen because they are covered by the random test points
        sparseMapSmall = healsparse.HealSparseMap.read(os.path.join(self.test_dir, 'healsparse_map_outoforder.fits'), pixels=[0, 1, 3179])

        # Test some values
        ipnestCov = np.right_shift(ipnest, sparseMapSmall._bitShift)
        testValuesSmall = testMap[ipnest]
        outsideSmall, = np.where((ipnestCov != 0) & (ipnestCov != 1) & (ipnestCov != 3179))
        testValuesSmall[outsideSmall] = hp.UNSEEN

        testing.assert_almost_equal(sparseMapSmall.getValuePixel(ipnest), testValuesSmall)

    def test_writeread_withheader(self):
        """
        Test i/o functionality with a header
        """

        random.seed(seed=12345)

        nsideCoverage = 32
        nsideMap = 64

        nRand = 1000
        ra = np.random.random(nRand) * 360.0
        dec = np.random.random(nRand) * 180.0 - 90.0

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        fullMap = np.zeros(hp.nside2npix(nsideMap)) + hp.UNSEEN
        fullMap[0: 20000] = np.random.random(size=20000)

        sparseMap = healsparse.HealSparseMap(healpixMap=fullMap, nsideCoverage=nsideCoverage, nest=True)
        hdr = fitsio.FITSHDR()
        hdr['TESTING'] = 1.0

        sparseMap.write(os.path.join(self.test_dir, 'sparsemap_with_header.fits'), header=hdr)

        retMap, retHdr = healsparse.HealSparseMap.read(os.path.join(self.test_dir, 'sparsemap_with_header.fits'), header=True)

        self.assertEqual(hdr['TESTING'], retHdr['TESTING'])

    def test_writeread_highres(self):
        """
        Test i/o functionality at very high resolution
        """

        random.seed(seed=12345)

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        nsideCoverage = 128
        nsideMap = 2**17

        vec = hp.ang2vec(100.0, 0.0, lonlat=True)
        rad = np.radians(0.2/60.)
        pixels = hp.query_disc(nsideMap, vec, rad, nest=True, inclusive=False)
        pixels.sort()
        values = np.zeros(pixels.size, dtype=np.int32) + 8

        sparseMap = healsparse.HealSparseMap.makeEmpty(nsideSparse=nsideMap, nsideCoverage=nsideCoverage, dtype=np.int32)
        sparseMap.updateValues(pixels, values)

        validPixels = sparseMap.validPixels
        validPixels.sort()

        testing.assert_array_equal(validPixels, pixels)
        testing.assert_array_equal(sparseMap.getValuePixel(validPixels), values)

        sparseMap.write(os.path.join(self.test_dir, 'healsparse_map.fits'))

        sparseMap2 = healsparse.HealSparseMap.read(os.path.join(self.test_dir, 'healsparse_map.fits'))

        validPixels2 = sparseMap2.validPixels

        testing.assert_array_equal(validPixels2, pixels)
        testing.assert_array_equal(sparseMap2.getValuePixel(validPixels2), values)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)

if __name__=='__main__':
    unittest.main()
