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

class RecArrayTestCase(unittest.TestCase):
    def test_writereadRecarray(self):
        """
        Test recarray writing and reading.
        """

        random.seed(seed=12345)

        nsideCoverage = 32
        nsideMap = 64

        nRand = 1000
        ra = np.random.random(nRand) * 360.0
        dec = np.random.random(nRand) * 180.0 - 90.0

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        dtype = [('col1', 'f8'), ('col2', 'f8')]
        sparseMap = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, dtype, primary='col1')
        pixel = np.arange(20000)
        values = np.zeros_like(pixel, dtype=dtype)
        values['col1'] = np.random.random(size=pixel.size)
        values['col2'] = np.random.random(size=pixel.size)
        sparseMap.updateValues(pixel, values)

        sparseMap.write(os.path.join(self.test_dir, 'healsparse_map_recarray.fits'))

        # Make the test values
        hpmapCol1 = np.zeros(hp.nside2npix(nsideMap)) + hp.UNSEEN
        hpmapCol2 = np.zeros(hp.nside2npix(nsideMap)) + hp.UNSEEN
        hpmapCol1[pixel] = values['col1']
        hpmapCol2[pixel] = values['col2']
        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        ipnestTest = hp.ang2pix(nsideMap, theta, phi, nest=True)

        # Read in the map
        sparseMap = healsparse.HealSparseMap.read(os.path.join(self.test_dir, 'healsparse_map_recarray.fits'))

        testing.assert_almost_equal(sparseMap.getValueRaDec(ra, dec)['col1'], hpmapCol1[ipnestTest])
        testing.assert_almost_equal(sparseMap.getValueRaDec(ra, dec)['col2'], hpmapCol2[ipnestTest])

        # Test the list of valid pixels
        validPixels = sparseMap.validPixels
        testing.assert_equal(validPixels, pixel)

        # Read in a partial map...
        sparseMapSmall = healsparse.HealSparseMap.read(os.path.join(self.test_dir, 'healsparse_map_recarray.fits'), pixels=[0, 1])

        # Test the coverage map only has two pixels
        covMask = sparseMapSmall.coverageMask
        self.assertEqual(covMask.sum(), 2)

        # Test lookup of values in these two pixels
        ipnestCov = np.right_shift(ipnestTest, sparseMapSmall._bitShift)
        outsideSmall, = np.where(ipnestCov > 1)
        # column1 is the "primary" column and will return UNSEEN
        testValues1b = hpmapCol1[ipnestTest].copy()
        testValues1b[outsideSmall] = hp.UNSEEN
        # column2 is not the primary column and will also return UNSEEN
        testValues2b = hpmapCol2[ipnestTest].copy()
        testValues2b[outsideSmall] = hp.UNSEEN

        testing.assert_almost_equal(sparseMapSmall.getValuePixel(ipnestTest)['col1'], testValues1b)
        testing.assert_almost_equal(sparseMapSmall.getValuePixel(ipnestTest)['col2'], testValues2b)


    def test_readOutOfOrderRecarray(self):
        """
        Test reading of recarray maps that have been written out-of-order
        """

        random.seed(seed=12345)

        nsideCoverage = 32
        nsideMap = 64

        nRand = 1000
        ra = np.random.random(nRand) * 360.0
        dec = np.random.random(nRand) * 180.0 - 90.0

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        # Create an empty map
        dtype = [('col1', 'f8'), ('col2', 'f8')]
        sparseMap = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, dtype, primary='col1')
        pixel = np.arange(4000, 20000)
        values = np.zeros_like(pixel, dtype=dtype)
        values['col1'] = np.random.random(size=pixel.size)
        values['col2'] = np.random.random(size=pixel.size)
        sparseMap.updateValues(pixel, values)
        pixel2 = np.arange(1000)
        values2 = np.zeros_like(pixel2, dtype=dtype)
        values2['col1'] = np.random.random(size=pixel2.size)
        values2['col2'] = np.random.random(size=pixel2.size)
        sparseMap.updateValues(pixel2, values2)

        sparseMap.write(os.path.join(self.test_dir, 'healsparse_map_recarray_outoforder.fits'))

        # And read it in...
        sparseMap = healsparse.HealSparseMap.read(os.path.join(self.test_dir, 'healsparse_map_recarray_outoforder.fits'))

        # Test some values
        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        ipnest = hp.ang2pix(nsideMap, theta, phi)
        testMapCol1 = np.zeros(hp.nside2npix(nsideMap)) + hp.UNSEEN
        testMapCol1[pixel] = values['col1']
        testMapCol1[pixel2] = values2['col1']
        testMapCol2 = np.zeros(hp.nside2npix(nsideMap)) + hp.UNSEEN
        testMapCol2[pixel] = values['col2']
        testMapCol2[pixel2] = values2['col2']

        testing.assert_almost_equal(sparseMap.getValuePixel(ipnest)['col1'], testMapCol1[ipnest])
        testing.assert_almost_equal(sparseMap.getValuePixel(ipnest)['col2'], testMapCol2[ipnest])

        # These pixels are chosen because they are covered by the random test points
        sparseMapSmall = healsparse.HealSparseMap.read(os.path.join(self.test_dir, 'healsparse_map_recarray_outoforder.fits'), pixels=[0, 1, 3179])

        ipnestCov = np.right_shift(ipnest, sparseMapSmall._bitShift)
        testValuesSmallCol1 = testMapCol1[ipnest]
        testValuesSmallCol2 = testMapCol2[ipnest]
        outsideSmall, = np.where((ipnestCov != 0) & (ipnestCov != 1) & (ipnestCov != 3179))
        testValuesSmallCol1[outsideSmall] = hp.UNSEEN
        testValuesSmallCol2[outsideSmall] = hp.UNSEEN

        testing.assert_almost_equal(sparseMapSmall.getValuePixel(ipnest)['col1'],
                                    testValuesSmallCol1)
        testing.assert_almost_equal(sparseMapSmall.getValuePixel(ipnest)['col2'],
                                    testValuesSmallCol2)


    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)

if __name__=='__main__':
    unittest.main()

