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

class BuildMapsTestCase(unittest.TestCase):
    def test_buildMapsSingle(self):
        """
        Test building a map for a single-value field
        """
        random.seed(seed=12345)

        nsideCoverage = 32
        nsideMap = 64

        nRand = 1000
        ra = np.random.random(nRand) * 360.0
        dec = np.random.random(nRand) * 180.0 - 90.0

        # Create an empty map
        sparseMap = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, np.float64)

        # Look up all the values, make sure they're all UNSEEN
        testing.assert_almost_equal(sparseMap.getValueRaDec(ra, dec), hp.UNSEEN)

        # Fail to append because of wrong dtype
        pixel = np.arange(4000, 20000)
        values = np.ones_like(pixel, dtype=np.float32)

        self.assertRaises(RuntimeError, sparseMap.updateValues, pixel, values)

        # Append a bunch of pixels
        values = np.ones_like(pixel, dtype=np.float64)
        sparseMap.updateValues(pixel, values)

        # Make a healpix map for comparison
        hpmap = np.zeros(hp.nside2npix(nsideMap)) + hp.UNSEEN
        hpmap[pixel] = values
        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        ipnestTest = hp.ang2pix(nsideMap, theta, phi, nest=True)
        testing.assert_almost_equal(sparseMap.getValueRaDec(ra, dec), hpmap[ipnestTest])

        # Replace the pixels
        values += 1
        sparseMap.updateValues(pixel, values)
        hpmap[pixel] = values
        testing.assert_almost_equal(sparseMap.getValueRaDec(ra, dec), hpmap[ipnestTest])

        # Replace and append more pixels
        # Note that these are lower-number pixels, so the map is out of order
        pixel2 = np.arange(3000) + 2000
        values2 = np.ones_like(pixel2, dtype=np.float64)
        sparseMap.updateValues(pixel2, values2)
        hpmap[pixel2] = values2
        testing.assert_almost_equal(sparseMap.getValueRaDec(ra, dec), hpmap[ipnestTest])

    def test_buildMapsRecarray(self):
        """
        Testing building a map for a recarray
        """
        random.seed(seed=12345)

        nsideCoverage = 32
        nsideMap = 64

        nRand = 1000
        ra = np.random.random(nRand) * 360.0
        dec = np.random.random(nRand) * 180.0 - 90.0

        # Create an empty map
        dtype = [('col1', 'f4'), ('col2', 'f8')]
        self.assertRaises(RuntimeError, healsparse.HealSparseMap.makeEmpty, nsideCoverage, nsideMap, dtype)

        sparseMap = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, dtype, primary='col1')

        # Look up all the values, make sure they're all UNSEEN
        testing.assert_almost_equal(sparseMap.getValueRaDec(ra, dec)['col1'], hp.UNSEEN)
        testing.assert_almost_equal(sparseMap.getValueRaDec(ra, dec)['col2'], hp.UNSEEN)

        pixel = np.arange(4000, 20000)
        values = np.zeros_like(pixel, dtype=dtype)
        values['col1'] = 1.0
        values['col2'] = 2.0
        sparseMap.updateValues(pixel, values)

        # Make healpix maps for comparison
        hpmapCol1 = np.zeros(hp.nside2npix(nsideMap), dtype=np.float32) + hp.UNSEEN
        hpmapCol2 = np.zeros(hp.nside2npix(nsideMap)) + hp.UNSEEN
        hpmapCol1[pixel] = values['col1']
        hpmapCol2[pixel] = values['col2']
        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        ipnestTest = hp.ang2pix(nsideMap, theta, phi, nest=True)
        testing.assert_almost_equal(sparseMap.getValueRaDec(ra, dec)['col1'],
                                    hpmapCol1[ipnestTest])
        testing.assert_almost_equal(sparseMap.getValueRaDec(ra, dec)['col2'],
                                    hpmapCol2[ipnestTest])

        # Replace the pixels
        values['col1'] += 1
        values['col2'] += 1
        sparseMap.updateValues(pixel, values)
        hpmapCol1[pixel] = values['col1']
        hpmapCol2[pixel] = values['col2']
        testing.assert_almost_equal(sparseMap.getValueRaDec(ra, dec)['col1'],
                                    hpmapCol1[ipnestTest])
        testing.assert_almost_equal(sparseMap.getValueRaDec(ra, dec)['col2'],
                                    hpmapCol2[ipnestTest])

        # Replace and append more pixels
        # Note that these are lower-number pixels, so the map is out of order
        pixel2 = np.arange(3000) + 2000
        values2 = np.zeros_like(pixel2, dtype=dtype)
        values2['col1'] = 1.0
        values2['col2'] = 2.0
        sparseMap.updateValues(pixel2, values2)
        hpmapCol1[pixel2] = values2['col1']
        hpmapCol2[pixel2] = values2['col2']
        testing.assert_almost_equal(sparseMap.getValueRaDec(ra, dec)['col1'],
                                    hpmapCol1[ipnestTest])
        testing.assert_almost_equal(sparseMap.getValueRaDec(ra, dec)['col2'],
                                    hpmapCol2[ipnestTest])






if __name__=='__main__':
    unittest.main()
