from __future__ import division, absolute_import, print_function

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
from numpy import random
import os
import healsparse

class GenerateHealpixMapTestCase(unittest.TestCase):
    def test_generateHealpixMap_single(self):
        """
        Test the generation of a healpix map from a sparse map for a single-value field
        """
        random.seed(seed=12345)

        nsideCoverage = 32
        nsideMap = 64

        nRand = 1000
        ra = np.random.random(nRand) * 360.0
        dec = np.random.random(nRand) * 180.0 - 90.0
        value = np.random.random(nRand)

        # Create a HEALPix map
        healpixMap = np.zeros(hp.nside2npix(nsideMap), dtype=np.float) + hp.UNSEEN
        idx = hp.ang2pix(nsideMap, np.pi/2-np.radians(dec), np.radians(ra), nest=True)
        healpixMap[idx]=value
        # Create a HealSparseMap
        sparseMap = healsparse.HealSparseMap(nsideCoverage=nsideCoverage, healpixMap=healpixMap)
        hp_out = sparseMap.generateHealpixMap(nside=nsideMap)
        testing.assert_almost_equal(healpixMap, hp_out)

        # Now check that it works specifying a different resolution
        nsideMap2 = 32
        hp_out = sparseMap.generateHealpixMap(nside=nsideMap2)
        # Let's compare with the original downgraded
        healpixMap = hp.ud_grade(healpixMap, nside_out=nsideMap2, order_in='NESTED', order_out='NESTED')
        testing.assert_almost_equal(healpixMap, hp_out)

    def test_generateHealpixMap_recarray(self):
        """
        Testing the generation of a healpix map from recarray healsparsemap
        we also test the pixel and position lookup
        """
        random.seed(seed=12345)

        nsideCoverage = 32
        nsideMap = 64

        nRand = 1000
        ra = np.random.random(nRand) * 360.0
        dec = np.random.random(nRand) * 180.0 - 90.0
        value = np.random.random(nRand)
        # Create empty healpix map
        healpixMap = np.zeros(hp.nside2npix(nsideMap), dtype='f4')+hp.UNSEEN
        healpixMap2 = np.zeros(hp.nside2npix(nsideMap), dtype='f8')+hp.UNSEEN
        healpixMap[hp.ang2pix(nsideMap, np.pi/2-np.radians(dec), np.radians(ra), nest=True)] = value
        healpixMap2[hp.ang2pix(nsideMap, np.pi/2-np.radians(dec), np.radians(ra), nest=True)] = value
        # Create an empty map
        dtype = [('col1', 'f4'), ('col2', 'f8')]

        self.assertRaises(RuntimeError, healsparse.HealSparseMap.makeEmpty, nsideCoverage, nsideMap, dtype)
        # Generate empty map that will be updated
        sparseMap = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, dtype, primary='col1')
        # Generate auxiliary map to get the correct coverage index map so we can lookup the positions
        aux_spMap = healsparse.HealSparseMap(nsideCoverage=nsideCoverage, healpixMap=healpixMap)
        pixel = hp.ang2pix(nsideMap, np.radians(90-dec), np.radians(ra), nest=True)
        values = np.zeros_like(pixel, dtype=dtype)
        values['col1'] = value
        values['col2'] = value
        sparseMap.updateValues(pixel, values) # Update values works with the HEALPix-like indexing scheme
        hp_out1 = sparseMap.generateHealpixMap(nside=nsideMap, key='col1')
        hp_out2 = sparseMap.generateHealpixMap(nside=nsideMap, key='col2')
        testing.assert_almost_equal(healpixMap, hp_out1)
        testing.assert_almost_equal(healpixMap2, hp_out2)

    def test_generateHealpixMap_int(self):
        """
        Testing the generation of a healpix map from an integer map
        """
        random.seed(seed=12345)

        nsideCoverage = 32
        nsideMap = 64

        sparseMap = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, np.int64)
        pixel = np.arange(4000, 20000)
        pixel = np.delete(pixel, 15000)
        # Get a random list of integers
        values = np.random.poisson(size=pixel.size, lam=10)
        sparseMap.updateValues(pixel, values)

        hpmap = sparseMap.generateHealpixMap()

        ok, = np.where(hpmap > hp.UNSEEN)

        testing.assert_almost_equal(hpmap[ok], sparseMap.getValuePixel(ok).astype(np.float64))




if __name__=='__main__':
    unittest.main()
