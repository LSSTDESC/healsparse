from __future__ import division, absolute_import, print_function

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
from numpy import random

import healsparse

class UniformRandomTestCase(unittest.TestCase):
    def test_uniform_randoms(self):
        """
        Test the uniform randoms
        """

        rng = np.random.RandomState(12345)

        nsideCoverage = 32
        nsideMap = 128

        sparseMap = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, dtype=np.float32)

        theta, phi = hp.pix2ang(nsideMap, np.arange(hp.nside2npix(nsideMap)), nest=True)
        ra = np.degrees(phi)
        dec = 90.0 - np.degrees(theta)
        # Arbitrarily chosen range
        gdPix, = np.where((ra > 100.0) & (ra < 180.0) & (dec > 5.0) & (dec < 30.0))
        sparseMap.updateValues(gdPix, np.zeros(gdPix.size, dtype=np.float32))

        nRandom = 100000
        raRand, decRand = healsparse.makeUniformRandoms(sparseMap, nRandom, rng=rng)

        self.assertEqual(raRand.size, nRandom)
        self.assertEqual(decRand.size, nRandom)

        # We have to have a cushion here because we have a finite pixel
        # size
        self.assertTrue(raRand.min() > (100.0 - 0.5))
        self.assertTrue(raRand.max() < (180.0 + 0.5))
        self.assertTrue(decRand.min() > (5.0 - 0.5))
        self.assertTrue(decRand.max() < (30.0 + 0.5))

        # And these are all in the map
        self.assertTrue(np.all(sparseMap.getValueRaDec(raRand, decRand) > hp.UNSEEN))

    def test_uniform_randoms_cross_ra0(self):
        """
        Test the uniform randoms, crossing ra = 0
        """

        rng = np.random.RandomState(12345)

        nsideCoverage = 32
        nsideMap = 128

        sparseMap = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, dtype=np.float32)

        theta, phi = hp.pix2ang(nsideMap, np.arange(hp.nside2npix(nsideMap)), nest=True)
        ra = np.degrees(phi)
        dec = 90.0 - np.degrees(theta)
        # Arbitrarily chosen range
        gdPix, = np.where(((ra > 300.0) | (ra < 80.0)) & (dec > -20.0) & (dec < -5.0))
        sparseMap.updateValues(gdPix, np.zeros(gdPix.size, dtype=np.float32))

        nRandom = 100000
        raRand, decRand = healsparse.makeUniformRandoms(sparseMap, nRandom, rng=rng)

        self.assertEqual(raRand.size, nRandom)
        self.assertEqual(decRand.size, nRandom)

        self.assertTrue(raRand.min() > 0.0)
        self.assertTrue(raRand.max() < 360.0)
        self.assertTrue(decRand.min() > (-20.0 - 0.5))
        self.assertTrue(decRand.max() < (-5.0 + 0.5))

        # And these are all in the map
        self.assertTrue(np.all(sparseMap.getValueRaDec(raRand, decRand) > hp.UNSEEN))

    def test_uniform_randoms_fast(self):
        """
        Test the fast uniform randoms
        """

        rng = np.random.RandomState(12345)

        nsideCoverage = 32
        nsideMap = 128

        sparseMap = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, dtype=np.float32)

        theta, phi = hp.pix2ang(nsideMap, np.arange(hp.nside2npix(nsideMap)), nest=True)
        ra = np.degrees(phi)
        dec = 90.0 - np.degrees(theta)
        # Arbitrarily chosen range
        gdPix, = np.where((ra > 100.0) & (ra < 180.0) & (dec > 5.0) & (dec < 30.0))
        sparseMap.updateValues(gdPix, np.zeros(gdPix.size, dtype=np.float32))

        nRandom = 100000
        raRand, decRand = healsparse.makeUniformRandomsFast(sparseMap, nRandom, rng=rng)

        self.assertEqual(raRand.size, nRandom)
        self.assertEqual(decRand.size, nRandom)

        # We have to have a cushion here because we have a finite pixel
        # size
        self.assertTrue(raRand.min() > (100.0 - 0.5))
        self.assertTrue(raRand.max() < (180.0 + 0.5))
        self.assertTrue(decRand.min() > (5.0 - 0.5))
        self.assertTrue(decRand.max() < (30.0 + 0.5))

        # And these are all in the map
        self.assertTrue(np.all(sparseMap.getValueRaDec(raRand, decRand) > hp.UNSEEN))

if __name__=='__main__':
    unittest.main()
