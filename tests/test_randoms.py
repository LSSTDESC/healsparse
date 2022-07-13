import unittest
import numpy as np
import hpgeom as hpg

import healsparse


class UniformRandomTestCase(unittest.TestCase):
    def test_uniform_randoms(self):
        """
        Test the uniform randoms
        """

        rng = np.random.RandomState(12345)

        nside_coverage = 32
        nside_map = 128

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype=np.float32)

        ra, dec = hpg.pixel_to_angle(nside_map, np.arange(hpg.nside_to_npixel(nside_map)))
        # Arbitrarily chosen range
        gd_pix, = np.where((ra > 100.0) & (ra < 180.0) & (dec > 5.0) & (dec < 30.0))
        sparse_map.update_values_pix(gd_pix, np.zeros(gd_pix.size, dtype=np.float32))

        n_random = 100000
        ra_rand, dec_rand = healsparse.make_uniform_randoms(sparse_map, n_random, rng=rng)

        self.assertEqual(ra_rand.size, n_random)
        self.assertEqual(dec_rand.size, n_random)

        # We have to have a cushion here because we have a finite pixel
        # size
        self.assertTrue(ra_rand.min() > (100.0 - 0.5))
        self.assertTrue(ra_rand.max() < (180.0 + 0.5))
        self.assertTrue(dec_rand.min() > (5.0 - 0.5))
        self.assertTrue(dec_rand.max() < (30.0 + 0.5))

        # And these are all in the map
        self.assertTrue(np.all(sparse_map.get_values_pos(ra_rand, dec_rand, lonlat=True) > hpg.UNSEEN))

    def test_uniform_randoms_cross_ra0(self):
        """
        Test the uniform randoms, crossing ra = 0
        """

        rng = np.random.RandomState(12345)

        nside_coverage = 32
        nside_map = 128

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype=np.float32)

        ra, dec = hpg.pixel_to_angle(nside_map, np.arange(hpg.nside_to_npixel(nside_map)))
        # Arbitrarily chosen range
        gd_pix, = np.where(((ra > 300.0) | (ra < 80.0)) & (dec > -20.0) & (dec < -5.0))
        sparse_map.update_values_pix(gd_pix, np.zeros(gd_pix.size, dtype=np.float32))

        n_random = 100000
        ra_rand, dec_rand = healsparse.make_uniform_randoms(sparse_map, n_random, rng=rng)

        self.assertEqual(ra_rand.size, n_random)
        self.assertEqual(dec_rand.size, n_random)

        self.assertTrue(ra_rand.min() > 0.0)
        self.assertTrue(ra_rand.max() < 360.0)
        self.assertTrue(dec_rand.min() > (-20.0 - 0.5))
        self.assertTrue(dec_rand.max() < (-5.0 + 0.5))

        # And these are all in the map
        self.assertTrue(np.all(sparse_map.get_values_pos(ra_rand, dec_rand, lonlat=True) > hpg.UNSEEN))

    def test_uniform_randoms_fast(self):
        """
        Test the fast uniform randoms
        """

        rng = np.random.RandomState(12345)

        nside_coverage = 32
        nside_map = 128

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype=np.float32)

        ra, dec = hpg.pixel_to_angle(nside_map, np.arange(hpg.nside_to_npixel(nside_map)))
        # Arbitrarily chosen range
        gd_pix, = np.where((ra > 100.0) & (ra < 180.0) & (dec > 5.0) & (dec < 30.0))
        sparse_map.update_values_pix(gd_pix, np.zeros(gd_pix.size, dtype=np.float32))

        n_random = 100000
        ra_rand, dec_rand = healsparse.make_uniform_randoms_fast(sparse_map, n_random, rng=rng)

        self.assertEqual(ra_rand.size, n_random)
        self.assertEqual(dec_rand.size, n_random)

        # We have to have a cushion here because we have a finite pixel
        # size
        self.assertTrue(ra_rand.min() > (100.0 - 0.5))
        self.assertTrue(ra_rand.max() < (180.0 + 0.5))
        self.assertTrue(dec_rand.min() > (5.0 - 0.5))
        self.assertTrue(dec_rand.max() < (30.0 + 0.5))

        # And these are all in the map
        self.assertTrue(np.all(sparse_map.get_values_pos(ra_rand, dec_rand, lonlat=True) > hpg.UNSEEN))


if __name__ == '__main__':
    unittest.main()
