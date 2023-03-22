import unittest
import numpy.testing as testing
import numpy as np
import hpgeom as hpg
from numpy import random

import healsparse


class BuildMapsTestCase(unittest.TestCase):
    def test_build_maps_single(self):
        """
        Test building a map for a single-value field
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        n_rand = 1000
        ra = np.random.random(n_rand) * 360.0
        dec = np.random.random(n_rand) * 180.0 - 90.0

        # Create an empty map
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.float64)

        # Look up all the values, make sure they're all UNSEEN
        testing.assert_almost_equal(sparse_map.get_values_pos(ra, dec, lonlat=True), hpg.UNSEEN)

        # Fail to append because of wrong dtype
        pixel = np.arange(4000, 20000)
        values = np.ones_like(pixel, dtype=np.float32)

        self.assertRaises(ValueError, sparse_map.update_values_pix, pixel, values)

        # Append a bunch of pixels
        values = np.ones_like(pixel, dtype=np.float64)
        sparse_map.update_values_pix(pixel, values)

        # Make a healpix map for comparison
        hpmap = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        hpmap[pixel] = values
        ipnest_test = hpg.angle_to_pixel(nside_map, ra, dec, nest=True)
        testing.assert_almost_equal(sparse_map.get_values_pos(ra, dec, lonlat=True), hpmap[ipnest_test])

        # Replace the pixels
        values += 1
        sparse_map.update_values_pix(pixel, values)
        hpmap[pixel] = values
        testing.assert_almost_equal(sparse_map.get_values_pos(ra, dec, lonlat=True), hpmap[ipnest_test])

        # Replace and append more pixels
        # Note that these are lower-number pixels, so the map is out of order
        pixel2 = np.arange(3000) + 2000
        values2 = np.ones_like(pixel2, dtype=np.float64)
        sparse_map.update_values_pix(pixel2, values2)
        hpmap[pixel2] = values2
        testing.assert_almost_equal(sparse_map.get_values_pos(ra, dec, lonlat=True), hpmap[ipnest_test])

        # Test making empty maps
        sparse_map2 = healsparse.HealSparseMap.make_empty_like(sparse_map)
        self.assertEqual(sparse_map2.nside_coverage, sparse_map.nside_coverage)
        self.assertEqual(sparse_map2.nside_sparse, sparse_map.nside_sparse)
        self.assertEqual(sparse_map2.dtype, sparse_map.dtype)
        self.assertEqual(sparse_map2.sentinel, sparse_map.sentinel)

        sparse_map2b = healsparse.HealSparseMap.make_empty_like(sparse_map, cov_pixels=[0, 2])
        self.assertEqual(sparse_map2b.nside_coverage, sparse_map.nside_coverage)
        self.assertEqual(sparse_map2b.nside_sparse, sparse_map.nside_sparse)
        self.assertEqual(sparse_map2b.dtype, sparse_map.dtype)
        self.assertEqual(sparse_map2b.sentinel, sparse_map.sentinel)
        self.assertEqual(len(sparse_map2b._sparse_map),
                         sparse_map2._cov_map.nfine_per_cov*3)
        testing.assert_array_equal(sparse_map2b._sparse_map, sparse_map.sentinel)

        sparse_map2 = healsparse.HealSparseMap.make_empty_like(sparse_map, nside_coverage=16)
        self.assertEqual(sparse_map2.nside_coverage, 16)
        self.assertEqual(sparse_map2.nside_sparse, sparse_map.nside_sparse)
        self.assertEqual(sparse_map2.dtype, sparse_map.dtype)
        self.assertEqual(sparse_map2.sentinel, sparse_map.sentinel)

        sparse_map2 = healsparse.HealSparseMap.make_empty_like(sparse_map, nside_sparse=128)
        self.assertEqual(sparse_map2.nside_coverage, sparse_map.nside_coverage)
        self.assertEqual(sparse_map2.nside_sparse, 128)
        self.assertEqual(sparse_map2.dtype, sparse_map.dtype)
        self.assertEqual(sparse_map2.sentinel, sparse_map.sentinel)

        sparse_map2 = healsparse.HealSparseMap.make_empty_like(sparse_map, dtype=np.int32, sentinel=0)
        self.assertEqual(sparse_map2.nside_coverage, sparse_map.nside_coverage)
        self.assertEqual(sparse_map2.nside_sparse, sparse_map.nside_sparse)
        self.assertEqual(sparse_map2.dtype, np.int32)

    def test_build_maps_recarray(self):
        """
        Testing building a map for a recarray
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        n_rand = 1000
        ra = np.random.random(n_rand) * 360.0
        dec = np.random.random(n_rand) * 180.0 - 90.0

        # Create an empty map
        dtype = [('col1', 'f4'), ('col2', 'f8')]
        self.assertRaises(RuntimeError, healsparse.HealSparseMap.make_empty, nside_coverage,
                          nside_map, dtype)

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype, primary='col1')

        # Look up all the values, make sure they're all UNSEEN
        testing.assert_almost_equal(sparse_map.get_values_pos(ra, dec, lonlat=True)['col1'], hpg.UNSEEN)
        testing.assert_almost_equal(sparse_map.get_values_pos(ra, dec, lonlat=True)['col2'], hpg.UNSEEN)

        pixel = np.arange(4000, 20000)
        values = np.zeros_like(pixel, dtype=dtype)
        values['col1'] = 1.0
        values['col2'] = 2.0
        sparse_map.update_values_pix(pixel, values)

        # Make healpix maps for comparison
        hpmapCol1 = np.zeros(hpg.nside_to_npixel(nside_map), dtype=np.float32) + hpg.UNSEEN
        hpmapCol2 = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        hpmapCol1[pixel] = values['col1']
        hpmapCol2[pixel] = values['col2']
        ipnest_test = hpg.angle_to_pixel(nside_map, ra, dec, nest=True)
        testing.assert_almost_equal(sparse_map.get_values_pos(ra, dec, lonlat=True)['col1'],
                                    hpmapCol1[ipnest_test])
        testing.assert_almost_equal(sparse_map.get_values_pos(ra, dec, lonlat=True)['col2'],
                                    hpmapCol2[ipnest_test])

        # Replace the pixels
        values['col1'] += 1
        values['col2'] += 1
        sparse_map.update_values_pix(pixel, values)
        hpmapCol1[pixel] = values['col1']
        hpmapCol2[pixel] = values['col2']
        testing.assert_almost_equal(sparse_map.get_values_pos(ra, dec, lonlat=True)['col1'],
                                    hpmapCol1[ipnest_test])
        testing.assert_almost_equal(sparse_map.get_values_pos(ra, dec, lonlat=True)['col2'],
                                    hpmapCol2[ipnest_test])

        # Replace and append more pixels
        # Note that these are lower-number pixels, so the map is out of order
        pixel2 = np.arange(3000) + 2000
        values2 = np.zeros_like(pixel2, dtype=dtype)
        values2['col1'] = 1.0
        values2['col2'] = 2.0
        sparse_map.update_values_pix(pixel2, values2)
        hpmapCol1[pixel2] = values2['col1']
        hpmapCol2[pixel2] = values2['col2']
        testing.assert_almost_equal(sparse_map.get_values_pos(ra, dec, lonlat=True)['col1'],
                                    hpmapCol1[ipnest_test])
        testing.assert_almost_equal(sparse_map.get_values_pos(ra, dec, lonlat=True)['col2'],
                                    hpmapCol2[ipnest_test])

        # Test making empty maps
        sparse_map2 = healsparse.HealSparseMap.make_empty_like(sparse_map)
        self.assertEqual(sparse_map2.nside_coverage, sparse_map.nside_coverage)
        self.assertEqual(sparse_map2.nside_sparse, sparse_map.nside_sparse)
        self.assertEqual(sparse_map2.dtype, sparse_map.dtype)
        self.assertEqual(sparse_map2.sentinel, sparse_map.sentinel)

        sparse_map2b = healsparse.HealSparseMap.make_empty_like(sparse_map, cov_pixels=[0, 2])
        self.assertEqual(sparse_map2b.nside_coverage, sparse_map.nside_coverage)
        self.assertEqual(sparse_map2b.nside_sparse, sparse_map.nside_sparse)
        self.assertEqual(sparse_map2b.dtype, sparse_map.dtype)
        self.assertEqual(sparse_map2b.sentinel, sparse_map.sentinel)
        self.assertEqual(len(sparse_map2b._sparse_map),
                         sparse_map2._cov_map.nfine_per_cov*3)
        testing.assert_array_equal(sparse_map2b._sparse_map['col1'], sparse_map.sentinel)
        testing.assert_array_equal(sparse_map2b._sparse_map['col2'], hpg.UNSEEN)


if __name__ == '__main__':
    unittest.main()
