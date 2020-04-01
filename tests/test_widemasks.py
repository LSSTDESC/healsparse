import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
from numpy import random
import tempfile
import os
import shutil

import healsparse


class WideMasksTestCase(unittest.TestCase):
    def test_make_wide_mask_map(self):
        """
        Test making a wide mask map.
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        n_rand = 1000
        ra = np.random.random(n_rand) * 360.0
        dec = np.random.random(n_rand) * 180.0 - 90.0

        # Test expected errors on creating maps

        # Create empty maps to test bit width
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         np.uint64, wide_mask_maxbits=50)

        self.assertTrue(sparse_map.is_wide_mask_map)
        self.assertEqual(sparse_map._wide_mask_maxbits, 64)
        self.assertEqual(sparse_map._sparse_map.shape, (4, 1))
        self.assertEqual(sparse_map._sentinel, 0)

        # Set bits and retrieve them
        pixel = np.arange(4000, 20000)
        sparse_map.set_bits_pix(pixel, [5])
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [5]), True)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [10]), False)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [5, 10]), True)

        pospix = hp.ang2pix(nside_map, ra, dec, lonlat=True, nest=True)
        inds = np.searchsorted(pixel, pospix)
        b, = np.where((inds > 0) & (inds < pixel.size))
        comp_arr = np.zeros(pospix.size, dtype=np.bool)
        comp_arr[b] = True
        testing.assert_array_equal(sparse_map.check_bits_pos(ra, dec, [5], lonlat=True), comp_arr)

        sparse_map.clear_bits_pix(pixel, [5])
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [5]), False)

        sparse_map.set_bits_pix(pixel, [5, 10])
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [5]), True)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [10]), True)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [15]), False)

        # This just makes sure that the size is correct
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         np.uint64, wide_mask_maxbits=64)
        self.assertEqual(sparse_map._wide_mask_maxbits, 64)

        # And now a double-wide to test
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         np.uint64, wide_mask_maxbits=65)
        self.assertEqual(sparse_map._wide_mask_maxbits, 128)

        sparse_map.set_bits_pix(pixel, [70])
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [70]), True)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [10]), False)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [75]), False)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [10, 70]), True)

        sparse_map.set_bits_pix(pixel, [5, 70])
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [5]), True)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [70]), True)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [15]), False)

        # This makes sure the sizes are correct
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         np.uint64, wide_mask_maxbits=128)
        self.assertEqual(sparse_map._wide_mask_maxbits, 128)
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         np.uint64, wide_mask_maxbits=129)
        self.assertEqual(sparse_map._wide_mask_maxbits, 192)

    def test_wide_mask_map_io(self):
        """
        Test i/o with wide mask maps.
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        n_rand = 1000
        ra = np.random.random(n_rand) * 360.0
        dec = np.random.random(n_rand) * 180.0 - 90.0

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        # Test with single-wide
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         np.uint64, wide_mask_maxbits=50)
        pixel = np.arange(4000, 20000)
        sparse_map.set_bits_pix(pixel, [5])

        fname = os.path.join(self.test_dir, 'healsparse_map.fits')
        sparse_map.write(fname, clobber=True)
        sparse_map_in = healsparse.HealSparseMap.read(fname)

        self.assertTrue(sparse_map_in.is_wide_mask_map)
        self.assertEqual(sparse_map_in._wide_mask_maxbits, 64)
        self.assertEqual(sparse_map_in._sparse_map.shape[1], 1)
        self.assertEqual(sparse_map_in._wide_mask_width, 1)
        self.assertEqual(sparse_map_in._sentinel, 0)

        testing.assert_array_equal(sparse_map_in.check_bits_pix(pixel, [5]), True)
        testing.assert_array_equal(sparse_map_in.check_bits_pix(pixel, [10]), False)
        testing.assert_array_equal(sparse_map_in.check_bits_pix(pixel, [5, 10]), True)

        pospix = hp.ang2pix(nside_map, ra, dec, lonlat=True, nest=True)
        inds = np.searchsorted(pixel, pospix)
        b, = np.where((inds > 0) & (inds < pixel.size))
        comp_arr = np.zeros(pospix.size, dtype=np.bool)
        comp_arr[b] = True
        testing.assert_array_equal(sparse_map_in.check_bits_pos(ra, dec, [5], lonlat=True), comp_arr)

        # Test with double-wide
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         np.uint64, wide_mask_maxbits=100)
        pixel = np.arange(4000, 20000)
        sparse_map.set_bits_pix(pixel, [5, 70])

        fname = os.path.join(self.test_dir, 'healsparse_map.fits')
        sparse_map.write(fname, clobber=True)
        sparse_map_in = healsparse.HealSparseMap.read(fname)

        self.assertTrue(sparse_map_in.is_wide_mask_map)
        self.assertEqual(sparse_map_in._wide_mask_maxbits, 128)
        self.assertEqual(sparse_map_in._sparse_map.shape[1], 2)
        self.assertEqual(sparse_map_in._wide_mask_width, 2)
        self.assertEqual(sparse_map_in._sentinel, 0)

        testing.assert_array_equal(sparse_map_in.check_bits_pix(pixel, [5]), True)
        testing.assert_array_equal(sparse_map_in.check_bits_pix(pixel, [70]), True)
        testing.assert_array_equal(sparse_map_in.check_bits_pix(pixel, [10]), False)
        testing.assert_array_equal(sparse_map_in.check_bits_pix(pixel, [75]), False)

    def test_wide_mask_or(self):
        """
        Test wide mask oring
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        sparse_map1 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.uint64,
                                                          wide_mask_maxbits=100)
        pixel1 = np.arange(4000, 20000)
        pixel1 = np.delete(pixel1, 15000)
        values1 = np.random.poisson(size=(pixel1.size, sparse_map1._wide_mask_width), lam=2).astype(np.uint64)
        sparse_map1.update_values_pix(pixel1, values1)

        sparse_map2 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.uint64,
                                                          wide_mask_maxbits=100)
        pixel2 = np.arange(15000, 25000)
        values2 = np.random.poisson(size=(pixel2.size, sparse_map2._wide_mask_width), lam=2).astype(np.uint64)
        sparse_map2.update_values_pix(pixel2, values2)

        or_map_intersection = healsparse.or_intersection([sparse_map1, sparse_map2])

        all_pixels = np.arange(hp.nside2npix(nside_map))
        arr1 = sparse_map1.get_values_pix(all_pixels)
        arr2 = sparse_map2.get_values_pix(all_pixels)
        gd, = np.where((arr1.sum(axis=1) > 0) & (arr2.sum(axis=1) > 0))
        for i in range(2):
            test_or_intersection = np.zeros_like(arr1[:, 0])
            test_or_intersection[gd] = arr1[gd, i] | arr2[gd, i]
            testing.assert_equal(or_map_intersection.get_values_pix(all_pixels)[:, i],
                                 test_or_intersection)

        or_map_union = healsparse.or_union([sparse_map1, sparse_map2])

        gd, = np.where((arr1.sum(axis=1) > 0) | (arr2.sum(axis=1) > 0))
        for i in range(2):
            test_or_union = np.zeros_like(arr1[:, 0])
            test_or_union[gd] = arr1[gd, i] | arr2[gd, i]
            testing.assert_equal(or_map_union.get_values_pix(all_pixels)[:, i],
                                 test_or_union)

    def test_wide_mask_and(self):
        """
        Test wide mask anding
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        sparse_map1 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.uint64,
                                                          wide_mask_maxbits=100)
        pixel1 = np.arange(4000, 20000)
        pixel1 = np.delete(pixel1, 15000)
        values1 = np.random.poisson(size=(pixel1.size, sparse_map1._wide_mask_width), lam=2).astype(np.uint64)
        sparse_map1.update_values_pix(pixel1, values1)

        sparse_map2 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.uint64,
                                                          wide_mask_maxbits=100)
        pixel2 = np.arange(15000, 25000)
        values2 = np.random.poisson(size=(pixel2.size, sparse_map2._wide_mask_width), lam=2).astype(np.uint64)
        sparse_map2.update_values_pix(pixel2, values2)

        and_map_intersection = healsparse.and_intersection([sparse_map1, sparse_map2])

        all_pixels = np.arange(hp.nside2npix(nside_map))
        arr1 = sparse_map1.get_values_pix(all_pixels)
        arr2 = sparse_map2.get_values_pix(all_pixels)
        gd, = np.where((arr1.sum(axis=1) > 0) & (arr2.sum(axis=1) > 0))
        for i in range(2):
            test_and_intersection = np.zeros_like(arr1[:, 0])
            test_and_intersection[gd] = arr1[gd, i] & arr2[gd, i]
            testing.assert_equal(and_map_intersection.get_values_pix(all_pixels)[:, i],
                                 test_and_intersection)

        and_map_union = healsparse.and_union([sparse_map1, sparse_map2])

        # Brute-force this.  Where there is a number in arr1 but not in arr2, use arr1.
        # Where there is a number in arr2 but not in arr1, use arr2
        # And where there are numbers in each, and them for each bit field.
        gd, = np.where((arr1.sum(axis=1) > 0) | (arr2.sum(axis=1) > 0))
        test_and_union = np.zeros_like(arr1)
        u1, = np.where(arr1[gd, :].sum(axis=1) == 0)
        test_and_union[gd[u1], :] = arr2[gd[u1], :]
        u2, = np.where(arr2[gd, :].sum(axis=1) == 0)
        test_and_union[gd[u2], :] = arr1[gd[u2], :]
        u3, = np.where((arr1[gd, :].sum(axis=1) > 0) & (arr2[gd, :].sum(axis=1) > 0))
        test_and_union[gd[u3], 0] = (arr1[gd[u3], 0] & arr2[gd[u3], 0])
        test_and_union[gd[u3], 1] = (arr1[gd[u3], 1] & arr2[gd[u3], 1])
        testing.assert_equal(and_map_union.get_values_pix(all_pixels),
                             test_and_union)

    def test_wide_mask_xor(self):
        """
        Test wide mask xoring
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        sparse_map1 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.uint64,
                                                          wide_mask_maxbits=100)
        pixel1 = np.arange(4000, 20000)
        pixel1 = np.delete(pixel1, 15000)
        values1 = np.random.poisson(size=(pixel1.size, sparse_map1._wide_mask_width), lam=2).astype(np.uint64)
        sparse_map1.update_values_pix(pixel1, values1)

        sparse_map2 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.uint64,
                                                          wide_mask_maxbits=100)
        pixel2 = np.arange(15000, 25000)
        values2 = np.random.poisson(size=(pixel2.size, sparse_map2._wide_mask_width), lam=2).astype(np.uint64)
        sparse_map2.update_values_pix(pixel2, values2)

        xor_map_intersection = healsparse.xor_intersection([sparse_map1, sparse_map2])

        all_pixels = np.arange(hp.nside2npix(nside_map))
        arr1 = sparse_map1.get_values_pix(all_pixels)
        arr2 = sparse_map2.get_values_pix(all_pixels)
        gd, = np.where((arr1.sum(axis=1) > 0) & (arr2.sum(axis=1) > 0))
        for i in range(2):
            test_xor_intersection = np.zeros_like(arr1[:, 0])
            test_xor_intersection[gd] = arr1[gd, i] ^ arr2[gd, i]
            testing.assert_equal(xor_map_intersection.get_values_pix(all_pixels)[:, i],
                                 test_xor_intersection)

        xor_map_union = healsparse.xor_union([sparse_map1, sparse_map2])

        gd, = np.where((arr1.sum(axis=1) > 0) | (arr2.sum(axis=1) > 0))
        for i in range(2):
            test_xor_union = np.zeros_like(arr1[:, 0])
            test_xor_union[gd] = arr1[gd, i] ^ arr2[gd, i]
            testing.assert_equal(xor_map_union.get_values_pix(all_pixels)[:, i],
                                 test_xor_union)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__ == '__main__':
    unittest.main()
