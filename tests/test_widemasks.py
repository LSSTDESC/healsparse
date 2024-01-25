import unittest
import numpy.testing as testing
import numpy as np
import hpgeom as hpg
from numpy import random
import tempfile
import os
import shutil
import pytest

import healsparse
from healsparse import WIDE_MASK


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
                                                         WIDE_MASK, wide_mask_maxbits=7)

        self.assertTrue(sparse_map.is_wide_mask_map)
        self.assertEqual(sparse_map.wide_mask_maxbits, 8)
        self.assertEqual(sparse_map._sparse_map.shape, (4, 1))
        self.assertEqual(sparse_map.sentinel, 0)

        # Set bits and retrieve them
        pixel = np.arange(4000, 20000)
        sparse_map.set_bits_pix(pixel, [4])
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [4]), True)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [6]), False)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [4, 6]), True)
        testing.assert_array_equal(sparse_map.valid_pixels, pixel)
        testing.assert_equal(sparse_map.n_valid, len(pixel))

        pospix = hpg.angle_to_pixel(nside_map, ra, dec)
        inds = np.searchsorted(pixel, pospix)
        b, = np.where((inds > 0) & (inds < pixel.size))
        comp_arr = np.zeros(pospix.size, dtype=np.bool_)
        comp_arr[b] = True
        testing.assert_array_equal(sparse_map.check_bits_pos(ra, dec, [4], lonlat=True), comp_arr)

        sparse_map.clear_bits_pix(pixel, [4])
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [4]), False)

        sparse_map.set_bits_pix(pixel, [4, 6])
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [4]), True)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [6]), True)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [7]), False)
        testing.assert_array_equal(sparse_map.valid_pixels, pixel)
        testing.assert_equal(sparse_map.n_valid, len(pixel))

        # This just makes sure that the size is correct
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         WIDE_MASK, wide_mask_maxbits=8)
        self.assertEqual(sparse_map.wide_mask_maxbits, 8)

        # And now a double-wide to test
        # Note that 9 will create a 16 bit mask
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         WIDE_MASK, wide_mask_maxbits=9)
        self.assertEqual(sparse_map.wide_mask_maxbits, 16)

        sparse_map.set_bits_pix(pixel, [12])
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [12]), True)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [4]), False)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [15]), False)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [4, 12]), True)

        sparse_map.set_bits_pix(pixel, [2, 3, 5, 15])
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [2]), True)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [3]), True)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [5]), True)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [15]), True)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [14]), False)

        # Clear a bit in the lower field, ensure upper field is untouched.
        sparse_map.clear_bits_pix(pixel, [5])
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [5]), False)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [15]), True)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [14]), False)
        testing.assert_array_equal(sparse_map.valid_pixels, pixel)
        testing.assert_equal(sparse_map.n_valid, len(pixel))

        # Clear multiple bits in the lower field, ensure upper field is untouched.
        sparse_map.clear_bits_pix(pixel, [2, 3])
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [2]), False)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [3]), False)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [15]), True)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [14]), False)
        testing.assert_array_equal(sparse_map.valid_pixels, pixel)
        testing.assert_equal(sparse_map.n_valid, len(pixel))

        # This makes sure the inferred size is correct
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         WIDE_MASK, wide_mask_maxbits=128)
        self.assertEqual(sparse_map.wide_mask_maxbits, 128)

        # And do a triple-wide
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         WIDE_MASK, wide_mask_maxbits=20)
        self.assertEqual(sparse_map.wide_mask_maxbits, 24)

        sparse_map.set_bits_pix(pixel, [5, 10, 20])
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [5]), True)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [10]), True)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [20]), True)
        testing.assert_array_equal(sparse_map.check_bits_pix(pixel, [21]), False)
        testing.assert_array_equal(sparse_map.valid_pixels, pixel)
        testing.assert_equal(sparse_map.n_valid, len(pixel))

    def test_wide_mask_map_fits_io(self):
        """
        Test fits i/o with wide mask maps.
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
                                                         WIDE_MASK, wide_mask_maxbits=8)
        pixel = np.arange(4000, 20000)
        sparse_map.set_bits_pix(pixel, [5])

        fname = os.path.join(self.test_dir, 'healsparse_map.hs')
        sparse_map.write(fname, clobber=True)
        sparse_map_in = healsparse.HealSparseMap.read(fname)

        self.assertTrue(sparse_map_in.is_wide_mask_map)
        self.assertEqual(sparse_map_in.wide_mask_maxbits, 8)
        self.assertEqual(sparse_map_in._sparse_map.shape[1], 1)
        self.assertEqual(sparse_map_in._wide_mask_width, 1)
        self.assertEqual(sparse_map_in.sentinel, 0)

        testing.assert_array_equal(sparse_map_in.check_bits_pix(pixel, [5]), True)
        testing.assert_array_equal(sparse_map_in.check_bits_pix(pixel, [7]), False)
        testing.assert_array_equal(sparse_map_in.check_bits_pix(pixel, [5, 7]), True)

        pospix = hpg.angle_to_pixel(nside_map, ra, dec)
        inds = np.searchsorted(pixel, pospix)
        b, = np.where((inds > 0) & (inds < pixel.size))
        comp_arr = np.zeros(pospix.size, dtype=np.bool_)
        comp_arr[b] = True
        testing.assert_array_equal(sparse_map_in.check_bits_pos(ra, dec, [5], lonlat=True), comp_arr)

        # And read a partial map
        sparse_map_in_partial = healsparse.HealSparseMap.read(fname, pixels=[1000, 1002])

        self.assertTrue(sparse_map_in_partial.is_wide_mask_map)
        self.assertEqual(sparse_map_in_partial.wide_mask_maxbits, 8)
        self.assertEqual(sparse_map_in_partial._sparse_map.shape[1], 1)
        self.assertEqual(sparse_map_in_partial._wide_mask_width, 1)
        self.assertEqual(sparse_map_in_partial.sentinel, 0)

        cov_pixels = sparse_map._cov_map.cov_pixels(pixel)
        pixel_sub = pixel[(cov_pixels == 1000) | (cov_pixels == 1002)]

        testing.assert_array_equal(sparse_map_in_partial.check_bits_pix(pixel_sub, [5]), True)
        testing.assert_array_equal(sparse_map_in_partial.check_bits_pix(pixel_sub, [7]), False)
        testing.assert_array_equal(sparse_map_in_partial.check_bits_pix(pixel_sub, [5, 7]), True)

        # Test with double-wide
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         WIDE_MASK, wide_mask_maxbits=16)
        pixel = np.arange(4000, 20000)
        sparse_map.set_bits_pix(pixel, [5, 10])

        fname = os.path.join(self.test_dir, 'healsparse_map.hs')
        sparse_map.write(fname, clobber=True)
        sparse_map_in = healsparse.HealSparseMap.read(fname)

        self.assertTrue(sparse_map_in.is_wide_mask_map)
        self.assertEqual(sparse_map_in.wide_mask_maxbits, 16)
        self.assertEqual(sparse_map_in._sparse_map.shape[1], 2)
        self.assertEqual(sparse_map_in._wide_mask_width, 2)
        self.assertEqual(sparse_map_in.sentinel, 0)

        testing.assert_array_equal(sparse_map_in.check_bits_pix(pixel, [5]), True)
        testing.assert_array_equal(sparse_map_in.check_bits_pix(pixel, [10]), True)
        testing.assert_array_equal(sparse_map_in.check_bits_pix(pixel, [4]), False)
        testing.assert_array_equal(sparse_map_in.check_bits_pix(pixel, [12]), False)

        # And read a partial double-wide map
        sparse_map_in_partial = healsparse.HealSparseMap.read(fname, pixels=[1000, 1002])

        self.assertTrue(sparse_map_in_partial.is_wide_mask_map)
        self.assertEqual(sparse_map_in_partial.wide_mask_maxbits, 16)
        self.assertEqual(sparse_map_in_partial._sparse_map.shape[1], 2)
        self.assertEqual(sparse_map_in_partial._wide_mask_width, 2)
        self.assertEqual(sparse_map_in_partial.sentinel, 0)

        cov_pixels = sparse_map._cov_map.cov_pixels(pixel)
        pixel_sub = pixel[(cov_pixels == 1000) | (cov_pixels == 1002)]

        testing.assert_array_equal(sparse_map_in_partial.check_bits_pix(pixel_sub, [5]), True)
        testing.assert_array_equal(sparse_map_in_partial.check_bits_pix(pixel_sub, [10]), True)
        testing.assert_array_equal(sparse_map_in_partial.check_bits_pix(pixel_sub, [4]), False)
        testing.assert_array_equal(sparse_map_in_partial.check_bits_pix(pixel_sub, [12]), False)

    @pytest.mark.skipif(not healsparse.parquet_shim.use_pyarrow, reason='Requires pyarrow')
    def test_wide_mask_map_parquet_io(self):
        """
        Test parquet i/o with wide mask maps.
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
                                                         WIDE_MASK, wide_mask_maxbits=8)
        pixel = np.arange(4000, 20000)
        sparse_map.set_bits_pix(pixel, [5])

        fname = os.path.join(self.test_dir, 'healsparse_map.hsparquet')
        sparse_map.write(fname, format='parquet')
        sparse_map_in = healsparse.HealSparseMap.read(fname)

        self.assertTrue(sparse_map_in.is_wide_mask_map)
        self.assertEqual(sparse_map_in.wide_mask_maxbits, 8)
        self.assertEqual(sparse_map_in._sparse_map.shape[1], 1)
        self.assertEqual(sparse_map_in._wide_mask_width, 1)
        self.assertEqual(sparse_map_in.sentinel, 0)

        testing.assert_array_equal(sparse_map_in.check_bits_pix(pixel, [5]), True)
        testing.assert_array_equal(sparse_map_in.check_bits_pix(pixel, [7]), False)
        testing.assert_array_equal(sparse_map_in.check_bits_pix(pixel, [5, 7]), True)

        pospix = hpg.angle_to_pixel(nside_map, ra, dec)
        inds = np.searchsorted(pixel, pospix)
        b, = np.where((inds > 0) & (inds < pixel.size))
        comp_arr = np.zeros(pospix.size, dtype=np.bool_)
        comp_arr[b] = True
        testing.assert_array_equal(sparse_map_in.check_bits_pos(ra, dec, [5], lonlat=True), comp_arr)

        # And read a partial map
        sparse_map_in_partial = healsparse.HealSparseMap.read(fname, pixels=[1000, 1002])

        self.assertTrue(sparse_map_in_partial.is_wide_mask_map)
        self.assertEqual(sparse_map_in_partial.wide_mask_maxbits, 8)
        self.assertEqual(sparse_map_in_partial._sparse_map.shape[1], 1)
        self.assertEqual(sparse_map_in_partial._wide_mask_width, 1)
        self.assertEqual(sparse_map_in_partial.sentinel, 0)

        cov_pixels = sparse_map._cov_map.cov_pixels(pixel)
        pixel_sub = pixel[(cov_pixels == 1000) | (cov_pixels == 1002)]

        testing.assert_array_equal(sparse_map_in_partial.check_bits_pix(pixel_sub, [5]), True)
        testing.assert_array_equal(sparse_map_in_partial.check_bits_pix(pixel_sub, [7]), False)
        testing.assert_array_equal(sparse_map_in_partial.check_bits_pix(pixel_sub, [5, 7]), True)

        # Test with double-wide
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         WIDE_MASK, wide_mask_maxbits=16)
        pixel = np.arange(4000, 20000)
        sparse_map.set_bits_pix(pixel, [5, 10])

        fname = os.path.join(self.test_dir, 'healsparse_map2.hsparquet')
        sparse_map.write(fname, format='parquet')
        sparse_map_in = healsparse.HealSparseMap.read(fname)

        self.assertTrue(sparse_map_in.is_wide_mask_map)
        self.assertEqual(sparse_map_in.wide_mask_maxbits, 16)
        self.assertEqual(sparse_map_in._sparse_map.shape[1], 2)
        self.assertEqual(sparse_map_in._wide_mask_width, 2)
        self.assertEqual(sparse_map_in.sentinel, 0)

        testing.assert_array_equal(sparse_map_in.check_bits_pix(pixel, [5]), True)
        testing.assert_array_equal(sparse_map_in.check_bits_pix(pixel, [10]), True)
        testing.assert_array_equal(sparse_map_in.check_bits_pix(pixel, [4]), False)
        testing.assert_array_equal(sparse_map_in.check_bits_pix(pixel, [12]), False)

        # And read a partial double-wide map
        sparse_map_in_partial = healsparse.HealSparseMap.read(fname, pixels=[1000, 1002])

        self.assertTrue(sparse_map_in_partial.is_wide_mask_map)
        self.assertEqual(sparse_map_in_partial.wide_mask_maxbits, 16)
        self.assertEqual(sparse_map_in_partial._sparse_map.shape[1], 2)
        self.assertEqual(sparse_map_in_partial._wide_mask_width, 2)
        self.assertEqual(sparse_map_in_partial.sentinel, 0)

        cov_pixels = sparse_map._cov_map.cov_pixels(pixel)
        pixel_sub = pixel[(cov_pixels == 1000) | (cov_pixels == 1002)]

        testing.assert_array_equal(sparse_map_in_partial.check_bits_pix(pixel_sub, [5]), True)
        testing.assert_array_equal(sparse_map_in_partial.check_bits_pix(pixel_sub, [10]), True)
        testing.assert_array_equal(sparse_map_in_partial.check_bits_pix(pixel_sub, [4]), False)
        testing.assert_array_equal(sparse_map_in_partial.check_bits_pix(pixel_sub, [12]), False)

    def test_wide_mask_map_fits_io_compression(self):
        """
        Test wide mask fits io with and without compression.
        """
        nside_coverage = 32
        nside_map = 4096

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        # Test with double-wide
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         WIDE_MASK, wide_mask_maxbits=16)
        sparse_map.set_bits_pix(np.arange(20000, 50000), [5])
        sparse_map.set_bits_pix(np.arange(120000, 150000), [5, 10])

        fname_comp = os.path.join(self.test_dir, 'test_mask_map_compressed.hs')
        sparse_map.write(fname_comp, clobber=True, nocompress=False)
        fname_nocomp = os.path.join(self.test_dir, 'test_mask_map_notcompressed.hs')
        sparse_map.write(fname_nocomp, clobber=True, nocompress=True)

        self.assertGreater(os.path.getsize(fname_nocomp), os.path.getsize(fname_comp))

        sparse_map_in_comp = healsparse.HealSparseMap.read(fname_comp)
        sparse_map_in_nocomp = healsparse.HealSparseMap.read(fname_nocomp)

        testing.assert_array_equal(sparse_map.valid_pixels,
                                   sparse_map_in_nocomp.valid_pixels)
        testing.assert_array_equal(sparse_map[sparse_map.valid_pixels],
                                   sparse_map_in_nocomp[sparse_map.valid_pixels])
        testing.assert_array_equal(sparse_map[0: 10],
                                   sparse_map_in_nocomp[0: 10])

        testing.assert_array_equal(sparse_map.valid_pixels,
                                   sparse_map_in_comp.valid_pixels)
        testing.assert_array_equal(sparse_map[sparse_map.valid_pixels],
                                   sparse_map_in_comp[sparse_map.valid_pixels])
        testing.assert_array_equal(sparse_map[0: 10],
                                   sparse_map_in_comp[0: 10])

    def test_wide_mask_or(self):
        """
        Test wide mask oring
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        sparse_map_empty = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, WIDE_MASK,
                                                               wide_mask_maxbits=100)

        sparse_map1 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, WIDE_MASK,
                                                          wide_mask_maxbits=100)
        pixel1 = np.arange(4000, 20000)
        pixel1 = np.delete(pixel1, 15000)
        values1 = np.random.poisson(size=(pixel1.size, sparse_map1._wide_mask_width), lam=2).astype(WIDE_MASK)
        sparse_map1.update_values_pix(pixel1, values1)

        sparse_map2 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, WIDE_MASK,
                                                          wide_mask_maxbits=100)
        pixel2 = np.arange(15000, 25000)
        values2 = np.random.poisson(size=(pixel2.size, sparse_map2._wide_mask_width), lam=2).astype(WIDE_MASK)
        sparse_map2.update_values_pix(pixel2, values2)

        or_map_intersection = healsparse.or_intersection([sparse_map1, sparse_map2])

        all_pixels = np.arange(hpg.nside_to_npixel(nside_map))
        arr1 = sparse_map1.get_values_pix(all_pixels)
        arr2 = sparse_map2.get_values_pix(all_pixels)
        gd, = np.where((arr1.sum(axis=1) > 0) & (arr2.sum(axis=1) > 0))
        for i in range(2):
            test_or_intersection = np.zeros_like(arr1[:, 0])
            test_or_intersection[gd] = arr1[gd, i] | arr2[gd, i]
            testing.assert_equal(or_map_intersection.get_values_pix(all_pixels)[:, i],
                                 test_or_intersection)

        or_map_intersection_with_empty = healsparse.or_intersection(
            [
                sparse_map_empty,
                sparse_map1,
                sparse_map2
            ]
        )

        self.assertEqual(or_map_intersection_with_empty.valid_pixels.size, 0)

        or_map_union = healsparse.or_union([sparse_map_empty, sparse_map1, sparse_map2])

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

        sparse_map1 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, WIDE_MASK,
                                                          wide_mask_maxbits=16)
        pixel1 = np.arange(4000, 20000)
        pixel1 = np.delete(pixel1, 15000)
        values1 = np.random.poisson(size=(pixel1.size, sparse_map1._wide_mask_width), lam=2).astype(WIDE_MASK)
        sparse_map1.update_values_pix(pixel1, values1)

        sparse_map2 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, WIDE_MASK,
                                                          wide_mask_maxbits=16)
        pixel2 = np.arange(15000, 25000)
        values2 = np.random.poisson(size=(pixel2.size, sparse_map2._wide_mask_width), lam=2).astype(WIDE_MASK)
        sparse_map2.update_values_pix(pixel2, values2)

        and_map_intersection = healsparse.and_intersection([sparse_map1, sparse_map2])

        all_pixels = np.arange(hpg.nside_to_npixel(nside_map))
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

        sparse_map1 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, WIDE_MASK,
                                                          wide_mask_maxbits=100)
        pixel1 = np.arange(4000, 20000)
        pixel1 = np.delete(pixel1, 15000)
        values1 = np.random.poisson(size=(pixel1.size, sparse_map1._wide_mask_width), lam=2).astype(WIDE_MASK)
        sparse_map1.update_values_pix(pixel1, values1)

        sparse_map2 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, WIDE_MASK,
                                                          wide_mask_maxbits=100)
        pixel2 = np.arange(15000, 25000)
        values2 = np.random.poisson(size=(pixel2.size, sparse_map2._wide_mask_width), lam=2).astype(WIDE_MASK)
        sparse_map2.update_values_pix(pixel2, values2)

        xor_map_intersection = healsparse.xor_intersection([sparse_map1, sparse_map2])

        all_pixels = np.arange(hpg.nside_to_npixel(nside_map))
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

    def test_wide_mask_polygon_or(self):
        """
        Make a wide mask with a polygon
        """
        nside_coverage = 128
        nside_sparse = 2**15

        ra = [200.0, 200.2, 200.3, 200.2, 200.1]
        dec = [0.0, 0.1, 0.2, 0.25, 0.13]
        poly = healsparse.geom.Polygon(ra=ra, dec=dec, value=[4, 15])

        # Test making a map from polygon, infer the maxbits
        smap = poly.get_map(nside_coverage=nside_coverage, nside_sparse=nside_sparse,
                            dtype=WIDE_MASK)
        self.assertTrue(smap.is_wide_mask_map)
        self.assertEqual(smap.wide_mask_maxbits, 16)

        ra = np.array([200.1, 200.15])
        dec = np.array([0.05, 0.015])

        vals = smap.get_values_pos(ra, dec, lonlat=True)
        testing.assert_array_equal(vals[:, 0], [2**4, 0])
        testing.assert_array_equal(vals[:, 1], [2**(15 - 8), 0])

        # Test making a map from polygon, specify the maxbits
        smap = poly.get_map(nside_coverage=nside_coverage, nside_sparse=nside_sparse,
                            dtype=WIDE_MASK, wide_mask_maxbits=24)
        self.assertTrue(smap.is_wide_mask_map)
        self.assertEqual(smap.wide_mask_maxbits, 24)

        vals = smap.get_values_pos(ra, dec, lonlat=True)
        testing.assert_array_equal(vals[:, 0], [2**4, 0])
        testing.assert_array_equal(vals[:, 1], [2**(15 - 8), 0])

        # And a map like another
        smap2 = poly.get_map_like(smap)
        self.assertTrue(smap2.is_wide_mask_map)
        self.assertEqual(smap2.wide_mask_maxbits, 24)

        vals = smap2.get_values_pos(ra, dec, lonlat=True)
        testing.assert_array_equal(vals[:, 0], [2**4, 0])
        testing.assert_array_equal(vals[:, 1], [2**(15 - 8), 0])

        # Test realizing two maps
        nside_sparse = 2**17

        radius1 = 0.075
        radius2 = 0.075
        ra1, dec1 = 200.0, 0.0
        ra2, dec2 = 200.1, 0.0
        value1 = [5, 15]
        value2 = [6]

        circle1 = healsparse.geom.Circle(ra=ra1, dec=dec1,
                                         radius=radius1, value=value1)
        circle2 = healsparse.geom.Circle(ra=ra2, dec=dec2,
                                         radius=radius2, value=value2)

        smap = healsparse.HealSparseMap.make_empty(nside_coverage, nside_sparse,
                                                   WIDE_MASK, wide_mask_maxbits=16)
        healsparse.geom.realize_geom([circle1, circle2], smap)

        out_ra, out_dec = 190.0, 25.0
        in1_ra, in1_dec = 200.02, 0.0
        in2_ra, in2_dec = 200.095, 0.0
        both_ra, both_dec = 200.05, 0.0

        out_vals = smap.get_values_pos(out_ra, out_dec, lonlat=True)
        in1_vals = smap.get_values_pos(in1_ra, in1_dec, lonlat=True)
        in2_vals = smap.get_values_pos(in2_ra, in2_dec, lonlat=True)
        both_vals = smap.get_values_pos(both_ra, both_dec, lonlat=True)

        testing.assert_array_equal(out_vals, [0, 0])
        testing.assert_array_equal(in1_vals, [2**value1[0], 2**(value1[1] - 8)])
        testing.assert_array_equal(in2_vals, [2**value2[0], 0])
        testing.assert_array_equal(both_vals, [2**value1[0] | 2**value2[0],
                                               2**(value1[1] - 8)])

    def test_wide_mask_apply_mask(self):
        """
        Test apply_mask with a wide mask map
        """
        nside_coverage = 128
        nside_sparse = 2**15

        box = healsparse.geom.Polygon(ra=[200.0, 200.2, 200.2, 200.0],
                                      dec=[10.0, 10.0, 10.2, 10.2],
                                      value=[70])

        mask_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_sparse,
                                                       WIDE_MASK, sentinel=0, wide_mask_maxbits=70)
        healsparse.geom.realize_geom(box, mask_map)

        # Create an integer value map, using a bigger box
        box2 = healsparse.geom.Polygon(ra=[199.8, 200.4, 200.4, 199.8],
                                       dec=[9.8, 9.8, 10.4, 10.4],
                                       value=1)
        int_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_sparse, np.int16, sentinel=0)
        healsparse.geom.realize_geom(box2, int_map)

        valid_pixels = int_map.valid_pixels

        # Default, mask all bits
        masked_map = int_map.apply_mask(mask_map, in_place=False)
        masked_pixels = mask_map.valid_pixels

        # Masked pixels should be zero
        testing.assert_array_equal(masked_map.get_values_pix(masked_pixels), 0)

        # Pixels that are in the original but are not in the masked pixels should be 1
        still_good, = np.where((int_map.get_values_pix(valid_pixels) > 0) &
                               (mask_map.get_values_pix(valid_pixels).sum(axis=1) == 0))
        testing.assert_array_equal(masked_map.get_values_pix(valid_pixels[still_good]), 1)

        # Mask specific bits
        masked_map = int_map.apply_mask(mask_map, mask_bit_arr=[70], in_place=False)

        # Masked pixels should be zero
        testing.assert_array_equal(masked_map.get_values_pix(masked_pixels), 0)

        # Pixels that are in the original but are not in the masked pixels should be 1
        still_good, = np.where((int_map.get_values_pix(valid_pixels) > 0) &
                               (mask_map.get_values_pix(valid_pixels).sum(axis=1) == 0))
        testing.assert_array_equal(masked_map.get_values_pix(valid_pixels[still_good]), 1)

        # And mask specific bits that are not set
        masked_map = int_map.apply_mask(mask_map, mask_bit_arr=[16], in_place=False)
        values = mask_map.get_values_pix(mask_map.valid_pixels)
        masked_pixels, = np.where((values[:, 0] & (2**16)) > 0)
        testing.assert_equal(masked_pixels.size, 0)

        still_good, = np.where((int_map.get_values_pix(valid_pixels) > 0) &
                               ((mask_map.get_values_pix(valid_pixels)[:, 0] & 2**16) == 0))
        testing.assert_array_equal(masked_map.get_values_pix(valid_pixels[still_good]), 1)

    def test_wide_mask_applied_mask(self):
        """
        Test apply_mask to a wide mask map
        """
        nside_coverage = 128
        nside_sparse = 2**15

        box = healsparse.geom.Polygon(ra=[200.0, 200.2, 200.2, 200.0],
                                      dec=[10.0, 10.0, 10.2, 10.2],
                                      value=1)
        mask_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_sparse,
                                                       np.int16, sentinel=0)
        healsparse.geom.realize_geom(box, mask_map)

        # Create a wide mask map, using a bigger box.
        # Do two times with a narrow and a wide wide mask
        for bitset in [1, 70]:
            box2 = healsparse.geom.Polygon(ra=[199.8, 200.4, 200.4, 199.8],
                                           dec=[9.8, 9.8, 10.4, 10.4],
                                           value=[bitset])
            wide_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_sparse,
                                                           WIDE_MASK, sentinel=0,
                                                           wide_mask_maxbits=bitset)
            healsparse.geom.realize_geom(box2, wide_map)

            valid_pixels = wide_map.valid_pixels

            masked_map = wide_map.apply_mask(mask_map, in_place=False)
            masked_pixels = mask_map.valid_pixels

            # Masked pixels should be 0
            testing.assert_array_equal(masked_map[masked_pixels], 0)

            # Pixels in the original but not in masked pixels should be 1
            still_good, = np.where((wide_map[valid_pixels].sum(axis=1) > 0) &
                                   (mask_map[valid_pixels] == 0))

            field, bitval = healsparse.utils._get_field_and_bitval(bitset)
            testing.assert_array_equal(masked_map[valid_pixels[still_good]][:, field],
                                       bitval)

    def test_wide_mask_constoperations(self):
        """
        Test wide mask operations with constants
        """
        nside_coverage = 32
        nside_map = 64

        sparse_map1 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, WIDE_MASK,
                                                          wide_mask_maxbits=100)
        pixel1 = np.arange(4000, 20000)
        sparse_map1.set_bits_pix(pixel1, [5])

        # Or to new map
        sparse_map2 = sparse_map1 | [20]

        self.assertEqual(pixel1.size, sparse_map2.valid_pixels.size)
        testing.assert_array_equal(sparse_map2.check_bits_pix(pixel1, [5]), True)
        testing.assert_array_equal(sparse_map2.check_bits_pix(pixel1, [20]), True)
        testing.assert_array_equal(sparse_map2.check_bits_pix(pixel1, [30]), False)

        # Or to the same map
        sparse_map1 |= [20]

        self.assertEqual(pixel1.size, sparse_map1.valid_pixels.size)
        testing.assert_array_equal(sparse_map1.check_bits_pix(pixel1, [5]), True)
        testing.assert_array_equal(sparse_map1.check_bits_pix(pixel1, [20]), True)
        testing.assert_array_equal(sparse_map1.check_bits_pix(pixel1, [30]), False)

        # And to a new map
        sparse_map2 = sparse_map1 & [5]
        testing.assert_array_equal(sparse_map2.check_bits_pix(pixel1, [5]), True)
        testing.assert_array_equal(sparse_map2.check_bits_pix(pixel1, [20]), False)
        testing.assert_array_equal(sparse_map2.check_bits_pix(pixel1, [30]), False)

        # And to itself
        sparse_map1 &= [5]
        testing.assert_array_equal(sparse_map1.check_bits_pix(pixel1, [5]), True)
        testing.assert_array_equal(sparse_map1.check_bits_pix(pixel1, [20]), False)
        testing.assert_array_equal(sparse_map1.check_bits_pix(pixel1, [30]), False)

        # Reset this
        sparse_map1.set_bits_pix(pixel1, [5, 20])

        # Test xor
        sparse_map2 = sparse_map1 ^ [5, 30]
        testing.assert_array_equal(sparse_map2.check_bits_pix(pixel1, [5]), False)
        testing.assert_array_equal(sparse_map2.check_bits_pix(pixel1, [20]), True)
        testing.assert_array_equal(sparse_map2.check_bits_pix(pixel1, [30]), True)

        sparse_map1 ^= [5, 30]
        testing.assert_array_equal(sparse_map1.check_bits_pix(pixel1, [5]), False)
        testing.assert_array_equal(sparse_map1.check_bits_pix(pixel1, [20]), True)
        testing.assert_array_equal(sparse_map1.check_bits_pix(pixel1, [30]), True)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__ == '__main__':
    unittest.main()
