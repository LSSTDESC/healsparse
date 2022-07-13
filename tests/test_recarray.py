import unittest
import numpy.testing as testing
import numpy as np
import hpgeom as hpg
from numpy import random
import tempfile
import shutil
import os
import pytest

import healsparse


class RecArrayTestCase(unittest.TestCase):
    def test_fits_writeread_recarray(self):
        """
        Test fits recarray writing and reading.
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        nRand = 1000
        ra = np.random.random(nRand) * 360.0
        dec = np.random.random(nRand) * 180.0 - 90.0

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')
        fname = os.path.join(self.test_dir, 'healsparse_map_recarray.hsp')

        dtype = [('col1', 'f8'), ('col2', 'f8')]
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype, primary='col1')
        pixel = np.arange(20000)
        values = np.zeros_like(pixel, dtype=dtype)
        values['col1'] = np.random.random(size=pixel.size)
        values['col2'] = np.random.random(size=pixel.size)
        sparse_map.update_values_pix(pixel, values)

        sparse_map.write(fname)

        # Make the test values
        hpmapCol1 = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        hpmapCol2 = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        hpmapCol1[pixel] = values['col1']
        hpmapCol2[pixel] = values['col2']

        ipnest_test = hpg.angle_to_pixel(nside_map, ra, dec)

        # Read in the map
        sparse_map = healsparse.HealSparseMap.read(fname)

        testing.assert_almost_equal(sparse_map.get_values_pos(ra, dec, lonlat=True)['col1'],
                                    hpmapCol1[ipnest_test])
        testing.assert_almost_equal(sparse_map.get_values_pos(ra, dec, lonlat=True)['col2'],
                                    hpmapCol2[ipnest_test])

        # Test the list of valid pixels
        valid_pixels = sparse_map.valid_pixels
        testing.assert_equal(valid_pixels, pixel)
        testing.assert_equal(sparse_map.n_valid, len(pixel))

        # Read in a partial map...
        sparse_map_small = healsparse.HealSparseMap.read(fname, pixels=[0, 1])

        # Test the coverage map only has two pixels
        cov_mask = sparse_map_small.coverage_mask
        self.assertEqual(cov_mask.sum(), 2)

        # Test lookup of values in these two pixels
        ipnest_cov = np.right_shift(ipnest_test, sparse_map_small._cov_map.bit_shift)
        outside_small, = np.where(ipnest_cov > 1)
        # column1 is the "primary" column and will return UNSEEN
        test_values1b = hpmapCol1[ipnest_test].copy()
        test_values1b[outside_small] = hpg.UNSEEN
        # column2 is not the primary column and will also return UNSEEN
        test_values2b = hpmapCol2[ipnest_test].copy()
        test_values2b[outside_small] = hpg.UNSEEN

        testing.assert_almost_equal(sparse_map_small.get_values_pix(ipnest_test)['col1'], test_values1b)
        testing.assert_almost_equal(sparse_map_small.get_values_pix(ipnest_test)['col2'], test_values2b)

    def test_fits_read_outoforder_recarray(self):
        """
        Test fits reading of recarray maps that have been written out-of-order
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        nRand = 1000
        ra = np.random.random(nRand) * 360.0
        dec = np.random.random(nRand) * 180.0 - 90.0

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')
        fname = os.path.join(self.test_dir, 'healsparse_map_recarray_outoforder.hsp')

        # Create an empty map
        dtype = [('col1', 'f8'), ('col2', 'f8')]
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype, primary='col1')
        pixel = np.arange(4000, 20000)
        values = np.zeros_like(pixel, dtype=dtype)
        values['col1'] = np.random.random(size=pixel.size)
        values['col2'] = np.random.random(size=pixel.size)
        sparse_map.update_values_pix(pixel, values)
        pixel2 = np.arange(1000)
        values2 = np.zeros_like(pixel2, dtype=dtype)
        values2['col1'] = np.random.random(size=pixel2.size)
        values2['col2'] = np.random.random(size=pixel2.size)
        sparse_map.update_values_pix(pixel2, values2)

        sparse_map.write(fname)

        # And read it in...
        sparse_map = healsparse.HealSparseMap.read(fname)

        # Test some values
        ipnest = hpg.angle_to_pixel(nside_map, ra, dec)
        test_map_col1 = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        test_map_col1[pixel] = values['col1']
        test_map_col1[pixel2] = values2['col1']
        test_map_col2 = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        test_map_col2[pixel] = values['col2']
        test_map_col2[pixel2] = values2['col2']

        testing.assert_almost_equal(sparse_map.get_values_pix(ipnest)['col1'], test_map_col1[ipnest])
        testing.assert_almost_equal(sparse_map.get_values_pix(ipnest)['col2'], test_map_col2[ipnest])

        # These pixels are chosen because they are covered by the random test points
        sparse_map_small = healsparse.HealSparseMap.read(fname, pixels=[0, 1, 3179])

        ipnest_cov = np.right_shift(ipnest, sparse_map_small._cov_map.bit_shift)
        test_values_small_col1 = test_map_col1[ipnest]
        test_values_small_col2 = test_map_col2[ipnest]
        outside_small, = np.where((ipnest_cov != 0) & (ipnest_cov != 1) & (ipnest_cov != 3179))
        test_values_small_col1[outside_small] = hpg.UNSEEN
        test_values_small_col2[outside_small] = hpg.UNSEEN

        testing.assert_almost_equal(sparse_map_small.get_values_pix(ipnest)['col1'],
                                    test_values_small_col1)
        testing.assert_almost_equal(sparse_map_small.get_values_pix(ipnest)['col2'],
                                    test_values_small_col2)

    @pytest.mark.skipif(not healsparse.parquet_shim.use_pyarrow, reason='Requires pyarrow')
    def test_parquet_writeread_recarray(self):
        """
        Test parquet recarray writing and reading.
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        nRand = 1000
        ra = np.random.random(nRand) * 360.0
        dec = np.random.random(nRand) * 180.0 - 90.0

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')
        fname = os.path.join(self.test_dir, 'healsparse_map_recarray.hsparquet')

        dtype = [('col1', 'f8'), ('col2', 'f8')]
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype, primary='col1')
        pixel = np.arange(20000)
        values = np.zeros_like(pixel, dtype=dtype)
        values['col1'] = np.random.random(size=pixel.size)
        values['col2'] = np.random.random(size=pixel.size)
        sparse_map.update_values_pix(pixel, values)

        sparse_map.write(fname, format='parquet')

        # Make the test values
        hpmapCol1 = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        hpmapCol2 = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        hpmapCol1[pixel] = values['col1']
        hpmapCol2[pixel] = values['col2']
        ipnest_test = hpg.angle_to_pixel(nside_map, ra, dec)

        # Read in the map
        sparse_map = healsparse.HealSparseMap.read(fname)

        testing.assert_almost_equal(sparse_map.get_values_pos(ra, dec, lonlat=True)['col1'],
                                    hpmapCol1[ipnest_test])
        testing.assert_almost_equal(sparse_map.get_values_pos(ra, dec, lonlat=True)['col2'],
                                    hpmapCol2[ipnest_test])

        # Test the list of valid pixels
        valid_pixels = sparse_map.valid_pixels
        testing.assert_equal(valid_pixels, pixel)

        # Read in a partial map...
        sparse_map_small = healsparse.HealSparseMap.read(fname, pixels=[0, 1])

        # Test the coverage map only has two pixels
        cov_mask = sparse_map_small.coverage_mask
        self.assertEqual(cov_mask.sum(), 2)

        # Test lookup of values in these two pixels
        ipnest_cov = np.right_shift(ipnest_test, sparse_map_small._cov_map.bit_shift)
        outside_small, = np.where(ipnest_cov > 1)
        # column1 is the "primary" column and will return UNSEEN
        test_values1b = hpmapCol1[ipnest_test].copy()
        test_values1b[outside_small] = hpg.UNSEEN
        # column2 is not the primary column and will also return UNSEEN
        test_values2b = hpmapCol2[ipnest_test].copy()
        test_values2b[outside_small] = hpg.UNSEEN

        testing.assert_almost_equal(sparse_map_small.get_values_pix(ipnest_test)['col1'], test_values1b)
        testing.assert_almost_equal(sparse_map_small.get_values_pix(ipnest_test)['col2'], test_values2b)

    @pytest.mark.skipif(not healsparse.parquet_shim.use_pyarrow, reason='Requires pyarrow')
    def test_parquet_read_outoforder_recarray(self):
        """
        Test parquet reading of recarray maps that have been written out-of-order
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        nRand = 1000
        ra = np.random.random(nRand) * 360.0
        dec = np.random.random(nRand) * 180.0 - 90.0

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')
        fname = os.path.join(self.test_dir, 'healsparse_map_recarray_outoforder.hsparquet')

        # Create an empty map
        dtype = [('col1', 'f8'), ('col2', 'f8')]
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype, primary='col1')
        pixel = np.arange(4000, 20000)
        values = np.zeros_like(pixel, dtype=dtype)
        values['col1'] = np.random.random(size=pixel.size)
        values['col2'] = np.random.random(size=pixel.size)
        sparse_map.update_values_pix(pixel, values)
        pixel2 = np.arange(1000)
        values2 = np.zeros_like(pixel2, dtype=dtype)
        values2['col1'] = np.random.random(size=pixel2.size)
        values2['col2'] = np.random.random(size=pixel2.size)
        sparse_map.update_values_pix(pixel2, values2)

        sparse_map.write(fname, format='parquet')

        # And read it in...
        sparse_map = healsparse.HealSparseMap.read(fname)

        # Test some values
        ipnest = hpg.angle_to_pixel(nside_map, ra, dec)
        test_map_col1 = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        test_map_col1[pixel] = values['col1']
        test_map_col1[pixel2] = values2['col1']
        test_map_col2 = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        test_map_col2[pixel] = values['col2']
        test_map_col2[pixel2] = values2['col2']

        testing.assert_almost_equal(sparse_map.get_values_pix(ipnest)['col1'], test_map_col1[ipnest])
        testing.assert_almost_equal(sparse_map.get_values_pix(ipnest)['col2'], test_map_col2[ipnest])

        # These pixels are chosen because they are covered by the random test points
        sparse_map_small = healsparse.HealSparseMap.read(fname, pixels=[0, 1, 3179])

        ipnest_cov = np.right_shift(ipnest, sparse_map_small._cov_map.bit_shift)
        test_values_small_col1 = test_map_col1[ipnest]
        test_values_small_col2 = test_map_col2[ipnest]
        outside_small, = np.where((ipnest_cov != 0) & (ipnest_cov != 1) & (ipnest_cov != 3179))
        test_values_small_col1[outside_small] = hpg.UNSEEN
        test_values_small_col2[outside_small] = hpg.UNSEEN

        testing.assert_almost_equal(sparse_map_small.get_values_pix(ipnest)['col1'],
                                    test_values_small_col1)
        testing.assert_almost_equal(sparse_map_small.get_values_pix(ipnest)['col2'],
                                    test_values_small_col2)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__ == '__main__':
    unittest.main()
