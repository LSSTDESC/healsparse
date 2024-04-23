import unittest
import numpy.testing as testing
import numpy as np
import hpgeom as hpg
from numpy import random
import tempfile
import shutil
import os
import pytest
import pathlib

import healsparse


if not healsparse.parquet_shim.use_pyarrow:
    pytest.skip("Skipping pyarrow/parquet tests", allow_module_level=True)


class ParquetIoTestCase(unittest.TestCase):
    def test_parquet_writeread(self):
        """
        Test parquet i/o functionality.
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        n_rand = 1000
        ra = np.random.random(n_rand) * 360.0
        dec = np.random.random(n_rand) * 180.0 - 90.0

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        # Generate a random map

        full_map = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        full_map[0: 20000] = np.random.random(size=20000)

        ipnest = hpg.angle_to_pixel(nside_map, ra, dec)
        test_values = full_map[ipnest]

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage,
                                                         nside_map,
                                                         full_map.dtype)
        u, = np.where(full_map > hpg.UNSEEN)
        sparse_map[u] = full_map[u]

        for mode in ("str", "path"):
            if mode == "str":
                readfile = os.path.join(self.test_dir, "healsparse_map.hsparquet")
            else:
                readfile = self.test_dir / pathlib.Path("healsparse_map2.hsparquet")

                # Write it in healsparse parquet format
                sparse_map.write(readfile, format="parquet")

                # Read in healsparse format (full map)
                sparse_map = healsparse.HealSparseMap.read(readfile)
                # Check that we can do a basic lookup
                testing.assert_almost_equal(sparse_map.get_values_pix(ipnest), test_values)

                # Try to read in healsparse format, non-unique pixels
                self.assertRaises(RuntimeError, healsparse.HealSparseMap.read,
                                  readfile, pixels=[0, 0])

                # Read in healsparse (two pixels)
                sparse_map_small = healsparse.HealSparseMap.read(readfile, pixels=[0, 1])

                # Test the coverage map only has two pixels
                cov_mask = sparse_map_small.coverage_mask
                self.assertEqual(cov_mask.sum(), 2)

                # Test lookup of values in those two pixels
                ipnestCov = np.right_shift(ipnest, sparse_map_small._cov_map.bit_shift)
                outside_small, = np.where(ipnestCov > 1)
                test_values2 = test_values.copy()
                test_values2[outside_small] = hpg.UNSEEN

                testing.assert_almost_equal(sparse_map_small.get_values_pix(ipnest), test_values2)

                # Read in healsparse format (all pixels)
                sparse_map_full = healsparse.HealSparseMap.read(
                    readfile,
                    pixels=np.arange(hpg.nside_to_npixel(nside_coverage)),
                )
                testing.assert_almost_equal(sparse_map_full.get_values_pix(ipnest), test_values)

    def test_parquet_read_outoforder(self):
        """
        Test reading parquet maps that have been written with out-of-order pixels
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        n_rand = 1000
        ra = np.random.random(n_rand) * 360.0
        dec = np.random.random(n_rand) * 180.0 - 90.0

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        # Create an empty map
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.float64)
        fname = os.path.join(self.test_dir, 'healsparse_map_outoforder.hsparquet')

        # Fill it out of order
        pixel = np.arange(4000, 20000)
        values = np.random.random(pixel.size)
        sparse_map.update_values_pix(pixel, values)
        pixel2 = np.arange(1000)
        values2 = np.random.random(pixel2.size)
        sparse_map.update_values_pix(pixel2, values2)

        sparse_map.write(fname, format='parquet')

        # And read it in...
        sparse_map = healsparse.HealSparseMap.read(fname)

        # Test some values
        ipnest = hpg.angle_to_pixel(nside_map, ra, dec)
        test_map = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        test_map[pixel] = values
        test_map[pixel2] = values2

        testing.assert_almost_equal(sparse_map.get_values_pix(ipnest), test_map[ipnest])

        # Read in the first two and the Nth pixel

        # These pixels are chosen because they are covered by the random test points
        sparse_map_small = healsparse.HealSparseMap.read(fname, pixels=[0, 1, 3179])

        # Test some values
        ipnest_cov = np.right_shift(ipnest, sparse_map_small._cov_map.bit_shift)
        test_values_small = test_map[ipnest]
        outside_small, = np.where((ipnest_cov != 0) & (ipnest_cov != 1) & (ipnest_cov != 3179))
        test_values_small[outside_small] = hpg.UNSEEN

        testing.assert_almost_equal(sparse_map_small.get_values_pix(ipnest), test_values_small)

    def test_parquet_writeread_withheader(self):
        """
        Test parquet i/o functionality with a header
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')
        fname = os.path.join(self.test_dir, 'sparsemap_with_header.hs')

        full_map = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        full_map[0: 20000] = np.random.random(size=20000)

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map,
                                              nside_coverage=nside_coverage, nest=True)

        hdr = {}
        hdr['TESTING'] = 1.0

        sparse_map.metadata = hdr
        sparse_map.write(fname, format='parquet')

        ret_map, ret_hdr = healsparse.HealSparseMap.read(fname, header=True)

        self.assertEqual(hdr['TESTING'], ret_hdr['TESTING'])

    def test_parquet_writeread_highres(self):
        """
        Test parquet i/o functionality at very high resolution
        """
        random.seed(seed=12345)

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')
        fname = os.path.join(self.test_dir, 'healsparse_map.hsparquet')

        nside_coverage = 32
        nside_map = 2**17

        sparse_map = healsparse.HealSparseMap.make_empty(nside_sparse=nside_map,
                                                         nside_coverage=nside_coverage,
                                                         dtype=bool)
        sparse_map[1_000_000: 20_000_000] = True
        sparse_map[1_700_000_000: 1_720_000_000] = True

        valid_pixels = sparse_map.valid_pixels

        sparse_map.write(fname, format='parquet')

        sparse_map2 = healsparse.HealSparseMap.read(fname)
        valid_pixels2 = sparse_map2.valid_pixels

        testing.assert_array_equal(valid_pixels2, valid_pixels)
        testing.assert_array_equal(sparse_map2.get_values_pix(valid_pixels2), True)

        # And read pixel by pixel
        for covpix_map in sparse_map.get_covpix_maps():
            covpix, = np.where(covpix_map.coverage_mask)

            covpix_map2 = healsparse.HealSparseMap.read(fname, pixels=covpix)

            testing.assert_array_equal(covpix_map2.valid_pixels, covpix_map.valid_pixels)

    def test_write_bad_nside_io(self):
        """
        Test raising when trying to write with bad nside_io values.
        """
        random.seed(seed=12345)

        nside_coverage = 8
        nside_map = 64

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        # Generate a random map

        full_map = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        full_map[0: 20000] = np.random.random(size=20000)

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage,
                                                         nside_map,
                                                         full_map.dtype)
        u, = np.where(full_map > hpg.UNSEEN)
        sparse_map[u] = full_map[u]

        fname = os.path.join(self.test_dir, 'healsparse_map.hsparquet')

        # This nside_io is larger than the nside_coverage
        self.assertRaises(ValueError, sparse_map.write, fname,
                          format='parquet', nside_io=16)

        sparse_map2 = healsparse.HealSparseMap.make_empty(64, nside_map, full_map.dtype)
        sparse_map2[u] = full_map[u]

        # This nside_io is larger than 16
        self.assertRaises(ValueError, sparse_map2.write, fname,
                          format='parquet', nside_io=32)

    def test_parquet_writeread_bool(self):
        """Test writing and reading a bool map."""
        nside_coverage = 32
        nside_map = 64

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, bool)
        sparse_map[30000: 30005] = True

        # Write it to healsparse format
        sparse_map.write(os.path.join(self.test_dir, 'healsparse_map.hsparquet'), format='parquet')

        # Read in healsparse format (full map)
        sparse_map2 = healsparse.HealSparseMap.read(os.path.join(self.test_dir, 'healsparse_map.hsparquet'))

        # Check that we can do a basic lookup
        testing.assert_array_equal(sparse_map2[30000: 30005], True)

        self.assertEqual(len(sparse_map2.valid_pixels), 5)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__ == '__main__':
    unittest.main()
