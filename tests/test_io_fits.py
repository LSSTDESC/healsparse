import unittest
import numpy.testing as testing
import numpy as np
import hpgeom as hpg
from numpy import random
import tempfile
import shutil
import os
import pathlib

import healsparse


class HealsparseFitsIoTestCase(unittest.TestCase):
    def test_fits_writeread(self):
        """
        Test healsparse fits i/o functionality
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

        # Test that we can do a basic set
        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage, nest=True)

        sparse_map[30000: 30005] = np.zeros(5, dtype=np.float64)

        for mode in ("str", "path"):
            if mode == "str":
                readfile = os.path.join(self.test_dir, "healsparse_map.hsp")
            else:
                readfile = self.test_dir / pathlib.Path("healsparse_map2.hsp")

            # Write it to healsparse format
            sparse_map.write(readfile)

            # Read in healsparse format (full map)
            sparse_map = healsparse.HealSparseMap.read(readfile)

            # Check that we can do a basic lookup
            testing.assert_almost_equal(sparse_map.get_values_pix(ipnest), test_values)

            # Check that we can do a basic set
            sparse_map[30000: 30005] = np.zeros(5, dtype=np.float64)

            # Try to read in healsparse format, non-unique pixels
            self.assertRaises(RuntimeError, healsparse.HealSparseMap.read, readfile, pixels=[0, 0])

            # Read in healsparse format (two pixels)
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

    def test_fits_read_outoforder(self):
        """
        Test reading fits maps that have been written with out-of-order pixels
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

        # Fill it out of order
        pixel = np.arange(4000, 20000)
        values = np.random.random(pixel.size)
        sparse_map.update_values_pix(pixel, values)
        pixel2 = np.arange(1000)
        values2 = np.random.random(pixel2.size)
        sparse_map.update_values_pix(pixel2, values2)

        sparse_map.write(os.path.join(self.test_dir, 'healsparse_map_outoforder.hs'))

        # And read it in...
        sparse_map = healsparse.HealSparseMap.read(os.path.join(self.test_dir,
                                                                'healsparse_map_outoforder.hs'))

        # Test some values
        ipnest = hpg.angle_to_pixel(nside_map, ra, dec)
        test_map = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        test_map[pixel] = values
        test_map[pixel2] = values2

        testing.assert_almost_equal(sparse_map.get_values_pix(ipnest), test_map[ipnest])

        # Read in the first two and the Nth pixel

        # These pixels are chosen because they are covered by the random test points
        sparse_map_small = healsparse.HealSparseMap.read(
            os.path.join(self.test_dir, 'healsparse_map_outoforder.hs'), pixels=[0, 1, 3179])

        # Test some values
        ipnest_cov = np.right_shift(ipnest, sparse_map_small._cov_map.bit_shift)
        test_values_small = test_map[ipnest]
        outside_small, = np.where((ipnest_cov != 0) & (ipnest_cov != 1) & (ipnest_cov != 3179))
        test_values_small[outside_small] = hpg.UNSEEN

        testing.assert_almost_equal(sparse_map_small.get_values_pix(ipnest), test_values_small)

    def test_fits_writeread_withheader(self):
        """
        Test fits i/o functionality with a header
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        full_map = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        full_map[0: 20000] = np.random.random(size=20000)

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map,
                                              nside_coverage=nside_coverage, nest=True)

        hdr = {}
        hdr['TESTING'] = 1.0

        sparse_map.metadata = hdr
        sparse_map.write(os.path.join(self.test_dir, 'sparsemap_with_header.hs'))

        ret_map, ret_hdr = healsparse.HealSparseMap.read(os.path.join(self.test_dir,
                                                                      'sparsemap_with_header.hs'),
                                                         header=True)

        self.assertEqual(hdr['TESTING'], ret_hdr['TESTING'])

    def test_fits_writeread_highres(self):
        """
        Test fits i/o functionality at very high resolution
        """
        random.seed(seed=12345)

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        nside_coverage = 128
        nside_map = 2**17

        pixels = hpg.query_circle(nside_map, 100.0, 0.0, 0.2/60.)
        values = np.zeros(pixels.size, dtype=np.int32) + 8

        sparse_map = healsparse.HealSparseMap.make_empty(nside_sparse=nside_map,
                                                         nside_coverage=nside_coverage,
                                                         dtype=np.int32)
        sparse_map.update_values_pix(pixels, values)

        valid_pixels = sparse_map.valid_pixels
        valid_pixels.sort()

        testing.assert_array_equal(valid_pixels, pixels)
        testing.assert_array_equal(sparse_map.get_values_pix(valid_pixels), values)

        sparse_map.write(os.path.join(self.test_dir, 'healsparse_map.hs'))

        sparse_map2 = healsparse.HealSparseMap.read(os.path.join(self.test_dir, 'healsparse_map.hs'))

        valid_pixels2 = sparse_map2.valid_pixels

        testing.assert_array_equal(valid_pixels2, pixels)
        testing.assert_array_equal(sparse_map2.get_values_pix(valid_pixels2), values)

    def test_fits_write_compression_int(self):
        """
        Test fits writing integer maps with and without compression
        """
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        random.seed(seed=12345)

        # We do not expect the 64-bit to be compressed, but check that it works.
        for int_dtype in [np.int16, np.int32, np.int64]:
            sparse_map = healsparse.HealSparseMap.make_empty(32, 4096, int_dtype)
            sparse_map[20000: 50000] = random.poisson(lam=100, size=30000).astype(int_dtype)
            sparse_map[120000: 150000] = -random.poisson(lam=100, size=30000).astype(int_dtype)

            fname_comp = os.path.join(self.test_dir, 'test_int_map_compressed.hs')
            sparse_map.write(fname_comp, clobber=True, nocompress=False)
            fname_nocomp = os.path.join(self.test_dir, 'test_int_map_notcompressed.hs')
            sparse_map.write(fname_nocomp, clobber=True, nocompress=True)

            # Compare the file sizes
            if int_dtype == np.int64:
                self.assertEqual(os.path.getsize(fname_nocomp), os.path.getsize(fname_comp))
            else:
                self.assertGreater(os.path.getsize(fname_nocomp), os.path.getsize(fname_comp))

            # Read in and compare
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

    def test_fits_write_compression_float(self):
        """
        Test fits writing floating point maps with and without compression
        """
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        random.seed(seed=12345)

        # We do not expect the 64-bit to be compressed, but check that it works.
        for float_dtype in [np.float32, np.float64]:
            sparse_map = healsparse.HealSparseMap.make_empty(32, 4096, float_dtype)
            sparse_map[20000: 50000] = random.normal(scale=100.0, size=30000).astype(float_dtype)
            sparse_map[120000: 150000] = random.normal(scale=1000.0, size=30000).astype(float_dtype)

            fname_comp = os.path.join(self.test_dir, 'test_float_map_compressed.hs')
            sparse_map.write(fname_comp, clobber=True, nocompress=False)
            fname_nocomp = os.path.join(self.test_dir, 'test_float_map_notcompressed.hs')
            sparse_map.write(fname_nocomp, clobber=True, nocompress=True)

            # Compare the file sizes
            self.assertGreater(os.path.getsize(fname_nocomp), os.path.getsize(fname_comp))

            # Read in and compare
            sparse_map_in_comp = healsparse.HealSparseMap.read(fname_comp)
            sparse_map_in_nocomp = healsparse.HealSparseMap.read(fname_nocomp)

            testing.assert_array_equal(sparse_map.valid_pixels,
                                       sparse_map_in_nocomp.valid_pixels)
            testing.assert_array_almost_equal(sparse_map[sparse_map.valid_pixels],
                                              sparse_map_in_nocomp[sparse_map.valid_pixels])
            testing.assert_array_almost_equal(sparse_map[0: 10],
                                              sparse_map_in_nocomp[0: 10])

            testing.assert_array_equal(sparse_map.valid_pixels,
                                       sparse_map_in_comp.valid_pixels)
            testing.assert_array_almost_equal(sparse_map[sparse_map.valid_pixels],
                                              sparse_map_in_comp[sparse_map.valid_pixels])
            testing.assert_array_almost_equal(sparse_map[0: 10],
                                              sparse_map_in_comp[0: 10])

    def test_read_non_fits(self):
        """Test reading a file that isn't a fits file or parquet file."""
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        fname = os.path.join(self.test_dir, 'NOT_A_FITS_FILE')
        with open(fname, 'w') as f:
            f.write('some text.')

        self.assertRaises(NotImplementedError, healsparse.HealSparseMap.read, fname)

    def test_read_missing_file(self):
        """Test reading a file that isn't there."""
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        fname = os.path.join(self.test_dir, 'NOT_A_FILE')

        self.assertRaises(IOError, healsparse.HealSparseMap.read, fname)

    def test_fits_writeread_bool(self):
        """Test writing and reading a bool map."""
        nside_coverage = 32
        nside_map = 64

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, bool)
        sparse_map[30000: 30005] = True

        # Write it to healsparse format
        sparse_map.write(os.path.join(self.test_dir, 'healsparse_map.hs'))

        # Read in healsparse format (full map)
        sparse_map2 = healsparse.HealSparseMap.read(os.path.join(self.test_dir, 'healsparse_map.hs'))

        # Check that we can do a basic lookup
        testing.assert_array_equal(sparse_map2[30000: 30005], True)

        self.assertEqual(len(sparse_map2.valid_pixels), 5)

        # Need to read in partial map (slightly different code path)
        sparse_map3 = healsparse.HealSparseMap.read(
            os.path.join(self.test_dir, 'healsparse_map.hs'),
            pixels=sparse_map.coverage_mask.nonzero()[0],
        )

        # Check that we can do a basic lookup
        testing.assert_array_equal(sparse_map3[30000: 30005], True)

        self.assertEqual(len(sparse_map3.valid_pixels), 5)

        # And degrade-on-read test ...
        sparse_map4 = healsparse.HealSparseMap.read(
            os.path.join(self.test_dir, 'healsparse_map.hs'),
            degrade_nside=32
        )

        sparse_map_dg = sparse_map.degrade(32)

        testing.assert_array_almost_equal(sparse_map4._sparse_map, sparse_map_dg._sparse_map)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__ == '__main__':
    unittest.main()
