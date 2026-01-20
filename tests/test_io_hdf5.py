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

#read specific coverage pixels not yet implemented
test_pixels_read = False

class Hdf5IoTestCase(unittest.TestCase):
    def test_hdf5_writeread(self):
        """
        Test HDF5 i/o functionality.
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
        full_map[0:20000] = np.random.random(size=20000)

        ipnest = hpg.angle_to_pixel(nside_map, ra, dec)
        test_values = full_map[ipnest]

        sparse_map = healsparse.HealSparseMap.make_empty(
            nside_coverage,
            nside_map,
            full_map.dtype
        )
        u, = np.where(full_map > hpg.UNSEEN)
        sparse_map[u] = full_map[u]

        for mode in ("str", "path"):
            if mode == "str":
                fname = os.path.join(self.test_dir, "healsparse_map.hdf5")
            else:
                fname = pathlib.Path(self.test_dir) / "healsparse_map2.hdf5"

            # Write map
            sparse_map.write(fname, format="hdf5")

            # Read full map
            sparse_map2 = healsparse.HealSparseMap.read(fname)

            testing.assert_almost_equal(
                sparse_map2.get_values_pix(ipnest),
                test_values
            )

            if test_pixels_read:
            # Non-unique pixels should error
                self.assertRaises(
                    RuntimeError,
                    healsparse.HealSparseMap.read,
                    fname,
                    pixels=[0, 0]
                )

                # Read two coverage pixels
                sparse_map_small = healsparse.HealSparseMap.read(
                    fname,
                    pixels=[0, 1]
                )

                cov_mask = sparse_map_small.coverage_mask
                self.assertEqual(cov_mask.sum(), 2)

                ipnest_cov = np.right_shift(
                    ipnest,
                    sparse_map_small._cov_map.bit_shift
                )

                test_values2 = test_values.copy()
                outside_small, = np.where(ipnest_cov > 1)
                test_values2[outside_small] = hpg.UNSEEN

                testing.assert_almost_equal(
                    sparse_map_small.get_values_pix(ipnest),
                    test_values2
                )

                # Read all coverage pixels explicitly
                sparse_map_full = healsparse.HealSparseMap.read(
                    fname,
                    pixels=np.arange(hpg.nside_to_npixel(nside_coverage)),
                )

                testing.assert_almost_equal(
                    sparse_map_full.get_values_pix(ipnest),
                    test_values
                )

    def test_hdf5_read_outoforder(self):
        """
        Test reading HDF5 maps written with out-of-order pixels.
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        n_rand = 1000
        ra = np.random.random(n_rand) * 360.0
        dec = np.random.random(n_rand) * 180.0 - 90.0

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')
        fname = os.path.join(self.test_dir, 'healsparse_map_outoforder.hdf5')

        sparse_map = healsparse.HealSparseMap.make_empty(
            nside_coverage, nside_map, np.float64
        )

        pixel = np.arange(4000, 20000)
        values = np.random.random(pixel.size)
        sparse_map.update_values_pix(pixel, values)

        pixel2 = np.arange(1000)
        values2 = np.random.random(pixel2.size)
        sparse_map.update_values_pix(pixel2, values2)

        sparse_map.write(fname, format='hdf5')

        sparse_map2 = healsparse.HealSparseMap.read(fname)

        ipnest = hpg.angle_to_pixel(nside_map, ra, dec)
        test_map = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        test_map[pixel] = values
        test_map[pixel2] = values2

        testing.assert_almost_equal(
            sparse_map2.get_values_pix(ipnest),
            test_map[ipnest]
        )

        if test_pixels_read:
            sparse_map_small = healsparse.HealSparseMap.read(
                fname,
                pixels=[0, 1, 3179]
            )

            ipnest_cov = np.right_shift(
                ipnest,
                sparse_map_small._cov_map.bit_shift
            )

            test_values_small = test_map[ipnest]
            outside_small, = np.where(
                (ipnest_cov != 0) &
                (ipnest_cov != 1) &
                (ipnest_cov != 3179)
            )
            test_values_small[outside_small] = hpg.UNSEEN

            testing.assert_almost_equal(
                sparse_map_small.get_values_pix(ipnest),
                test_values_small
            )

    def test_hdf5_writeread_withheader(self):
        """
        Test HDF5 i/o functionality with metadata.
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')
        fname = os.path.join(self.test_dir, 'sparsemap_with_header.hdf5')

        full_map = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        full_map[0:20000] = np.random.random(size=20000)

        sparse_map = healsparse.HealSparseMap(
            healpix_map=full_map,
            nside_coverage=nside_coverage,
            nest=True
        )

        hdr = {'TESTING': 1.0}
        sparse_map.metadata = hdr

        sparse_map.write(fname, format='hdf5')

        ret_map, ret_hdr = healsparse.HealSparseMap.read(fname, header=True)

        self.assertEqual(hdr['TESTING'], ret_hdr['TESTING'])

    def test_hdf5_writeread_highres(self):
        """
        Test HDF5 i/o functionality at very high resolution.
        """
        random.seed(seed=12345)

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')
        fname = os.path.join(self.test_dir, 'healsparse_map.hdf5')

        nside_coverage = 32
        nside_map = 2**17

        sparse_map = healsparse.HealSparseMap.make_empty(
            nside_sparse=nside_map,
            nside_coverage=nside_coverage,
            dtype=bool
        )

        sparse_map[1_000_000:20_000_000] = True
        sparse_map[1_700_000_000:1_720_000_000] = True

        valid_pixels = sparse_map.valid_pixels

        sparse_map.write(fname, format='hdf5')

        sparse_map2 = healsparse.HealSparseMap.read(fname)

        testing.assert_array_equal(
            sparse_map2.valid_pixels,
            valid_pixels
        )

        testing.assert_array_equal(
            sparse_map2.get_values_pix(valid_pixels),
            True
        )

        if test_pixels_read:
            for covpix_map in sparse_map.get_covpix_maps():
                covpix, = np.where(covpix_map.coverage_mask)

                covpix_map2 = healsparse.HealSparseMap.read(
                    fname,
                    pixels=covpix
                )

                testing.assert_array_equal(
                    covpix_map2.valid_pixels,
                    covpix_map.valid_pixels
                )

    def test_hdf5_writeread_bool(self):
        """Test writing and reading a bool map."""
        nside_coverage = 32
        nside_map = 64

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')
        fname = os.path.join(self.test_dir, 'healsparse_map.hdf5')

        sparse_map = healsparse.HealSparseMap.make_empty(
            nside_coverage, nside_map, bool
        )
        sparse_map[30000:30005] = True

        sparse_map.write(fname, format='hdf5')

        sparse_map2 = healsparse.HealSparseMap.read(fname)

        testing.assert_array_equal(
            sparse_map2[30000:30005],
            True
        )

        self.assertEqual(len(sparse_map2.valid_pixels), 5)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, True)


if __name__ == '__main__':
    unittest.main()