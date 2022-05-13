import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
from numpy import random
import tempfile
import shutil
import os

import healsparse


class HealpixIoTestCase(unittest.TestCase):
    def test_healpix_implicit_read(self):
        """Test reading healpix full (implicit) maps."""
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        n_rand = 1000
        ra = np.random.random(n_rand) * 360.0
        dec = np.random.random(n_rand) * 180.0 - 90.0

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        # Generate a random map
        full_map = np.zeros(hp.nside2npix(nside_map)) + hp.UNSEEN
        full_map[0: 20000] = np.random.random(size=20000)

        ipnest = hp.ang2pix(nside_map, ra, dec, nest=True, lonlat=True)

        test_values = full_map[ipnest]

        filename = os.path.join(self.test_dir, 'healpix_map_ring.fits')

        full_map_ring = hp.reorder(full_map, n2r=True)
        hp.write_map(filename, full_map_ring, dtype=np.float64)

        # Read it with healsparse
        sparse_map = healsparse.HealSparseMap.read(filename, nside_coverage=nside_coverage)

        # Check that we can do a basic lookup
        testing.assert_almost_equal(sparse_map.get_values_pix(ipnest), test_values)

        # Save map to healpy in nest
        filename = os.path.join(self.test_dir, 'healpix_map_nest.fits')
        hp.write_map(filename, full_map, dtype=np.float64, nest=True)

        # Read it with healsparse
        sparse_map = healsparse.HealSparseMap.read(filename, nside_coverage=nside_coverage)

        # Check that we can do a basic lookup
        testing.assert_almost_equal(sparse_map.get_values_pix(ipnest), test_values)

        # Test that we get an exception if reading without nside_coverage
        with self.assertRaises(RuntimeError):
            sparse_map = healsparse.HealSparseMap.read(filename)

    def test_healpix_explicit_read(self):
        """Test reading healpix partial (explicit) maps."""
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        n_rand = 1000
        ra = np.random.random(n_rand) * 360.0
        dec = np.random.random(n_rand) * 180.0 - 90.0

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        # Generate a random map
        full_map = np.zeros(hp.nside2npix(nside_map)) + hp.UNSEEN
        full_map[0: 20000] = np.random.random(size=20000)

        ipnest = hp.ang2pix(nside_map, ra, dec, nest=True, lonlat=True)

        test_values = full_map[ipnest]

        filename = os.path.join(self.test_dir, 'healpix_map_ring_explicit.fits')

        full_map_ring = hp.reorder(full_map, n2r=True)
        hp.write_map(filename, full_map_ring, dtype=np.float64, partial=True)

        # Read it with healsparse
        sparse_map = healsparse.HealSparseMap.read(filename, nside_coverage=nside_coverage)

        # Check that we can do a basic lookup
        testing.assert_almost_equal(sparse_map.get_values_pix(ipnest), test_values)

        filename = os.path.join(self.test_dir, 'healpix_map_nest_explicit.fits')
        hp.write_map(filename, full_map, dtype=np.float64, nest=True, partial=True)

        # Read it with healsparse
        sparse_map = healsparse.HealSparseMap.read(filename, nside_coverage=nside_coverage)

        # Check that we can do a basic lookup
        testing.assert_almost_equal(sparse_map.get_values_pix(ipnest), test_values)

    def test_healpix_explicit_write(self):
        """Test writing healpix partial (explicit) maps (floating point)."""
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        n_rand = 1000
        ra = np.random.random(n_rand) * 360.0
        dec = np.random.random(n_rand) * 180.0 - 90.0

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        # Generate a random map
        full_map = np.zeros(hp.nside2npix(nside_map)) + hp.UNSEEN
        full_map[0: 20000] = np.random.random(size=20000)

        ipnest = hp.ang2pix(nside_map, ra, dec, nest=True, lonlat=True)

        test_values = full_map[ipnest]

        filename = os.path.join(self.test_dir, 'healsparse_healpix_partial_map.fits')

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage, nest=True)
        sparse_map.write(filename, format='healpix')

        # Read in with healsparse and make sure it is the same.
        sparse_map2 = healsparse.HealSparseMap.read(filename, nside_coverage=nside_coverage)

        np.testing.assert_array_equal(sparse_map2.valid_pixels, sparse_map.valid_pixels)
        testing.assert_array_almost_equal(sparse_map2.get_values_pix(ipnest), test_values)

        # Read in with healpy and make sure it is the same.
        full_map2 = hp.read_map(filename, nest=True)
        testing.assert_array_equal((full_map2 > hp.UNSEEN).nonzero()[0], sparse_map.valid_pixels)
        testing.assert_array_almost_equal(full_map2[ipnest], test_values)

    def test_healpix_explicit_int_write(self):
        """Test writing healpix partial (explicit) maps (integer)."""
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        n_rand = 1000
        ra = np.random.random(n_rand) * 360.0
        dec = np.random.random(n_rand) * 180.0 - 90.0

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        # Generate a map
        full_map = np.zeros(hp.nside2npix(nside_map), dtype=np.int32)
        full_map[0: 10000] = 4
        full_map[20000: 30000] = 5

        ipnest = hp.ang2pix(nside_map, ra, dec, nest=True, lonlat=True)
        test_values = full_map[ipnest]

        filename = os.path.join(self.test_dir, 'healsparse_healpix_int_partial_map.fits')

        sparse_map = healsparse.HealSparseMap(
            healpix_map=full_map,
            nside_coverage=nside_coverage,
            nest=True,
            sentinel=0
        )
        with self.assertWarns(UserWarning):
            sparse_map.write(filename, format='healpix')

        # Read in with healsparse and make sure it is the same.
        sparse_map2 = healsparse.HealSparseMap.read(filename, nside_coverage=nside_coverage)

        np.testing.assert_array_equal(sparse_map2.valid_pixels, sparse_map.valid_pixels)
        testing.assert_almost_equal(sparse_map2.get_values_pix(ipnest), test_values)

        # Read in with healpy and make sure it is the same.
        full_map2 = hp.read_map(filename, nest=True)
        testing.assert_array_equal((full_map2 > hp.UNSEEN).nonzero()[0], sparse_map.valid_pixels)

        # healpy will convert all the BAD_DATA to UNSEEN
        good, = (test_values > 0).nonzero()
        bad, = (test_values == 0).nonzero()
        testing.assert_array_equal(full_map2[ipnest[good]], test_values[good])
        testing.assert_array_almost_equal(full_map2[ipnest[bad]], hp.UNSEEN)

    def test_healpix_recarray_write(self):
        """Test that the proper error is raised if you try to persist via healpix format."""
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        nside_coverage = 32
        nside_map = 64

        filename = os.path.join(self.test_dir, 'test_file.fits')

        dtype = [('a', 'f4'), ('b', 'i2')]
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype, primary='a')
        with self.assertRaises(NotImplementedError):
            sparse_map.write(filename, format='healpix')

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__ == '__main__':
    unittest.main()
