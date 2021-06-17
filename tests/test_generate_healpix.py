from __future__ import division, absolute_import, print_function

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
from numpy import random
import healsparse


class GenerateHealpixMapTestCase(unittest.TestCase):
    def test_generate_healpix_map_single(self):
        """
        Test the generation of a healpix map from a sparse map for a single-value field
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        n_rand = 1000
        ra = np.random.random(n_rand) * 360.0
        dec = np.random.random(n_rand) * 180.0 - 90.0
        value = np.random.random(n_rand)

        # Create a HEALPix map
        healpix_map = np.zeros(hp.nside2npix(nside_map), dtype=np.float64) + hp.UNSEEN
        idx = hp.ang2pix(nside_map, np.pi/2 - np.radians(dec), np.radians(ra), nest=True)
        healpix_map[idx] = value
        # Create a HealSparseMap
        sparse_map = healsparse.HealSparseMap(nside_coverage=nside_coverage, healpix_map=healpix_map)
        hp_out = sparse_map.generate_healpix_map(nside=nside_map)
        testing.assert_almost_equal(healpix_map, hp_out)

        # Now check that it works specifying a different resolution
        nside_map2 = 32
        hp_out = sparse_map.generate_healpix_map(nside=nside_map2)
        # Let's compare with the original downgraded
        healpix_map = hp.ud_grade(healpix_map, nside_out=nside_map2, order_in='NESTED', order_out='NESTED')
        testing.assert_almost_equal(healpix_map, hp_out)

    def test_generate_healpix_map_recarray(self):
        """
        Testing the generation of a healpix map from recarray healsparsemap
        we also test the pixel and position lookup
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        n_rand = 1000
        ra = np.random.random(n_rand) * 360.0
        dec = np.random.random(n_rand) * 180.0 - 90.0
        value = np.random.random(n_rand)

        # Make sure our pixels are unique
        ipnest = hp.ang2pix(nside_map, ra, dec, lonlat=True, nest=True)
        _, uind = np.unique(ipnest, return_index=True)
        ra = ra[uind]
        dec = dec[uind]
        value = value[uind]

        # Create empty healpix map
        healpix_map = np.zeros(hp.nside2npix(nside_map), dtype='f4') + hp.UNSEEN
        healpix_map2 = np.zeros(hp.nside2npix(nside_map), dtype='f8') + hp.UNSEEN
        healpix_map[hp.ang2pix(nside_map, ra, dec, lonlat=True, nest=True)] = value
        healpix_map2[hp.ang2pix(nside_map, ra, dec, lonlat=True, nest=True)] = value

        # Create an empty map
        dtype = [('col1', 'f4'), ('col2', 'f8')]

        self.assertRaises(RuntimeError, healsparse.HealSparseMap.make_empty, nside_coverage, nside_map, dtype)
        # Generate empty map that will be updated
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype, primary='col1')
        pixel = hp.ang2pix(nside_map, ra, dec, nest=True, lonlat=True)
        values = np.zeros_like(pixel, dtype=dtype)
        values['col1'] = value
        values['col2'] = value
        # Update values works with the HEALPix-like indexing scheme
        sparse_map.update_values_pix(pixel, values)
        hp_out1 = sparse_map.generate_healpix_map(nside=nside_map, key='col1')
        hp_out2 = sparse_map.generate_healpix_map(nside=nside_map, key='col2')
        testing.assert_almost_equal(healpix_map, hp_out1)
        testing.assert_almost_equal(healpix_map2, hp_out2)

    def test_generate_healpix_map_int(self):
        """
        Testing the generation of a healpix map from an integer map
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.int64)
        pixel = np.arange(4000, 20000)
        pixel = np.delete(pixel, 15000)
        # Get a random list of integers
        values = np.random.poisson(size=pixel.size, lam=10)
        sparse_map.update_values_pix(pixel, values)

        hpmap = sparse_map.generate_healpix_map()

        ok, = np.where(hpmap > hp.UNSEEN)

        testing.assert_almost_equal(hpmap[ok], sparse_map.get_values_pix(ok).astype(np.float64))

    def test_generate_healpix_map_ring(self):
        """
        Test the generation of a healpixmap in ring type
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        n_rand = 1000
        ra = np.random.random(n_rand) * 360.0
        dec = np.random.random(n_rand) * 180.0 - 90.0
        value = np.random.random(n_rand)

        # Create a HEALPix map
        healpix_map = np.zeros(hp.nside2npix(nside_map), dtype=np.float64) + hp.UNSEEN
        idx = hp.ang2pix(nside_map, np.pi/2 - np.radians(dec), np.radians(ra), nest=True)
        healpix_map[idx] = value
        # Create a HealSparseMap
        sparse_map = healsparse.HealSparseMap(nside_coverage=nside_coverage, healpix_map=healpix_map)
        hp_out_ring = sparse_map.generate_healpix_map(nside=nside_map, nest=False)
        healpix_map_ring = hp.reorder(healpix_map, n2r=True)
        testing.assert_almost_equal(healpix_map_ring, hp_out_ring)


if __name__ == '__main__':
    unittest.main()
