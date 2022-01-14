from __future__ import division, absolute_import, print_function

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp

import healsparse


class LookupTestCase(unittest.TestCase):
    def test_lookup(self):
        """
        Test lookup functionality
        """
        np.random.seed(12345)

        nside_coverage = 32
        nside_map = 1024

        full_map = np.zeros(hp.nside2npix(nside_map)) + hp.UNSEEN
        full_map[0: 200000] = np.random.random(size=200000)

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage)

        n_rand = 100000
        ra = np.random.random(n_rand) * 360.0
        dec = np.random.random(n_rand) * 180.0 - 90.0

        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        ipnest = hp.ang2pix(nside_map, theta, phi, nest=True)

        test_values = full_map[ipnest]

        # Test the pixel lookup
        comp_values = sparse_map.get_values_pix(ipnest)
        testing.assert_almost_equal(comp_values, test_values)

        # Test pixel lookup (valid pixels)
        # Note that this tests all the downstream functions
        valid_mask = sparse_map.get_values_pix(ipnest, valid_mask=True)
        testing.assert_equal(valid_mask, comp_values > hp.UNSEEN)

        # Test pixel lookup (ring)
        ipring = hp.nest2ring(nside_map, ipnest)
        comp_values = sparse_map.get_values_pix(ipring, nest=False)
        testing.assert_almost_equal(comp_values, test_values)

        # Test pixel lookup (higher nside)
        comp_values = sparse_map.get_values_pix(
            hp.ang2pix(4096, ra, dec, lonlat=True, nest=True),
            nside=4096
        )
        testing.assert_almost_equal(comp_values, test_values)

        # Test pixel lookup (lower nside)
        lowres_pix = hp.ang2pix(256, ra, dec, lonlat=True, nest=True)
        self.assertRaises(ValueError, sparse_map.get_values_pix, lowres_pix, nside=256)

        # Test the theta/phi lookup
        comp_values = sparse_map.get_values_pos(theta, phi, lonlat=False)
        testing.assert_almost_equal(comp_values, test_values)

        # Test the ra/dec lookup
        comp_values = sparse_map.get_values_pos(ra, dec, lonlat=True)
        testing.assert_almost_equal(comp_values, test_values)

        # Test the list of valid pixels
        valid_pixels = sparse_map.valid_pixels
        testing.assert_equal(valid_pixels, np.where(full_map > hp.UNSEEN)[0])

        # Test the position of valid pixels
        ra_sp, dec_sp = sparse_map.valid_pixels_pos(lonlat=True)
        _ra_sp, _dec_sp = hp.pix2ang(nside_map, np.where(full_map > hp.UNSEEN)[0], lonlat=True, nest=True)
        testing.assert_equal(ra_sp, _ra_sp)
        testing.assert_equal(dec_sp, _dec_sp)

        # Test position of valid pixels and valid pixels
        valid_pixels, ra_sp, dec_sp = sparse_map.valid_pixels_pos(lonlat=True,
                                                                  return_pixels=True)
        _ra_sp, _dec_sp = hp.pix2ang(nside_map, np.where(full_map > hp.UNSEEN)[0], lonlat=True, nest=True)
        testing.assert_equal(ra_sp, _ra_sp)
        testing.assert_equal(dec_sp, _dec_sp)
        testing.assert_equal(valid_pixels, np.where(full_map > hp.UNSEEN)[0])


if __name__ == '__main__':
    unittest.main()
