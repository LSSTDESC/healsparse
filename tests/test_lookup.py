import unittest
import numpy.testing as testing
import numpy as np
import hpgeom as hpg

import healsparse


class LookupTestCase(unittest.TestCase):
    def test_lookup(self):
        """
        Test lookup functionality
        """
        np.random.seed(12345)

        nside_coverage = 32
        nside_map = 1024

        full_map = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        full_map[0: 200000] = np.random.random(size=200000)

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage)

        n_rand = 100000
        ra = np.random.random(n_rand) * 360.0
        dec = np.random.random(n_rand) * 180.0 - 90.0

        theta, phi = hpg.lonlat_to_thetaphi(ra, dec)

        ipnest = hpg.angle_to_pixel(nside_map, ra, dec)

        test_values = full_map[ipnest]

        # Test the pixel lookup
        comp_values = sparse_map.get_values_pix(ipnest)
        testing.assert_array_almost_equal(comp_values, test_values)

        # Test pixel lookup (valid pixels)
        # Note that this tests all the downstream functions
        valid_mask = sparse_map.get_values_pix(ipnest, valid_mask=True)
        testing.assert_array_equal(valid_mask, comp_values > hpg.UNSEEN)

        # Test pixel lookup (ring)
        ipring = hpg.nest_to_ring(nside_map, ipnest)
        comp_values = sparse_map.get_values_pix(ipring, nest=False)
        testing.assert_array_almost_equal(comp_values, test_values)

        # Test pixel lookup (higher nside)
        comp_values = sparse_map.get_values_pix(
            hpg.angle_to_pixel(4096, ra, dec),
            nside=4096
        )
        testing.assert_array_almost_equal(comp_values, test_values)

        # Test pixel lookup (lower nside)
        lowres_pix = hpg.angle_to_pixel(256, ra, dec)
        self.assertRaises(ValueError, sparse_map.get_values_pix, lowres_pix, nside=256)

        # Test the theta/phi lookup
        comp_values = sparse_map.get_values_pos(theta, phi, lonlat=False)
        testing.assert_array_almost_equal(comp_values, test_values)

        # Test the ra/dec lookup
        comp_values = sparse_map.get_values_pos(ra, dec, lonlat=True)
        testing.assert_array_almost_equal(comp_values, test_values)

        # Test the list of valid pixels
        valid_pixels = sparse_map.valid_pixels
        testing.assert_array_equal(valid_pixels, np.where(full_map > hpg.UNSEEN)[0])

        # Test the position of valid pixels
        ra_sp, dec_sp = sparse_map.valid_pixels_pos(lonlat=True)
        _ra_sp, _dec_sp = hpg.pixel_to_angle(nside_map, np.where(full_map > hpg.UNSEEN)[0])
        testing.assert_array_almost_equal(ra_sp, _ra_sp)
        testing.assert_array_almost_equal(dec_sp, _dec_sp)

        # Test position of valid pixels and valid pixels
        valid_pixels, ra_sp, dec_sp = sparse_map.valid_pixels_pos(lonlat=True,
                                                                  return_pixels=True)
        _ra_sp, _dec_sp = hpg.pixel_to_angle(nside_map, np.where(full_map > hpg.UNSEEN)[0])
        testing.assert_array_almost_equal(ra_sp, _ra_sp)
        testing.assert_array_almost_equal(dec_sp, _dec_sp)
        testing.assert_array_equal(valid_pixels, np.where(full_map > hpg.UNSEEN)[0])


if __name__ == '__main__':
    unittest.main()
