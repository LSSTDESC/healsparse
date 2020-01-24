from __future__ import division, absolute_import, print_function

import unittest
import numpy.testing as testing
import numpy as np
import healsparse


class UpdateValuesTestCase(unittest.TestCase):
    def test_updateValues_inorder(self):
        """
        Test doing updateValues, in coarse pixel order.
        """
        nside_coverage = 32
        nside_map = 64
        dtype = np.float64

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype)

        nfine_per_cov = 2**sparse_map._bit_shift

        test_pix = np.arange(nfine_per_cov) + nfine_per_cov * 10
        test_values = np.zeros(nfine_per_cov)

        sparse_map.update_values_pix(test_pix, test_values)
        testing.assert_almost_equal(sparse_map.get_values_pix(test_pix), test_values)

        valid_pixels = sparse_map.valid_pixels
        testing.assert_equal(valid_pixels, test_pix)

        test_pix2 = np.arange(nfine_per_cov) + nfine_per_cov * 16
        test_values2 = np.zeros(nfine_per_cov) + 100

        sparse_map.update_values_pix(test_pix2, test_values2)
        testing.assert_almost_equal(sparse_map.get_values_pix(test_pix), test_values)
        testing.assert_almost_equal(sparse_map.get_values_pix(test_pix2), test_values2)

        valid_pixels = sparse_map.valid_pixels
        testing.assert_equal(np.sort(valid_pixels), np.sort(np.concatenate((test_pix, test_pix2))))

    def test_updateValues_outoforder(self):
        """
        Test doing updateValues, out of order.
        """

        nside_coverage = 32
        nside_map = 64
        dtype = np.float64

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype)

        nfine_per_cov = 2**sparse_map._bit_shift

        test_pix = np.arange(nfine_per_cov) + nfine_per_cov * 16
        test_values = np.zeros(nfine_per_cov)

        sparse_map.update_values_pix(test_pix, test_values)
        testing.assert_almost_equal(sparse_map.get_values_pix(test_pix), test_values)

        valid_pixels = sparse_map.valid_pixels
        testing.assert_equal(valid_pixels, test_pix)

        test_pix2 = np.arange(nfine_per_cov) + nfine_per_cov * 10
        test_values2 = np.zeros(nfine_per_cov) + 100

        sparse_map.update_values_pix(test_pix2, test_values2)
        testing.assert_almost_equal(sparse_map.get_values_pix(test_pix), test_values)
        testing.assert_almost_equal(sparse_map.get_values_pix(test_pix2), test_values2)

        valid_pixels = sparse_map.valid_pixels
        testing.assert_equal(np.sort(valid_pixels), np.sort(np.concatenate((test_pix, test_pix2))))


if __name__ == '__main__':
    unittest.main()
