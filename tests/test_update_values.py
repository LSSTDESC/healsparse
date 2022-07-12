import unittest
import numpy.testing as testing
import numpy as np
import hpgeom as hpg
import healsparse


class UpdateValuesTestCase(unittest.TestCase):
    def test_update_values_inorder(self):
        """
        Test doing update_values, in coarse pixel order.
        """
        nside_coverage = 32
        nside_map = 64
        dtype = np.float64

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype)

        nfine_per_cov = 2**sparse_map._cov_map.bit_shift

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

    def test_update_values_outoforder(self):
        """
        Test doing updateValues, out of order.
        """

        nside_coverage = 32
        nside_map = 64
        dtype = np.float64

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype)

        nfine_per_cov = 2**sparse_map._cov_map.bit_shift

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

    def test_update_values_nonunique(self):
        """
        Test doing update_values with non-unique pixels.
        """
        nside_coverage = 32
        nside_map = 64
        dtype = np.float64

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype)

        pixels = np.array([0, 1, 5, 10, 0])

        self.assertRaises(ValueError, sparse_map.update_values_pix, pixels, 0.0)
        self.assertRaises(ValueError, sparse_map.__setitem__, pixels, 0.0)

    def test_update_values_or(self):
        """
        Test doing update_values with or operation.
        """
        nside_coverage = 32
        nside_map = 64
        dtype = np.int32

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype, sentinel=0)

        # Check with new unique pixels
        pixels = np.arange(4)
        values = np.array([2**0, 2**1, 2**2, 2**4], dtype=dtype)
        sparse_map.update_values_pix(pixels, values, operation='or')

        testing.assert_array_equal(sparse_map[pixels], values)

        # Check with pre-existing unique pixels
        values2 = np.array([2**1, 2**2, 2**3, 2**4], dtype=dtype)
        sparse_map.update_values_pix(pixels, values2, operation='or')

        testing.assert_array_equal(sparse_map[pixels],
                                   values | values2)

        # Check with new non-unique pixels
        pixels = np.array([100, 101, 102, 100])
        values = np.array([2**0, 2**1, 2**2, 2**4], dtype=dtype)
        sparse_map.update_values_pix(pixels, values, operation='or')

        testing.assert_array_equal(sparse_map[pixels],
                                   np.array([2**0 | 2**4, 2**1, 2**2, 2**0 | 2**4]))

        # Check with pre-existing non-unique pixels
        values = np.array([2**1, 2**2, 2**3, 2**5], dtype=dtype)
        sparse_map.update_values_pix(pixels, values, operation='or')

        testing.assert_array_equal(sparse_map[pixels],
                                   np.array([2**0 | 2**4 | 2**1 | 2**5,
                                             2**1 | 2**2,
                                             2**2 | 2**3,
                                             2**0 | 2**4 | 2**1 | 2**5]))

    def test_update_values_and(self):
        """
        Test doing update_values with and operation.
        """
        nside_coverage = 32
        nside_map = 64
        dtype = np.int32

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype, sentinel=0)

        # Check with new unique pixels
        pixels = np.arange(4)
        values = np.array([2**0, 2**1, 2**2, 2**4], dtype=dtype)

        sparse_map.update_values_pix(pixels, values, operation='and')
        testing.assert_array_equal(sparse_map[pixels], values*0)

        # Check with pre-existing unique pixels
        sparse_map[pixels] = values
        sparse_map.update_values_pix(pixels, values, operation='and')
        testing.assert_array_equal(sparse_map[pixels], values)

        # Check with new non-unique pixels
        pixels = np.array([100, 101, 102, 100])
        values = np.array([2**0, 2**1, 2**2, 2**4], dtype=dtype)

        sparse_map.update_values_pix(pixels, values, operation='and')
        testing.assert_array_equal(sparse_map[pixels], values*0)

        # Check with pre-existing non-unique pixels
        sparse_map[100] = 2**0 | 2**4
        sparse_map[101] = 2**1
        sparse_map[102] = 2**2

        sparse_map.update_values_pix(pixels, values, operation='and')
        # The first and last will be 0 because we get anded sequentially.
        testing.assert_array_equal(sparse_map[pixels],
                                   [0, 2**1, 2**2, 0])

    def test_update_values_pos(self):
        """
        Test doing update_values with positions (unique and non-unique).
        """
        nside_coverage = 32
        nside_map = 64
        dtype = np.float64

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype)

        pixels = np.array([0, 1, 5, 10, 20])
        ra, dec = hpg.pixel_to_angle(nside_map, pixels)

        sparse_map.update_values_pos(ra, dec, 0.0)
        testing.assert_array_almost_equal(sparse_map[pixels], 0.0)

        # Test non-unique raise
        pixels = np.array([0, 1, 5, 10, 0])
        ra, dec = hpg.pixel_to_angle(nside_map, pixels)
        self.assertRaises(ValueError, sparse_map.update_values_pos, ra, dec, 0.0)


if __name__ == '__main__':
    unittest.main()
