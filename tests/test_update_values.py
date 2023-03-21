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

    def test_update_values_add(self):
        """
        Test doing update_values with the add operation.
        """
        nside_coverage = 32
        nside_map = 64
        dtype = np.float64

        for sentinel in [hpg.UNSEEN, 0.0]:
            for update_type in ['single', 'array']:
                sparse_map = healsparse.HealSparseMap.make_empty(
                    nside_coverage,
                    nside_map,
                    dtype,
                    sentinel=sentinel,
                )

                # Add a constant value
                test_pix = np.array([0, 0, 10, 10, 1000, 1000, 1000, 10000])
                if update_type == 'single':
                    sparse_map.update_values_pix(test_pix, 1.0, operation='add')
                else:
                    sparse_map.update_values_pix(test_pix, np.ones(test_pix.size), operation='add')

                testing.assert_array_equal(sparse_map.valid_pixels, np.unique(test_pix))
                testing.assert_array_equal(sparse_map[sparse_map.valid_pixels], [2.0, 2.0, 3.0, 1.0])

                # Try again with a constant value, with a few old and a few new pixels
                test_pix2 = np.array([0, 0, 1, 1, 1])
                if update_type == 'single':
                    sparse_map.update_values_pix(test_pix2, 2.0, operation='add')
                else:
                    sparse_map.update_values_pix(test_pix2, np.full(test_pix2.size, 2.0), operation='add')

                testing.assert_array_equal(
                    sparse_map.valid_pixels,
                    np.unique(np.concatenate((test_pix, test_pix2))),
                )
                testing.assert_array_equal(sparse_map[sparse_map.valid_pixels], [6.0, 6.0, 2.0, 3.0, 1.0])

                # And add some more with positions
                ra = np.array([10.0, 10.0])
                dec = np.array([70.0, 70.0])
                test_pix3 = hpg.angle_to_pixel(sparse_map.nside_sparse, ra, dec)

                sparse_map.update_values_pos(ra, dec, 3.0, operation='add')

                testing.assert_array_equal(
                    np.sort(sparse_map.valid_pixels),
                    np.unique(np.concatenate((test_pix, test_pix2, test_pix3))),
                )
                testing.assert_array_equal(
                    sparse_map[sparse_map.valid_pixels],
                    [6.0, 6.0, 2.0, 3.0, 1.0, 6.0],
                )

        # Test recarray raise
        dtype = [('col1', 'f8'), ('col2', 'f8')]
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype, primary='col1')

        with self.assertRaises(ValueError):
            sparse_map.update_values_pix(0, 1.0, operation='add')

        # Test None raise
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.float32)
        with self.assertRaises(ValueError):
            sparse_map.update_values_pix(0, None, operation='add')

    def test_update_values_pos(self):
        """
        Test doing update_values with positions (unique and non-unique).
        """
        nside_coverage = 32
        nside_map = 64
        dtype = np.float64

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype)

        pixels = np.array([0, 1, 5, 10, 20])
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype)
        ra, dec = hpg.pixel_to_angle(nside_map, pixels)

        sparse_map.update_values_pos(ra, dec, 0.0)
        testing.assert_array_almost_equal(sparse_map[pixels], 0.0)

        # Test non-unique raise
        pixels = np.array([0, 1, 5, 10, 0])
        ra, dec = hpg.pixel_to_angle(nside_map, pixels)
        self.assertRaises(ValueError, sparse_map.update_values_pos, ra, dec, 0.0)


if __name__ == '__main__':
    unittest.main()
