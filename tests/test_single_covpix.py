import unittest
import numpy.testing as testing
import numpy as np
import healsparse
from healsparse import WIDE_MASK


class SingleCovpixTestCase(unittest.TestCase):
    def test_get_single_pixel_in_map(self):
        """
        Test getting a single coverage pixel map that is in the map.
        """
        nside_sparse = 1024
        nside_coverage = 32

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage,
                                                         nside_sparse,
                                                         np.float32)
        pixels = np.array([1000, 100000, 10000000])
        sparse_map[pixels] = 1.0

        cov_pixels, = np.where(sparse_map.coverage_mask)

        for i, cov_pixel in enumerate(cov_pixels):
            sub_map = sparse_map.get_single_covpix_map(cov_pixel)
            sub_cov_pixels, = np.where(sub_map.coverage_mask)
            self.assertEqual(len(sub_cov_pixels), 1)
            self.assertEqual(sub_cov_pixels[0], cov_pixel)
            self.assertEqual(sub_map.n_valid, 1)
            testing.assert_array_equal(sub_map.valid_pixels, pixels[i])
            testing.assert_almost_equal(sub_map[pixels[i]], 1.0)

            # Test getting the cov_pixel valid_pixels from the full map.
            sub_valid_pixels = sparse_map.valid_pixels_single_covpix(cov_pixel)
            self.assertEqual(len(sub_valid_pixels), 1)
            testing.assert_array_equal(sub_valid_pixels, pixels[i])

    def test_get_single_pixel_in_map_recarray(self):
        """
        Test getting a single coverage pixel in a recarray map.
        """
        nside_sparse = 1024
        nside_coverage = 32
        dtype = [('a', 'f4'),
                 ('b', 'i4')]

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage,
                                                         nside_sparse,
                                                         dtype,
                                                         primary='a')
        pixels = np.array([1000, 100000, 10000000])
        value = np.ones(1, dtype=dtype)
        sparse_map[pixels] = value

        cov_pixels, = np.where(sparse_map.coverage_mask)

        for i, cov_pixel in enumerate(cov_pixels):
            sub_map = sparse_map.get_single_covpix_map(cov_pixel)
            sub_cov_pixels, = np.where(sub_map.coverage_mask)
            self.assertEqual(len(sub_cov_pixels), 1)
            self.assertEqual(sub_cov_pixels[0], cov_pixel)
            self.assertEqual(sub_map.n_valid, 1)
            testing.assert_array_equal(sub_map.valid_pixels, pixels[i])
            testing.assert_almost_equal(sub_map[pixels[i]]['a'], 1.0)
            self.assertEqual(sub_map[pixels[i]]['b'], 1)

            # Test getting the cov_pixel valid_pixels from the full map.
            sub_valid_pixels = sparse_map.valid_pixels_single_covpix(cov_pixel)
            self.assertEqual(len(sub_valid_pixels), 1)
            testing.assert_array_equal(sub_valid_pixels, pixels[i])

    def test_get_single_pixel_in_map_wide(self):
        """
        Test getting a single coverage pixel in a wide mask map.
        """
        nside_sparse = 1024
        nside_coverage = 32

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage,
                                                         nside_sparse,
                                                         WIDE_MASK,
                                                         wide_mask_maxbits=15)

        pixels = np.array([1000, 100000, 10000000])
        sparse_map.set_bits_pix(pixels, [4])
        sparse_map.set_bits_pix(pixels, [13])

        cov_pixels, = np.where(sparse_map.coverage_mask)

        for i, cov_pixel in enumerate(cov_pixels):
            sub_map = sparse_map.get_single_covpix_map(cov_pixel)
            sub_cov_pixels, = np.where(sub_map.coverage_mask)
            self.assertEqual(len(sub_cov_pixels), 1)
            self.assertEqual(sub_cov_pixels[0], cov_pixel)
            self.assertEqual(sub_map.n_valid, 1)
            testing.assert_array_equal(sub_map.valid_pixels, pixels[i])
            self.assertEqual(sub_map.check_bits_pix(pixels[i], [4]), True)
            self.assertEqual(sub_map.check_bits_pix(pixels[i], [13]), True)

            # Test getting the cov_pixel valid_pixels from the full map.
            sub_valid_pixels = sparse_map.valid_pixels_single_covpix(cov_pixel)
            self.assertEqual(len(sub_valid_pixels), 1)
            testing.assert_array_equal(sub_valid_pixels, pixels[i])

    def test_get_single_pixel_not_in_map(self):
        """
        Test getting a single coverage pixel map that is not in the map.
        """
        nside_sparse = 1024
        nside_coverage = 32

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage,
                                                         nside_sparse,
                                                         np.float32)
        pixels = np.array([1000, 100000, 10000000])
        sparse_map[pixels] = 1.0

        sub_map = sparse_map.get_single_covpix_map(40)
        sub_cov_pixels, = np.where(sub_map.coverage_mask)
        self.assertEqual(len(sub_cov_pixels), 0)
        self.assertEqual(sub_map.n_valid, 0)
        self.assertEqual(len(sub_map.valid_pixels), 0)

        sub_valid_pixels = sparse_map.valid_pixels_single_covpix(40)
        self.assertEqual(len(sub_valid_pixels), 0)

    def test_single_pixel_generator(self):
        """
        Test iterating over all of the single coverage pixel maps.
        """
        nside_sparse = 1024
        nside_coverage = 32

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage,
                                                         nside_sparse,
                                                         np.float32)
        pixels = np.array([1000, 100000, 10000000])
        sparse_map[pixels] = 1.0

        cov_pixels, = np.where(sparse_map.coverage_mask)

        for i, sub_map in enumerate(sparse_map.get_covpix_maps()):
            sub_cov_pixels, = np.where(sub_map.coverage_mask)

            self.assertEqual(len(sub_cov_pixels), 1)
            self.assertEqual(sub_cov_pixels[0], cov_pixels[i])
            self.assertEqual(sub_map.n_valid, 1)
            testing.assert_array_equal(sub_map.valid_pixels, pixels[i])
            testing.assert_almost_equal(sub_map[pixels[i]], 1.0)

        for i, sub_valid_pixels in enumerate(sparse_map.iter_valid_pixels_by_covpix()):
            self.assertEqual(len(sub_valid_pixels), 1)
            testing.assert_array_equal(sub_valid_pixels, pixels[i])


if __name__ == '__main__':
    unittest.main()
