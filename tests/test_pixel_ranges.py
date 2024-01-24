import unittest
import numpy.testing as testing
import numpy as np
import hpgeom as hpg
import healsparse


class UpdateValuesPixelRangesTestCase(unittest.TestCase):
    def test_update_values_pixel_ranges(self):
        nside_coverage = 32
        nside_map = 256

        orig_threshold = healsparse.healSparseMap.PIXEL_RANGE_THRESHOLD

        # Monkey-patch PIXEL_RANGE_THRESHOLD for testing.
        healsparse.healSparseMap.PIXEL_RANGE_THRESHOLD = 0

        for dtype in [np.float64, np.int64, np.bool_]:
            if dtype == np.bool_:
                bit_packed = True
            else:
                bit_packed = False

            m1 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype, bit_packed=bit_packed)
            m2 = healsparse.HealSparseMap.make_empty_like(m1)

            pixel_ranges = np.zeros((3, 2), dtype=np.int64)
            pixel_ranges[0, :] = [10000, 20000]
            pixel_ranges[1, :] = [30000, 40000]
            pixel_ranges[2, :] = [100000, 110000]

            if dtype == np.float64:
                value = 1.0
            elif dtype == np.int64:
                value = 1
            else:
                value = True

            pixels = hpg.pixel_ranges_to_pixels(pixel_ranges)

            m1[pixel_ranges] = value
            m2[pixels] = value

            testing.assert_array_equal(m1.valid_pixels, pixels)
            testing.assert_array_equal(m1.valid_pixels, m2.valid_pixels)
            testing.assert_array_almost_equal(m1[pixels], m2[pixels])

            cov_mask = m1.coverage_mask

            def _none_checker(_m1, _m2, start, stop, cov_mask):
                pixel_ranges2 = np.zeros((1, 2), dtype=np.int64)
                pixel_ranges2[0, :] = [start, stop]

                pixels2 = hpg.pixel_ranges_to_pixels(pixel_ranges2)

                _m1[pixel_ranges2] = None
                _m2[pixels2] = None

                testing.assert_array_equal(_m1.coverage_mask, cov_mask)

                testing.assert_array_equal(_m1.valid_pixels, _m2.valid_pixels)
                testing.assert_array_almost_equal(_m1[pixels], _m2[pixels])

            # And try None value: broad range, no overlap.
            _none_checker(m1, m2, 200000, 250000, cov_mask)

            # None value: broad range, overlap.
            _none_checker(m1, m2, 105000, 200000, cov_mask)

            # None value: narrow range, overlap.
            _none_checker(m1, m2, 100000, 100010, cov_mask)

            # None value: narrow range, no overlap.
            _none_checker(m1, m2, 200000, 200010, cov_mask)

        healsparse.healSparseMap.PIXEL_RANGE_THRESHOLD = orig_threshold

    def test_update_values_pixel_ranges_empty(self):
        nside_coverage = 32
        nside_map = 256

        orig_threshold = healsparse.healSparseMap.PIXEL_RANGE_THRESHOLD

        # Monkey-patch PIXEL_RANGE_THRESHOLD for testing.
        healsparse.healSparseMap.PIXEL_RANGE_THRESHOLD = 0

        m1 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.float64)

        pixel_ranges = np.zeros((0, 2), dtype=np.int64)

        m1[pixel_ranges] = 1.0

        self.assertEqual(m1.n_valid, 0)

        healsparse.healSparseMap.PIXEL_RANGE_THRESHOLD = orig_threshold

    def test_update_values_pixel_ranges_or(self):
        nside_coverage = 32
        nside_map = 256

        orig_threshold = healsparse.healSparseMap.PIXEL_RANGE_THRESHOLD

        # Monkey-patch PIXEL_RANGE_THRESHOLD for testing.
        healsparse.healSparseMap.PIXEL_RANGE_THRESHOLD = 0

        for dtype in [np.int64, np.bool_]:
            if dtype == np.bool_:
                bit_packed = True
                sentinel = False
                value = True
                or_value = True
            else:
                bit_packed = False
                sentinel = 0
                value = 3
                or_value = 1

            m1 = healsparse.HealSparseMap.make_empty(
                nside_coverage,
                nside_map,
                dtype,
                sentinel=sentinel,
                bit_packed=bit_packed,
            )
            m2 = healsparse.HealSparseMap.make_empty_like(m1)

            m1[10000: 20000] = value
            m2[10000: 20000] = value

            pixel_ranges = np.zeros((1, 2), dtype=np.int64)
            pixel_ranges[0, :] = [15000, 25000]

            pixels = hpg.pixel_ranges_to_pixels(pixel_ranges)

            m1.update_values_pix(pixel_ranges, or_value, operation="or")
            m2.update_values_pix(pixels, or_value, operation="or")

            testing.assert_array_equal(m1.valid_pixels, m2.valid_pixels)
            testing.assert_array_equal(m1[m1.valid_pixels], m2[m2.valid_pixels])

        healsparse.healSparseMap.PIXEL_RANGE_THRESHOLD = orig_threshold

    def test_update_values_pixel_ranges_and(self):
        nside_coverage = 32
        nside_map = 256

        orig_threshold = healsparse.healSparseMap.PIXEL_RANGE_THRESHOLD

        # Monkey-patch PIXEL_RANGE_THRESHOLD for testing.
        healsparse.healSparseMap.PIXEL_RANGE_THRESHOLD = 0

        for dtype in [np.int64, np.bool_]:
            if dtype == np.bool_:
                bit_packed = True
                sentinel = False
                value = True
                and_value = True
            else:
                bit_packed = False
                sentinel = 0
                value = 3
                and_value = 1

            m1 = healsparse.HealSparseMap.make_empty(
                nside_coverage,
                nside_map,
                dtype,
                sentinel=sentinel,
                bit_packed=bit_packed,
            )
            m2 = healsparse.HealSparseMap.make_empty_like(m1)

            m1[10000: 20000] = value
            m2[10000: 20000] = value

            pixel_ranges = np.zeros((1, 2), dtype=np.int64)
            pixel_ranges[0, :] = [15000, 25000]

            pixels = hpg.pixel_ranges_to_pixels(pixel_ranges)

            m1.update_values_pix(pixel_ranges, and_value, operation="and")
            m2.update_values_pix(pixels, and_value, operation="and")

            testing.assert_array_equal(m1.valid_pixels, m2.valid_pixels)
            testing.assert_array_equal(m1[m1.valid_pixels], m2[m2.valid_pixels])

        healsparse.healSparseMap.PIXEL_RANGE_THRESHOLD = orig_threshold

    def test_update_values_pixel_ranges_add(self):
        nside_coverage = 32
        nside_map = 256

        orig_threshold = healsparse.healSparseMap.PIXEL_RANGE_THRESHOLD

        # Monkey-patch PIXEL_RANGE_THRESHOLD for testing.
        healsparse.healSparseMap.PIXEL_RANGE_THRESHOLD = 0

        for dtype in [np.float64, np.int64]:
            if dtype == np.float64:
                value = 1.0
                add_value = 2.0
            else:
                value = 1
                add_value = 2

            m1 = healsparse.HealSparseMap.make_empty(
                nside_coverage,
                nside_map,
                dtype,
            )
            m2 = healsparse.HealSparseMap.make_empty_like(m1)

            m1[10000: 20000] = value
            m2[10000: 20000] = value

            pixel_ranges = np.zeros((1, 2), dtype=np.int64)
            pixel_ranges[0, :] = [15000, 25000]

            pixels = hpg.pixel_ranges_to_pixels(pixel_ranges)

            m1.update_values_pix(pixel_ranges, add_value, operation="add")
            m2.update_values_pix(pixels, add_value, operation="add")

            testing.assert_array_equal(m1.valid_pixels, m2.valid_pixels)
            testing.assert_array_equal(m1[m1.valid_pixels], m2[m2.valid_pixels])

        healsparse.healSparseMap.PIXEL_RANGE_THRESHOLD = orig_threshold

    def test_set_bits_pix_pixel_ranges(self):
        nside_coverage = 32
        nside_map = 256

        orig_threshold = healsparse.healSparseMap.PIXEL_RANGE_THRESHOLD

        # Monkey-patch PIXEL_RANGE_THRESHOLD for testing.
        healsparse.healSparseMap.PIXEL_RANGE_THRESHOLD = 0

        m1 = healsparse.HealSparseMap.make_empty(
            nside_coverage,
            nside_map,
            healsparse.WIDE_MASK,
            wide_mask_maxbits=32,
        )
        m2 = healsparse.HealSparseMap.make_empty_like(m1)

        pixel_ranges = np.zeros((1, 2), dtype=np.int64)
        pixel_ranges[0, :] = [10000, 20000]

        pixels = hpg.pixel_ranges_to_pixels(pixel_ranges)

        m1.set_bits_pix(pixel_ranges, [0, 5, 20])
        m2.set_bits_pix(pixels, [0, 5, 20])

        testing.assert_array_equal(m1.valid_pixels, m2.valid_pixels)
        testing.assert_array_equal(m1[m1.valid_pixels], m2[m2.valid_pixels])

        healsparse.healSparseMap.PIXEL_RANGE_THRESHOLD = orig_threshold


if __name__ == '__main__':
    unittest.main()
