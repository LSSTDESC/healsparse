import unittest
import numpy.testing as testing
import numpy as np

import healsparse
from healsparse import WIDE_MASK


class EmptyPixelsTestCase(unittest.TestCase):
    def test_emptypixels_get_values(self):
        """
        Test getting an empty list via get_values_pix()
        """
        m = healsparse.HealSparseMap.make_empty(32, 512, np.float32)
        m[1000] = 1.0

        empty_arr = np.array([], dtype=m.dtype)

        testing.assert_array_equal(m.get_values_pix([]), empty_arr)
        testing.assert_array_equal(m.get_values_pix(np.array([], dtype=np.int64)), empty_arr)

    def test_emptypixels_get_values_widemask(self):
        """
        Test getting an empty list via get_values_pix() (wide mask).
        """
        m = healsparse.HealSparseMap.make_empty(32, 512, WIDE_MASK, wide_mask_maxbits=16)
        testing.assert_array_equal(m.get_values_pix([]), np.zeros((0, 2), dtype=np.uint8))

    def test_emptypixels_get_by_index(self):
        """
        Test getting an empty list by index
        """
        m = healsparse.HealSparseMap.make_empty(32, 512, np.float64)
        m[1000] = 1.0

        empty_arr = np.array([], dtype=m.dtype)

        testing.assert_array_equal(m[[]], empty_arr)
        testing.assert_array_equal(m[np.array([], dtype=np.int64)], empty_arr)

    def test_emptypixels_get_by_index_widemask(self):
        """
        Test getting an empty list by index (wide mask).
        """
        m = healsparse.HealSparseMap.make_empty(32, 512, WIDE_MASK, wide_mask_maxbits=16)
        testing.assert_array_equal(m[[]], np.zeros((0, 2), dtype=np.uint8))

    def test_emptypixels_get_by_slice(self):
        """
        Test getting an empty pixel list by slice
        """
        m = healsparse.HealSparseMap.make_empty(32, 512, np.float64)
        m[1000] = 1.0

        empty_arr = np.array([], dtype=m.dtype)

        testing.assert_array_equal(m[0: 0], empty_arr)

    def test_emptypixels_get_by_slice_widemask(self):
        """
        Test getting an empty pixel list by slice (wide mask).
        """
        m = healsparse.HealSparseMap.make_empty(32, 512, WIDE_MASK, wide_mask_maxbits=16)
        testing.assert_array_equal(m[0: 0], np.zeros((0, 2), dtype=np.uint8))

    def test_emptypixels_update_values(self):
        """
        Test setting an empty list with update_values_pix
        """
        m = healsparse.HealSparseMap.make_empty(32, 512, np.int64)
        m[1000] = 1

        m.update_values_pix([], np.array([], dtype=np.int64))
        self.assertEqual(m[1000], 1)
        self.assertEqual(len(m.valid_pixels), 1)

        m.update_values_pix(np.array([], dtype=np.int32), np.array([], dtype=np.int64))
        self.assertEqual(m[1000], 1)
        self.assertEqual(len(m.valid_pixels), 1)

        m.update_values_pix([], np.array([], dtype=np.int64))
        m.update_values_pix([], 5)

        self.assertWarns(UserWarning, m.update_values_pix, [], np.zeros(5, dtype=m.dtype))

        m.update_values_pos(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
        )

    def test_emptypixels_set_by_index(self):
        """
        Test setting an empty list by index
        """
        m = healsparse.HealSparseMap.make_empty(32, 512, np.int32)
        m[1000] = 1

        m[[]] = np.array([], dtype=np.int32)
        self.assertEqual(m[1000], 1)
        self.assertEqual(len(m.valid_pixels), 1)

        m[np.array([], dtype=np.int32)] = np.array([], dtype=np.int32)
        self.assertEqual(m[1000], 1)
        self.assertEqual(len(m.valid_pixels), 1)

        m[[]] = np.array([], dtype=np.int32)
        m[[]] = 0

        self.assertWarns(UserWarning, m.__setitem__, [], np.zeros(5, dtype=np.int32))


if __name__ == '__main__':
    unittest.main()
