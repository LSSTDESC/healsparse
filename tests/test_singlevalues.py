import unittest
import numpy.testing as testing
import numpy as np

import healsparse


class SingleValuesTestCase(unittest.TestCase):
    def test_singlevalue_int(self):
        """
        Test assigning single integer values
        """
        m = healsparse.HealSparseMap.make_empty(32, 512, np.int32)

        # Test plain integer
        m.update_values_pix(np.arange(100), 0)
        testing.assert_array_equal(m[0: 100], np.zeros(100, dtype=np.int32))

        m[50] = 5
        self.assertEqual(m[50], 5)

        # Test ndarray, length 1
        m.update_values_pix(np.arange(100), np.ones(1, dtype=np.int32))
        testing.assert_array_equal(m[0: 100], np.ones(100, dtype=np.int32))

        m[50] = 0
        self.assertEqual(m[50], 0)

        # Test numpy integer
        m.update_values_pix(np.arange(100), np.ones(1, dtype=np.int32)[0])
        testing.assert_array_equal(m[0: 100], np.ones(100, dtype=np.int32))

        m[50] = np.zeros(1, dtype=np.int32)[0]
        self.assertEqual(m[50], 0)

        # Test None
        m.update_values_pix(np.arange(25), None)
        testing.assert_array_equal(m[0: 25], m.sentinel)

        m[50] = None
        self.assertEqual(m[50], m.sentinel)

        # Test None (no append)
        covmask_orig = m.coverage_mask
        m[1000] = None
        testing.assert_array_equal(m.coverage_mask, covmask_orig)

        # Test floating point
        self.assertRaises(ValueError, m.update_values_pix, np.arange(100),
                          1.0)

        # Test floating point ndarray, length 1
        self.assertRaises(ValueError, m.update_values_pix, np.arange(100),
                          np.ones(1, dtype=np.float32))

        # Test numpy floating point
        self.assertRaises(ValueError, m.update_values_pix, np.arange(100),
                          np.ones(1, dtype=np.float32)[0])

    def test_singlevalue_float(self):
        """
        Test assigning single floating point values
        """
        m = healsparse.HealSparseMap.make_empty(32, 512, np.float32)

        # Test plain float
        m.update_values_pix(np.arange(100), 0.0)
        testing.assert_array_equal(m[0: 100], np.zeros(100, dtype=np.float32))

        m[50] = 5.0
        self.assertEqual(m[50], 5.0)

        # Test ndarray, length 1
        m.update_values_pix(np.arange(100), np.ones(1, dtype=np.float32))
        testing.assert_array_equal(m[0: 100], np.ones(100, dtype=np.float32))

        m[50] = 0.0
        self.assertEqual(m[50], 0.0)

        # Test numpy float
        m.update_values_pix(np.arange(100), np.ones(1, dtype=np.float32)[0])
        testing.assert_array_equal(m[0: 100], np.ones(100, dtype=np.float32))

        m[50] = 0.0
        self.assertEqual(m[50], 0.0)

        # Test None
        m.update_values_pix(np.arange(100), 1.0)
        m.update_values_pix(np.arange(25), None)
        testing.assert_array_equal(m[0: 25], m.sentinel)

        m[50] = None
        self.assertEqual(m[50], m.sentinel)

        # Test None (no append)
        covmask_orig = m.coverage_mask
        m[1000] = None
        testing.assert_array_equal(m.coverage_mask, covmask_orig)

        # Test int
        self.assertRaises(ValueError, m.update_values_pix, np.arange(100),
                          1)

        # Test int ndarray, length 1
        self.assertRaises(ValueError, m.update_values_pix, np.arange(100),
                          np.ones(1, dtype=np.int32))

        # Test numpy int
        self.assertRaises(ValueError, m.update_values_pix, np.arange(100),
                          np.ones(1, dtype=np.int32)[0])

    def test_singlevalue_recarray(self):
        """
        Test assigning single recarray values
        """
        dtype = [('a', 'f4'), ('b', 'i4')]

        m = healsparse.HealSparseMap.make_empty(32, 512, dtype, primary='a')

        # Test single recarray
        val = np.zeros(1, dtype=dtype)
        m.update_values_pix(np.arange(100), val)
        testing.assert_array_equal(m[0: 100]['a'], np.zeros(100, dtype=np.float32))
        testing.assert_array_equal(m[0: 100]['b'], np.zeros(100, dtype=np.int32))

        # Test None
        m.update_values_pix(np.arange(25), None)
        testing.assert_array_equal(m[0: 25][m.primary], m.sentinel)

        m[50] = None
        self.assertEqual(m[50][m.primary], m.sentinel)

        # Test None (no append)
        covmask_orig = m.coverage_mask
        m[1000] = None
        testing.assert_array_equal(m.coverage_mask, covmask_orig)

        # Test wrong dtype recarray
        val = np.ones(1, dtype=[('a', 'f4'), ('b', 'f4')])
        self.assertRaises(ValueError, m.update_values_pix, np.arange(100), val)

        val = np.ones(1, dtype=[('a', 'f4'), ('b', 'i4'), ('c', 'f4')])
        self.assertRaises(ValueError, m.update_values_pix, np.arange(100), val)

        # Test non-recarray setting
        self.assertRaises(ValueError, m.update_values_pix, np.arange(100), 1)

    def test_singlevalue_widemask(self):
        """
        Test assigning single widemask values
        """
        m = healsparse.HealSparseMap.make_empty(32, 512, healsparse.WIDE_MASK, wide_mask_maxbits=16)

        val = np.ones(2, dtype=np.uint8)
        val[1] = 2
        m.update_values_pix(np.arange(100), val)
        testing.assert_array_equal(m[0: 100][:, 0], np.ones(100, dtype=np.int8))
        testing.assert_array_equal(m[0: 100][:, 1], np.full(100, 2, dtype=np.int8))

        # Test None
        m.update_values_pix(np.arange(25), None)
        testing.assert_array_equal(m[0: 25], 0)

        m[50] = None
        self.assertEqual(m[50][0], 0)
        self.assertEqual(m[50][1], 0)

        # Test None (no append)
        covmask_orig = m.coverage_mask
        m[1000] = None
        testing.assert_array_equal(m.coverage_mask, covmask_orig)

        self.assertRaises(ValueError, m.update_values_pix, np.arange(100), 1)
        self.assertRaises(ValueError, m.update_values_pix, np.arange(100),
                          np.zeros(3, dtype=np.int8))
        self.assertRaises(ValueError, m.update_values_pix, np.arange(100),
                          np.zeros(3, dtype=np.int16))

        # Test an edge case where there is ambiguity between the length of the array and
        # the width of the wide mask

        m = healsparse.HealSparseMap.make_empty(32, 512, healsparse.WIDE_MASK, wide_mask_maxbits=32)

        val = np.ones((4, 4), dtype=np.uint8)
        m.update_values_pix(np.arange(4), val)

        testing.assert_array_equal(m[0: 4][:, 0], np.ones(4, dtype=np.uint8))
        testing.assert_array_equal(m[0: 4][:, 1], np.ones(4, dtype=np.uint8))
        testing.assert_array_equal(m[0: 4][:, 2], np.ones(4, dtype=np.uint8))
        testing.assert_array_equal(m[0: 4][:, 3], np.ones(4, dtype=np.uint8))


if __name__ == '__main__':
    unittest.main()
