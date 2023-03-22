import unittest
import numpy as np
import hpgeom as hpg

import healsparse


class SingleDatatypesTestCase(unittest.TestCase):
    def test_datatypes_ints(self):
        """
        Test making maps with integer datatypes
        """
        nside_coverage = 32
        nside_map = 64

        # int16
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.int16)

        self.assertEqual(sparse_map.dtype, np.int16)
        self.assertEqual(sparse_map.sentinel, -32768)

        # int32
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.int32)

        self.assertEqual(sparse_map.dtype, np.int32)
        self.assertEqual(sparse_map.sentinel, -2147483648)

        # int64
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.int64)

        self.assertEqual(sparse_map.dtype, np.int64)
        self.assertEqual(sparse_map.sentinel, -9223372036854775808)

    def test_datatypes_floats(self):
        """
        Test making maps with float datatypes
        """
        nside_coverage = 32
        nside_map = 64

        # float32
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.float32)

        self.assertEqual(sparse_map.dtype, np.float32)
        self.assertEqual(sparse_map.sentinel, np.float32(hpg.UNSEEN))

        # int64
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.float64)

        self.assertEqual(sparse_map.dtype, np.float64)
        self.assertEqual(sparse_map.sentinel, np.float64(hpg.UNSEEN))

    def test_datatypes_uints(self):
        """
        Test making maps with unsigned integer datatypes
        """
        nside_coverage = 32
        nside_map = 64

        # uint16
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.uint16)

        self.assertEqual(sparse_map.dtype, np.uint16)
        self.assertEqual(sparse_map.sentinel, 0)

        # uint32
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.uint32)

        self.assertEqual(sparse_map.dtype, np.uint32)
        self.assertEqual(sparse_map.sentinel, 0)

        # uint64
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.uint64)

        self.assertEqual(sparse_map.dtype, np.uint64)
        self.assertEqual(sparse_map.sentinel, 0)

    def test_datatypes_bools(self):
        """
        Test making maps with boolean datatypes
        """
        nside_coverage = 32
        nside_map = 64

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, bool)

        self.assertEqual(sparse_map.dtype, np.bool_)
        self.assertEqual(sparse_map.sentinel, False)

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, bool, sentinel=True)

        self.assertEqual(sparse_map.dtype, np.bool_)
        self.assertEqual(sparse_map.sentinel, True)

        with self.assertRaises(ValueError):
            sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, bool, sentinel=0)


if __name__ == '__main__':
    unittest.main()
