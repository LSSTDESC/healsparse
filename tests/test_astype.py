import unittest
import numpy.testing as testing
import numpy as np
import hpgeom as hpg

import healsparse


class AstypeCase(unittest.TestCase):
    def test_float_to_int(self):
        """
        Test float to int type conversion.
        """
        np.random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.float64)
        sparse_map[0: 5000] = 1.0
        nfine_per_cov = sparse_map._cov_map.nfine_per_cov

        # Convert to an int map with default sentinel
        sparse_map_int = sparse_map.astype(np.int32)
        self.assertEqual(sparse_map_int.dtype.type, np.dtype(np.int32).type)

        testing.assert_array_equal(sparse_map.valid_pixels, sparse_map_int.valid_pixels)
        testing.assert_array_almost_equal(sparse_map[sparse_map.valid_pixels],
                                          sparse_map_int[sparse_map.valid_pixels])
        testing.assert_array_equal(sparse_map_int._sparse_map[0: nfine_per_cov],
                                   np.iinfo(np.dtype(np.int32).type).min)

        # Convert to an int map with 0 sentinel
        sparse_map_int2 = sparse_map.astype(np.int32, sentinel=0)

        self.assertEqual(sparse_map_int2.dtype.type, np.dtype(np.int32).type)

        testing.assert_array_equal(sparse_map.valid_pixels, sparse_map_int2.valid_pixels)
        testing.assert_array_almost_equal(sparse_map[sparse_map.valid_pixels],
                                          sparse_map_int2[sparse_map.valid_pixels])
        testing.assert_array_equal(sparse_map_int2._sparse_map[0: nfine_per_cov], 0)

        self.assertRaises(ValueError, sparse_map.astype, np.int32, sentinel=hpg.UNSEEN)

    def test_int_to_float(self):
        """
        Test int to float type conversion.
        """
        np.random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.int64)
        sparse_map[0: 5000] = 1
        nfine_per_cov = sparse_map._cov_map.nfine_per_cov

        # Convert to a float map with default sentinel
        sparse_map_float = sparse_map.astype(np.float32)
        self.assertEqual(sparse_map_float.dtype.type, np.dtype(np.float32).type)

        testing.assert_array_equal(sparse_map.valid_pixels, sparse_map_float.valid_pixels)
        testing.assert_array_almost_equal(sparse_map[sparse_map.valid_pixels],
                                          sparse_map_float[sparse_map.valid_pixels])
        testing.assert_array_equal(sparse_map_float._sparse_map[0: nfine_per_cov],
                                   hpg.UNSEEN)

        # Convert to a float map with 0 sentinel
        sparse_map_float2 = sparse_map.astype(np.float32, sentinel=0.0)

        self.assertEqual(sparse_map_float2.dtype.type, np.dtype(np.float32).type)

        testing.assert_array_equal(sparse_map.valid_pixels, sparse_map_float2.valid_pixels)
        testing.assert_array_almost_equal(sparse_map[sparse_map.valid_pixels],
                                          sparse_map_float2[sparse_map.valid_pixels])
        testing.assert_array_equal(sparse_map_float2._sparse_map[0: nfine_per_cov], 0)

        self.assertRaises(ValueError, sparse_map.astype, np.float32, sentinel=0)

    def test_int_to_int(self):
        """
        Test int to int type conversion.
        """
        np.random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.int64)
        sparse_map[0: 5000] = 1
        nfine_per_cov = sparse_map._cov_map.nfine_per_cov

        # Convert to a different int map with default sentinel
        sparse_map_int = sparse_map.astype(np.int32)
        self.assertEqual(sparse_map_int.dtype.type, np.dtype(np.int32).type)

        testing.assert_array_equal(sparse_map.valid_pixels, sparse_map_int.valid_pixels)
        testing.assert_array_almost_equal(sparse_map[sparse_map.valid_pixels],
                                          sparse_map_int[sparse_map.valid_pixels])
        testing.assert_array_equal(sparse_map_int._sparse_map[0: nfine_per_cov],
                                   np.iinfo(np.dtype(np.int32).type).min)

        # Convert to a different int map with 0 sentinel
        sparse_map_int2 = sparse_map.astype(np.int32, sentinel=0)

        self.assertEqual(sparse_map_int2.dtype.type, np.dtype(np.int32).type)

        testing.assert_array_equal(sparse_map.valid_pixels, sparse_map_int2.valid_pixels)
        testing.assert_array_almost_equal(sparse_map[sparse_map.valid_pixels],
                                          sparse_map_int2[sparse_map.valid_pixels])
        testing.assert_array_equal(sparse_map_int2._sparse_map[0: nfine_per_cov], 0)

        self.assertRaises(ValueError, sparse_map.astype, np.int32, sentinel=hpg.UNSEEN)

    def test_float_to_float(self):
        """
        Test float to float type conversion.
        """
        np.random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.float64)
        sparse_map[0: 5000] = 1.0
        nfine_per_cov = sparse_map._cov_map.nfine_per_cov

        # Convert to a different float map with default sentinel
        sparse_map_float = sparse_map.astype(np.float32)
        self.assertEqual(sparse_map_float.dtype.type, np.dtype(np.float32).type)

        testing.assert_array_equal(sparse_map.valid_pixels, sparse_map_float.valid_pixels)
        testing.assert_array_almost_equal(sparse_map[sparse_map.valid_pixels],
                                          sparse_map_float[sparse_map.valid_pixels])
        testing.assert_array_equal(sparse_map_float._sparse_map[0: nfine_per_cov],
                                   hpg.UNSEEN)

        # Convert to a different float map with 0 sentinel
        sparse_map_float2 = sparse_map.astype(np.float32, sentinel=0.0)

        self.assertEqual(sparse_map_float2.dtype.type, np.dtype(np.float32).type)

        testing.assert_array_equal(sparse_map.valid_pixels, sparse_map_float2.valid_pixels)
        testing.assert_array_almost_equal(sparse_map[sparse_map.valid_pixels],
                                          sparse_map_float2[sparse_map.valid_pixels])
        testing.assert_array_equal(sparse_map_float2._sparse_map[0: nfine_per_cov], 0)

        self.assertRaises(ValueError, sparse_map.astype, np.float32, sentinel=0)


if __name__ == '__main__':
    unittest.main()
