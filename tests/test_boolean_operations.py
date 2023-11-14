import unittest
import numpy.testing as testing
import numpy as np

from healsparse import HealSparseMap


class BooleanOperationsTestCase(unittest.TestCase):
    def _check_operated_maps_value(self, map_new, map_old, operation, value):
        nfine_per_cov = map_old._cov_map.nfine_per_cov

        testing.assert_array_equal(
            map_new._sparse_map[: nfine_per_cov],
            map_old._sparse_map[: nfine_per_cov],
        )
        # And the other region is as expected.
        if operation == "and":
            compare = map_old._sparse_map[nfine_per_cov:] & value
        elif operation == "or":
            compare = map_old._sparse_map[nfine_per_cov:] | value
        elif operation == "xor":
            compare = map_old._sparse_map[nfine_per_cov:] ^ value

        testing.assert_array_equal(
            map_new._sparse_map[nfine_per_cov:],
            compare
        )

    def _check_operated_maps_map(self, map_new, map1, operation, map2):
        # Check combined coverage mask.
        coverage_mask = map1.coverage_mask | map2.coverage_mask
        testing.assert_array_equal(map_new.coverage_mask, coverage_mask)

        cov_pixels2, = map2.coverage_mask.nonzero()

        # Over these coverage pixels, we should match.
        for cov_pixel in coverage_mask.nonzero()[0]:
            pixels = np.arange(map_new._cov_map.nfine_per_cov) + cov_pixel*map_new._cov_map.nfine_per_cov
            if cov_pixel in cov_pixels2:
                # These should be altered.
                if operation == "and":
                    compare = map1[pixels] & map2[pixels]
                elif operation == "or":
                    compare = map1[pixels] | map2[pixels]
                elif operation == "xor":
                    compare = map1[pixels] ^ map2[pixels]
            else:
                # These should be untouched.
                compare = map1[pixels]

            testing.assert_array_equal(
                map_new[pixels],
                compare,
            )

    def test_and_const(self):
        m_bool = HealSparseMap.make_empty(32, 256, np.bool_)
        m_bool[0: 1000] = True
        m_bool[500_000: 600_000] = True

        m_bool2 = m_bool & True
        self._check_operated_maps_value(m_bool2, m_bool, "and", True)
        m_bool2 = m_bool & False
        self._check_operated_maps_value(m_bool2, m_bool, "and", False)

        m_bool2 = m_bool.copy()
        m_bool2 &= True
        self._check_operated_maps_value(m_bool2, m_bool, "and", True)
        m_bool2 = m_bool.copy()
        m_bool2 &= False
        self._check_operated_maps_value(m_bool2, m_bool, "and", False)

        m_packed = m_bool.as_bit_packed_map()
        m_packed2 = m_packed & True
        self._check_operated_maps_value(m_packed2, m_packed, "and", True)
        m_packed2 = m_packed & False
        self._check_operated_maps_value(m_packed2, m_packed, "and", False)

        m_packed2 = m_packed.copy()
        m_packed2 &= True
        self._check_operated_maps_value(m_packed2, m_packed, "and", True)
        m_packed2 = m_packed.copy()
        m_packed2 &= False
        self._check_operated_maps_value(m_packed2, m_packed, "and", False)

    def test_or_const(self):
        m_bool = HealSparseMap.make_empty(32, 256, np.bool_)
        m_bool[0: 1000] = True
        m_bool[500_000: 600_000] = True

        m_bool2 = m_bool | True
        self._check_operated_maps_value(m_bool2, m_bool, "or", True)
        m_bool2 = m_bool | False
        self._check_operated_maps_value(m_bool2, m_bool, "or", False)

        m_bool2 = m_bool.copy()
        m_bool2 |= True
        self._check_operated_maps_value(m_bool2, m_bool, "or", True)
        m_bool2 = m_bool.copy()
        m_bool2 |= False
        self._check_operated_maps_value(m_bool2, m_bool, "or", False)

        m_packed = m_bool.as_bit_packed_map()
        m_packed2 = m_packed | True
        self._check_operated_maps_value(m_packed2, m_packed, "or", True)
        m_packed2 = m_packed | False
        self._check_operated_maps_value(m_packed2, m_packed, "or", False)

        m_packed2 = m_packed.copy()
        m_packed2 |= True
        self._check_operated_maps_value(m_packed2, m_packed, "or", True)
        m_packed2 = m_packed.copy()
        m_packed2 |= False
        self._check_operated_maps_value(m_packed2, m_packed, "or", False)

    def test_xor_const(self):
        m_bool = HealSparseMap.make_empty(32, 256, np.bool_)
        m_bool[0: 1000] = True
        m_bool[500_000: 600_000] = True

        m_bool2 = m_bool ^ True
        self._check_operated_maps_value(m_bool2, m_bool, "xor", True)
        m_bool2 = m_bool ^ False
        self._check_operated_maps_value(m_bool2, m_bool, "xor", False)

        m_bool2 = m_bool.copy()
        m_bool2 ^= True
        self._check_operated_maps_value(m_bool2, m_bool, "xor", True)
        m_bool2 = m_bool.copy()
        m_bool2 ^= False
        self._check_operated_maps_value(m_bool2, m_bool, "xor", False)

        m_packed = m_bool.as_bit_packed_map()
        m_packed2 = m_packed ^ True
        self._check_operated_maps_value(m_packed2, m_packed, "xor", True)
        m_packed2 = m_packed ^ False
        self._check_operated_maps_value(m_packed2, m_packed, "xor", False)

        m_packed2 = m_packed.copy()
        m_packed2 ^= True
        self._check_operated_maps_value(m_packed2, m_packed, "xor", True)
        m_packed2 = m_packed.copy()
        m_packed2 ^= False
        self._check_operated_maps_value(m_packed2, m_packed, "xor", False)

    def test_and_map(self):
        # We need to test several combinations.
        #  bool/bool, bool/packed, packed/bool, packed/packed
        #  Same coverage, first with more, second with more

        for cov_type in ["same", "first", "second"]:
            m_bool = HealSparseMap.make_empty(32, 256, np.bool_)
            m2_bool = HealSparseMap.make_empty(32, 256, np.bool_)

            if cov_type == "same":
                m_bool[10_000: 50_000] = True
                m2_bool[10_000: 50_000] = True
                m2_bool[15_000: 16_000] = False
            elif cov_type == "first":
                m_bool[10_000: 50_000] = True
                m2_bool[10_000: 20_000] = True
                m2_bool[15_000: 16_000] = False
            elif cov_type == "second":
                m_bool[10_000: 20_000] = True
                m2_bool[10_000: 50_000] = True
                m2_bool[15_000: 16_000] = False

            for combo in range(4):
                if combo == 0:
                    # Bool/Bool
                    map1 = m_bool
                    map2 = m2_bool
                elif combo == 1:
                    # Bool/Packed
                    map1 = m_bool
                    map2 = m2_bool.as_bit_packed_map()
                elif combo == 2:
                    # Packed/Bool
                    map1 = m_bool.as_bit_packed_map()
                    map2 = m2_bool
                else:
                    # Packed/Packed
                    map1 = m_bool.as_bit_packed_map()
                    map2 = m2_bool.as_bit_packed_map()

                map3 = map1 & map2
                self._check_operated_maps_map(map3, map1, "and", map2)

                map3 = map1.copy()
                map3 &= map2
                self._check_operated_maps_map(map3, map1, "and", map2)

    def test_or_map(self):
        # We need to test several combinations.
        #  bool/bool, bool/packed, packed/bool, packed/packed
        #  Same coverage, first with more, second with more

        for cov_type in ["same", "first", "second"]:
            m_bool = HealSparseMap.make_empty(32, 256, np.bool_)
            m2_bool = HealSparseMap.make_empty(32, 256, np.bool_)

            if cov_type == "same":
                m_bool[10_000: 50_000] = True
                m2_bool[10_000: 50_000] = True
                m2_bool[15_000: 16_000] = False
            elif cov_type == "first":
                m_bool[10_000: 50_000] = True
                m2_bool[10_000: 20_000] = True
                m2_bool[15_000: 16_000] = False
            elif cov_type == "second":
                m_bool[10_000: 20_000] = True
                m2_bool[10_000: 50_000] = True
                m2_bool[15_000: 16_000] = False

            for combo in range(4):
                if combo == 0:
                    # Bool/Bool
                    map1 = m_bool
                    map2 = m2_bool
                elif combo == 1:
                    # Bool/Packed
                    map1 = m_bool
                    map2 = m2_bool.as_bit_packed_map()
                elif combo == 2:
                    # Packed/Bool
                    map1 = m_bool.as_bit_packed_map()
                    map2 = m2_bool
                else:
                    # Packed/Packed
                    map1 = m_bool.as_bit_packed_map()
                    map2 = m2_bool.as_bit_packed_map()

                map3 = map1 | map2
                self._check_operated_maps_map(map3, map1, "or", map2)

                map3 = map1.copy()
                map3 |= map2
                self._check_operated_maps_map(map3, map1, "or", map2)

    def test_xor_map(self):
        # We need to test several combinations.
        #  bool/bool, bool/packed, packed/bool, packed/packed
        #  Same coverage, first with more, second with more

        for cov_type in ["same", "first", "second"]:
            m_bool = HealSparseMap.make_empty(32, 256, np.bool_)
            m2_bool = HealSparseMap.make_empty(32, 256, np.bool_)

            if cov_type == "same":
                m_bool[10_000: 50_000] = True
                m2_bool[10_000: 50_000] = True
                m2_bool[15_000: 16_000] = False
            elif cov_type == "first":
                m_bool[10_000: 50_000] = True
                m2_bool[10_000: 20_000] = True
                m2_bool[15_000: 16_000] = False
            elif cov_type == "second":
                m_bool[10_000: 20_000] = True
                m2_bool[10_000: 50_000] = True
                m2_bool[15_000: 16_000] = False

            for combo in range(4):
                if combo == 0:
                    # Bool/Bool
                    map1 = m_bool
                    map2 = m2_bool
                elif combo == 1:
                    # Bool/Packed
                    map1 = m_bool
                    map2 = m2_bool.as_bit_packed_map()
                elif combo == 2:
                    # Packed/Bool
                    map1 = m_bool.as_bit_packed_map()
                    map2 = m2_bool
                else:
                    # Packed/Packed
                    map1 = m_bool.as_bit_packed_map()
                    map2 = m2_bool.as_bit_packed_map()

                map3 = map1 ^ map2
                self._check_operated_maps_map(map3, map1, "xor", map2)

                map3 = map1.copy()
                map3 ^= map2
                self._check_operated_maps_map(map3, map1, "xor", map2)

    def test_invert(self):
        m_bool = HealSparseMap.make_empty(32, 256, np.bool_)
        m_bool[0: 1000] = True
        m_bool[500_000: 600_000] = True

        m_bool2 = ~m_bool

        testing.assert_array_equal(
            m_bool2._sparse_map[m_bool2._cov_map.nfine_per_cov:],
            ~m_bool._sparse_map[m_bool2._cov_map.nfine_per_cov:],
        )

        m_bool2 = m_bool.copy()
        m_bool2.invert()

        testing.assert_array_equal(
            m_bool2._sparse_map[m_bool2._cov_map.nfine_per_cov:],
            ~m_bool._sparse_map[m_bool2._cov_map.nfine_per_cov:],
        )

        m_packed = m_bool.as_bit_packed_map()
        m_packed2 = ~m_packed

        testing.assert_array_equal(
            m_packed2._sparse_map[m_packed2._cov_map.nfine_per_cov:],
            ~m_packed._sparse_map[m_packed2._cov_map.nfine_per_cov:],
        )

        m_packed = m_bool.as_bit_packed_map()
        m_packed2 = m_packed.copy()
        m_packed2.invert()

        testing.assert_array_equal(
            m_packed2._sparse_map[m_packed2._cov_map.nfine_per_cov:],
            ~m_packed._sparse_map[m_packed2._cov_map.nfine_per_cov:],
        )

        m_temp = HealSparseMap.make_empty(32, 256, np.int32)
        with self.assertRaises(NotImplementedError):
            _ = ~m_temp

        m_temp = HealSparseMap.make_empty(32, 256, np.int32)
        with self.assertRaises(NotImplementedError):
            m_temp.invert()


if __name__ == '__main__':
    unittest.main()
