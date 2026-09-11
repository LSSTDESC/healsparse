import unittest
import numpy as np
import hpgeom as hpg

import healsparse


class HealsparseDefragmentation(unittest.TestCase):
    def test_is_fragmented(self):
        nside_coverage = 32
        nside_map = 128

        for t in ["plain", "wide_mask", "bit_packed", "recarray"]:
            wide_mask_maxbits = None
            bit_packed = False
            primary = None
            empty = None
            if t == "plain":
                dtype = np.int32
                value1 = 100
                value2 = 200
            elif t == "wide_mask":
                dtype = healsparse.WIDE_MASK
                wide_mask_maxbits = 16
                value1 = np.asarray([1, 10])
                value2 = np.asarray([2, 11])
                empty = np.asarray([0, 0])
            elif t == "bit_packed":
                dtype = np.bool_
                bit_packed = True
                value1 = True
                value2 = True
            elif t == "recarray":
                dtype = [("a", "f8"), ("b", "i4")]
                primary = "a"
                value1 = np.zeros(1, dtype=dtype)
                value1["a"] = 100.0
                value2 = np.zeros(1, dtype=dtype)
                value2["a"] = 200.0
                empty = np.zeros(1, dtype=dtype)
                empty["a"] = hpg.UNSEEN

            # Make a map out of order.
            m = healsparse.HealSparseMap.make_empty(
                nside_coverage,
                nside_map,
                dtype,
                wide_mask_maxbits=wide_mask_maxbits,
                bit_packed=bit_packed,
                primary=primary,
            )
            m[100000: 100010] = value1
            m[10000: 10010] = value2

            self.assertTrue(m.is_fragmented, msg=f"Failure with {t}")

            # Make a map and erase a coverage pixel.
            m = healsparse.HealSparseMap.make_empty(
                nside_coverage,
                nside_map,
                dtype,
                wide_mask_maxbits=wide_mask_maxbits,
                bit_packed=bit_packed,
                primary=primary,
            )
            m[10000: 10010] = value2
            m[100000: 100010] = value1

            self.assertFalse(m.is_fragmented, msg=f"Failure with {t}")

            if empty is None:
                m[100000: 100010] = m.sentinel
            else:
                m[100000: 100010] = empty

            self.assertTrue(m.is_fragmented, msg=f"Failure with {t}")

    def test_defragment_outoforder(self):
        nside_coverage = 32
        nside_map = 128

        for t in ["plain", "wide_mask", "bit_packed", "recarray"]:
            wide_mask_maxbits = None
            bit_packed = False
            primary = None
            if t == "plain":
                dtype = np.int32
                value1 = 100
                value2 = 200
            elif t == "wide_mask":
                dtype = healsparse.WIDE_MASK
                wide_mask_maxbits = 16
                value1 = np.asarray([1, 10])
                value2 = np.asarray([2, 11])
            elif t == "bit_packed":
                dtype = np.bool_
                bit_packed = True
                value1 = True
                value2 = True
            elif t == "recarray":
                dtype = [("a", "f8"), ("b", "i4")]
                primary = "a"
                value1 = np.zeros(1, dtype=dtype)
                value1["a"] = 100.0
                value2 = np.zeros(1, dtype=dtype)
                value2["a"] = 200.0

            # Make a map out of order.
            m = healsparse.HealSparseMap.make_empty(
                nside_coverage,
                nside_map,
                dtype,
                wide_mask_maxbits=wide_mask_maxbits,
                bit_packed=bit_packed,
                primary=primary,
            )
            m[100000: 100010] = value1
            m[10000: 10010] = value2

            self.assertTrue(m.is_fragmented, msg=f"Failure with {t}")

            # Defragment into a new map.
            m2 = m.defragment(in_place=False)
            self.assertFalse(m2.is_fragmented, msg=f"Failure with {t}")

            # Assert same valid pixels.
            # These need sorting because original is out-of-order.
            np.testing.assert_array_equal(
                np.sort(m2.valid_pixels),
                np.sort(m.valid_pixels),
                err_msg=f"Failure with {t}",
            )

            # Assert same values.
            np.testing.assert_array_equal(
                m2[m2.valid_pixels],
                m[m2.valid_pixels],
                err_msg=f"Failure with {t}",
            )

            # Defragment in-place.
            m.defragment()
            self.assertFalse(m.is_fragmented, msg=f"Failure with {t}")

            # Assert same valid pixels.
            # These do not need sorting.
            np.testing.assert_array_equal(m.valid_pixels, m2.valid_pixels, err_msg=f"Failure with {t}")

            # Assert same values.
            np.testing.assert_array_equal(m[m.valid_pixels], m2[m2.valid_pixels], err_msg=f"Failure with {t}")

    def test_defragment_blank(self):
        nside_coverage = 32
        nside_map = 128

        for t in ["plain", "wide_mask", "bit_packed", "recarray"]:
            wide_mask_maxbits = None
            bit_packed = False
            primary = None
            empty = None
            if t == "plain":
                dtype = np.int32
                value1 = 100
                value2 = 200
            elif t == "wide_mask":
                dtype = healsparse.WIDE_MASK
                wide_mask_maxbits = 16
                value1 = np.asarray([1, 10])
                value2 = np.asarray([2, 11])
                empty = np.asarray([0, 0])
            elif t == "bit_packed":
                dtype = np.bool_
                bit_packed = True
                value1 = True
                value2 = True
            elif t == "recarray":
                dtype = [("a", "f8"), ("b", "i4")]
                primary = "a"
                value1 = np.zeros(1, dtype=dtype)
                value1["a"] = 100.0
                value2 = np.zeros(1, dtype=dtype)
                value2["a"] = 200.0
                empty = np.zeros(1, dtype=dtype)
                empty["a"] = hpg.UNSEEN

            # Make a map out of order.
            m = healsparse.HealSparseMap.make_empty(
                nside_coverage,
                nside_map,
                dtype,
                wide_mask_maxbits=wide_mask_maxbits,
                bit_packed=bit_packed,
                primary=primary,
            )
            m[10000: 10010] = value2
            m[100000: 100010] = value1

            if empty is None:
                m[100000: 100010] = m.sentinel
            else:
                m[100000: 100010] = empty

            self.assertTrue(m.is_fragmented, msg=f"Failure with {t}")

            # Defragment into a new map.
            m2 = m.defragment(in_place=False)
            self.assertFalse(m2.is_fragmented, msg=f"Failure with {t}")

            # Assert same valid pixels.
            # These need sorting because original is out-of-order.
            np.testing.assert_array_equal(
                np.sort(m2.valid_pixels),
                np.sort(m.valid_pixels),
                err_msg=f"Failure with {t}",
            )

            # Assert same values.
            np.testing.assert_array_equal(
                m2[m2.valid_pixels],
                m[m2.valid_pixels],
                err_msg=f"Failure with {t}",
            )

            # Defragment in-place.
            m.defragment()
            self.assertFalse(m.is_fragmented, msg=f"Failure with {t}")

            # Assert same valid pixels.
            # These do not need sorting.
            np.testing.assert_array_equal(m.valid_pixels, m2.valid_pixels, err_msg=f"Failure with {t}")

            # Assert same values.
            np.testing.assert_array_equal(m[m.valid_pixels], m2[m2.valid_pixels], err_msg=f"Failure with {t}")


if __name__ == '__main__':
    unittest.main()
