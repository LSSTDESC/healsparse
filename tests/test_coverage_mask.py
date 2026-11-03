import unittest
import numpy.testing as testing
import numpy as np
import hpgeom as hpg
import healsparse


class CoverageMaskTestCase(unittest.TestCase):
    def test_coverage_mask(self):
        """
        Test coverageMask functionality
        """
        nside_coverage = 8
        nside_map = 64
        # Number of non-masked pixels in the coverage map resolution
        non_masked_px = 10
        nfine = (nside_map//nside_coverage)**2
        full_map = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        full_map[0: non_masked_px*nfine] = 1 + np.random.random(size=non_masked_px*nfine)

        # Generate sparse map

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage)

        # Build the "original" coverage mask

        cov_mask_orig = np.zeros(hpg.nside_to_npixel(nside_coverage), dtype=np.bool_)
        idx_cov = np.unique(np.right_shift(np.arange(0, non_masked_px*nfine), sparse_map._cov_map.bit_shift))
        cov_mask_orig[idx_cov] = 1

        # Get the built coverage mask

        cov_mask = sparse_map.coverage_mask

        # Test the mask generation and lookup

        testing.assert_equal(cov_mask_orig, cov_mask)

    def test_valid_coverage_mask(self):
        bit_shift = healsparse.utils._compute_bitshift(32, 512)

        for mtype in ["wide", "bit_packed", "rec_array", "float"]:
            if mtype == "wide":
                m = healsparse.HealSparseMap.make_empty(32, 512, healsparse.WIDE_MASK, wide_mask_maxbits=7)
                m.set_bits_pix(np.arange(1000), [4])
                m.set_bits_pix(np.arange(10000, 11000), [3])
                value = None
            elif mtype == "bit_packed":
                m = healsparse.HealSparseMap.make_empty(32, 512, bool, bit_packed=True)
                value = True
            elif mtype == "rec_array":
                dtype = [("a", "f8"), ("b", "i4")]
                m = healsparse.HealSparseMap.make_empty(32, 512, dtype, primary="a")
                value = np.ones(1, dtype=dtype)
            else:
                m = healsparse.HealSparseMap.make_empty(32, 512, np.float32)
                value = 5.0

            if value is not None:
                m[0: 1000] = value
                m[10000: 11000] = value

            valid_coverage_mask = m.valid_coverage_mask
            valid_pixels = m.valid_pixels

            testing.assert_array_equal(
                np.nonzero(valid_coverage_mask)[0],
                np.unique(np.right_shift(valid_pixels, bit_shift)),
            )

    def test_valid_coverage_mask_outoforder(self):
        bit_shift = healsparse.utils._compute_bitshift(32, 512)

        for mtype in ["wide", "bit_packed", "rec_array", "float"]:
            if mtype == "wide":
                m = healsparse.HealSparseMap.make_empty(32, 512, healsparse.WIDE_MASK, wide_mask_maxbits=7)
                m.set_bits_pix(np.arange(10000, 11000), [3])
                m.set_bits_pix(np.arange(1000), [4])
                value = None
            elif mtype == "bit_packed":
                m = healsparse.HealSparseMap.make_empty(32, 512, bool, bit_packed=True)
                value = True
            elif mtype == "rec_array":
                dtype = [("a", "f8"), ("b", "i4")]
                m = healsparse.HealSparseMap.make_empty(32, 512, dtype, primary="a")
                value = np.ones(1, dtype=dtype)
            else:
                m = healsparse.HealSparseMap.make_empty(32, 512, np.float32)
                value = 5.0

            if value is not None:
                m[10000: 11000] = value
                m[0: 1000] = value

            valid_coverage_mask = m.valid_coverage_mask
            valid_pixels = m.valid_pixels

            testing.assert_array_equal(
                np.nonzero(valid_coverage_mask)[0],
                np.unique(np.right_shift(valid_pixels, bit_shift)),
            )

    def test_valid_coverage_mask_blanked(self):
        bit_shift = healsparse.utils._compute_bitshift(32, 512)

        for mtype in ["wide", "bit_packed", "rec_array", "float"]:
            if mtype == "wide":
                m = healsparse.HealSparseMap.make_empty(32, 512, healsparse.WIDE_MASK, wide_mask_maxbits=7)
                m.set_bits_pix(np.arange(10000, 11000), [3])
                m.set_bits_pix(np.arange(1000), [4])
                m.set_bits_pix(np.arange(1000), [4])
                value = None
            elif mtype == "bit_packed":
                m = healsparse.HealSparseMap.make_empty(32, 512, bool, bit_packed=True)
                value = True
            elif mtype == "rec_array":
                dtype = [("a", "f8"), ("b", "i4")]
                m = healsparse.HealSparseMap.make_empty(32, 512, dtype, primary="a")
                value = np.ones(1, dtype=dtype)
            else:
                m = healsparse.HealSparseMap.make_empty(32, 512, np.float32)
                value = 5.0

            if value is not None:
                m[10000: 11000] = value
                m[0: 1000] = value
                m[0: 1000] = None

            valid_coverage_mask = m.valid_coverage_mask
            valid_pixels = m.valid_pixels

            testing.assert_array_equal(
                np.nonzero(valid_coverage_mask)[0],
                np.unique(np.right_shift(valid_pixels, bit_shift)),
            )


if __name__ == '__main__':
    unittest.main()
