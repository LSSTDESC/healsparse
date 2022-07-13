import unittest
import numpy.testing as testing
import numpy as np
import hpgeom as hpg
import healsparse


class CoverageMapTestCase(unittest.TestCase):
    def test_coverage_map_float(self):
        """
        Test coverage_map functionality for floats
        """

        nside_coverage = 16
        nside_map = 512
        # Number of non-masked pixels in the coverage map resolution
        non_masked_px = 10.5
        nfine = (nside_map//nside_coverage)**2
        full_map = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        full_map[0: int(non_masked_px*nfine)] = 1 + np.random.random(size=int(non_masked_px*nfine))

        # Generate sparse map

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage)

        # Build the "original" coverage map
        cov_map_orig = self.compute_cov_map(nside_coverage, non_masked_px, nfine,
                                            sparse_map._cov_map.bit_shift)

        # Get the built coverage map

        cov_map = sparse_map.coverage_map

        # Test the coverage map generation and lookup

        testing.assert_array_almost_equal(cov_map_orig, cov_map)

    def test_coverage_map_int(self):
        """
        Test coverage_map functionality for ints
        """
        nside_coverage = 16
        nside_map = 512
        # Number of non-masked pixels in the coverage map resolution
        non_masked_px = 10.5
        nfine = (nside_map//nside_coverage)**2
        sentinel = healsparse.utils.check_sentinel(np.int32, None)
        full_map = np.zeros(hpg.nside_to_npixel(nside_map), dtype=np.int32) + sentinel
        full_map[0: int(non_masked_px*nfine)] = 1

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map,
                                              nside_coverage=nside_coverage,
                                              sentinel=sentinel)

        cov_map_orig = self.compute_cov_map(nside_coverage, non_masked_px, nfine,
                                            sparse_map._cov_map.bit_shift)

        cov_map = sparse_map.coverage_map

        testing.assert_array_almost_equal(cov_map_orig, cov_map)

    def test_coverage_map_recarray(self):
        """
        Test coverage_map functionality for a recarray
        """
        nside_coverage = 16
        nside_map = 512
        # Number of non-masked pixels in the coverage map resolution
        non_masked_px = 10.5
        nfine = (nside_map//nside_coverage)**2

        dtype = [('a', np.float64),
                 ('b', np.int32)]
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         dtype, primary='a')
        sparse_map.update_values_pix(np.arange(int(non_masked_px*nfine)),
                                     np.ones(1, dtype=dtype))

        cov_map_orig = self.compute_cov_map(nside_coverage, non_masked_px, nfine,
                                            sparse_map._cov_map.bit_shift)

        cov_map = sparse_map.coverage_map

        testing.assert_array_almost_equal(cov_map_orig, cov_map)

    def test_coverage_map_widemask(self):
        """
        Test coverage_map functionality for wide masks
        """
        nside_coverage = 16
        nside_map = 512
        # Number of non-masked pixels in the coverage map resolution
        non_masked_px = 10.5
        nfine = (nside_map//nside_coverage)**2

        # Do a 1-byte wide
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         healsparse.WIDE_MASK,
                                                         wide_mask_maxbits=2)
        # Set bits in different columns
        sparse_map.set_bits_pix(np.arange(int(non_masked_px*nfine)), [1])

        cov_map_orig = self.compute_cov_map(nside_coverage, non_masked_px, nfine,
                                            sparse_map._cov_map.bit_shift)

        cov_map = sparse_map.coverage_map

        testing.assert_array_almost_equal(cov_map_orig, cov_map)

        # Do a 3-byte wide
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         healsparse.WIDE_MASK,
                                                         wide_mask_maxbits=24)
        # Set bits in different columns
        sparse_map.set_bits_pix(np.arange(int(2*nfine)), [2])
        sparse_map.set_bits_pix(np.arange(int(non_masked_px*nfine)), [20])

        cov_map_orig = self.compute_cov_map(nside_coverage, non_masked_px, nfine,
                                            sparse_map._cov_map.bit_shift)

        cov_map = sparse_map.coverage_map

        testing.assert_array_almost_equal(cov_map_orig, cov_map)

    def compute_cov_map(self, nside_coverage, non_masked_px, nfine, bit_shift):
        cov_map_orig = np.zeros(hpg.nside_to_npixel(nside_coverage), dtype=np.float64)
        idx_cov = np.right_shift(np.arange(int(non_masked_px*nfine)), bit_shift)
        unique_idx_cov = np.unique(idx_cov)
        idx_counts = np.bincount(idx_cov, minlength=hpg.nside_to_npixel(nside_coverage)).astype(np.float64)

        cov_map_orig[unique_idx_cov] = idx_counts[unique_idx_cov]/nfine

        return cov_map_orig

    def test_large_coverage_map_warning(self):
        """
        Test coverage_map raises warning for large
        values of nside_coverage
        """

        nside_coverage = 256
        nside_map = 512

        # Generate sparse map and check that it rasises a warning
        testing.assert_warns(ResourceWarning, healsparse.HealSparseMap.make_empty, nside_sparse=nside_map,
                             nside_coverage=nside_coverage, dtype=np.float32)


if __name__ == '__main__':
    unittest.main()
