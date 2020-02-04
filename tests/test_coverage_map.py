from __future__ import division, absolute_import, print_function

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
import healsparse


class CoverageMapTestCase(unittest.TestCase):
    def test_coverage_map(self):
        """
        Test coverageMap functionality
        """

        nside_coverage = 16
        nsideMap = 512
        # Number of non-masked pixels in the coverage map resolution
        non_masked_px = 10
        nfine = (nsideMap//nside_coverage)**2
        full_map = np.zeros(hp.nside2npix(nsideMap)) + hp.UNSEEN
        full_map[0: non_masked_px*nfine] = 1 + np.random.random(size=non_masked_px*nfine)

        # Generate sparse map

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage)

        # Build the "original" coverage map

        cov_map_orig = np.zeros(hp.nside2npix(nside_coverage), dtype=np.double)
        idx_cov = np.right_shift(np.arange(0, non_masked_px*nfine), sparse_map._bit_shift)
        unique_idx_cov = np.unique(idx_cov)
        idx_counts = np.bincount(idx_cov, minlength=hp.nside2npix(nside_coverage)).astype(float)

        cov_map_orig[unique_idx_cov] = idx_counts[unique_idx_cov]/nfine

        # Get the built coverage map

        cov_map = sparse_map.coverage_map

        # Test the coverage map generation and lookup

        testing.assert_equal(cov_map_orig, cov_map)


if __name__ == '__main__':
    unittest.main()
