from __future__ import division, absolute_import, print_function

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
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
        full_map = np.zeros(hp.nside2npix(nside_map)) + hp.UNSEEN
        full_map[0: non_masked_px*nfine] = 1 + np.random.random(size=non_masked_px*nfine)

        # Generate sparse map

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage)

        # Build the "original" coverage mask

        cov_mask_orig = np.zeros(hp.nside2npix(nside_coverage), dtype=np.bool_)
        idx_cov = np.unique(np.right_shift(np.arange(0, non_masked_px*nfine), sparse_map._cov_map.bit_shift))
        cov_mask_orig[idx_cov] = 1

        # Get the built coverage mask

        cov_mask = sparse_map.coverage_mask

        # Test the mask generation and lookup

        testing.assert_equal(cov_mask_orig, cov_mask)


if __name__ == '__main__':
    unittest.main()
