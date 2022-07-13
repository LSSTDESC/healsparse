import unittest
import numpy as np
import hpgeom as hpg
import healsparse


class FracdetTestCase(unittest.TestCase):
    def test_fracdet_map_float(self):
        """
        Test fracdet_map functionality for floats
        """
        nside_coverage = 16
        nside_fracdet = 32
        nside_map = 512
        non_masked_px = 10.5
        nfine = (nside_map//nside_coverage)**2
        full_map = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        full_map[0: int(non_masked_px*nfine)] = 1 + np.random.random(size=int(non_masked_px*nfine))

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage)

        # Test that the fracdet map is equal to the coverage map with same nside_coverage
        fracdet_map1 = sparse_map.fracdet_map(nside_coverage)

        np.testing.assert_array_almost_equal(fracdet_map1[:], sparse_map.coverage_map)

        # Test that the fracdet map is good for target nside
        fracdet_map2 = sparse_map.fracdet_map(nside_fracdet)

        fracdet_map_orig = self.compute_fracdet_map(nside_map, nside_fracdet,
                                                    non_masked_px, nfine)

        np.testing.assert_array_almost_equal(fracdet_map2[:], fracdet_map_orig)

    def test_fracdet_map_int(self):
        """
        Test fracdet_map functionality for ints
        """
        nside_coverage = 16
        nside_fracdet = 32
        nside_map = 512
        non_masked_px = 10.5
        nfine = (nside_map//nside_coverage)**2
        sentinel = healsparse.utils.check_sentinel(np.int32, None)
        full_map = np.zeros(hpg.nside_to_npixel(nside_map), dtype=np.int32) + sentinel
        full_map[0: int(non_masked_px*nfine)] = 1

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map,
                                              nside_coverage=nside_coverage,
                                              sentinel=sentinel)

        # Test that the fracdet map is equal to the coverage map with same nside_coverage
        fracdet_map1 = sparse_map.fracdet_map(nside_coverage)

        np.testing.assert_array_almost_equal(fracdet_map1[:], sparse_map.coverage_map)

        # Test that the fracdet map is good for target nside
        fracdet_map2 = sparse_map.fracdet_map(nside_fracdet)

        fracdet_map_orig = self.compute_fracdet_map(nside_map, nside_fracdet,
                                                    non_masked_px, nfine)

        np.testing.assert_array_almost_equal(fracdet_map2[:], fracdet_map_orig)

    def test_fracdet_map_recarray(self):
        """
        Test fracdet_map functionality for recarrays
        """
        nside_coverage = 16
        nside_fracdet = 32
        nside_map = 512
        non_masked_px = 10.5
        nfine = (nside_map//nside_coverage)**2

        dtype = [('a', np.float64),
                 ('b', np.int32)]
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         dtype, primary='a')
        sparse_map.update_values_pix(np.arange(int(non_masked_px*nfine)),
                                     np.ones(1, dtype=dtype))

        # Test that the fracdet map is equal to the coverage map with same nside_coverage
        fracdet_map1 = sparse_map.fracdet_map(nside_coverage)

        np.testing.assert_array_almost_equal(fracdet_map1[:], sparse_map.coverage_map)

        # Test that the fracdet map is good for target nside
        fracdet_map2 = sparse_map.fracdet_map(nside_fracdet)

        fracdet_map_orig = self.compute_fracdet_map(nside_map, nside_fracdet,
                                                    non_masked_px, nfine)

        np.testing.assert_array_almost_equal(fracdet_map2[:], fracdet_map_orig)

    def test_fracdet_map_widemask(self):
        """
        Test fracdet_map functionality for wide masks
        """
        nside_coverage = 16
        nside_fracdet = 32
        nside_map = 512
        non_masked_px = 10.5
        nfine = (nside_map//nside_coverage)**2

        # Do a 1-byte wide
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         healsparse.WIDE_MASK,
                                                         wide_mask_maxbits=2)
        # Set bits in different columns
        sparse_map.set_bits_pix(np.arange(int(non_masked_px*nfine)), [1])

        # Test that the fracdet map is equal to the coverage map with same nside_coverage
        fracdet_map1 = sparse_map.fracdet_map(nside_coverage)

        np.testing.assert_array_almost_equal(fracdet_map1[:], sparse_map.coverage_map)

        # Test that the fracdet map is good for target nside
        fracdet_map2 = sparse_map.fracdet_map(nside_fracdet)

        fracdet_map_orig = self.compute_fracdet_map(nside_map, nside_fracdet,
                                                    non_masked_px, nfine)

        np.testing.assert_array_almost_equal(fracdet_map2[:], fracdet_map_orig)

        # Do a 3-byte wide
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         healsparse.WIDE_MASK,
                                                         wide_mask_maxbits=24)
        # Set bits in different columns
        sparse_map.set_bits_pix(np.arange(int(2*nfine)), [2])
        sparse_map.set_bits_pix(np.arange(int(non_masked_px*nfine)), [20])

        # Test that the fracdet map is equal to the coverage map with same nside_coverage
        fracdet_map1 = sparse_map.fracdet_map(nside_coverage)

        np.testing.assert_array_almost_equal(fracdet_map1[:], sparse_map.coverage_map)

        # Test that the fracdet map is good for target nside
        fracdet_map2 = sparse_map.fracdet_map(nside_fracdet)

        fracdet_map_orig = self.compute_fracdet_map(nside_map, nside_fracdet,
                                                    non_masked_px, nfine)

        np.testing.assert_array_almost_equal(fracdet_map2[:], fracdet_map_orig)

    def test_fracdet_map_raises(self):
        """
        Test limitations of fracdet_map
        """
        nside_coverage = 16
        nside_map = 512
        non_masked_px = 10.5
        nfine = (nside_map//nside_coverage)**2
        full_map = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        full_map[0: int(non_masked_px*nfine)] = 1 + np.random.random(size=int(non_masked_px*nfine))

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage)

        for nside_fracdet in [8, 1024]:
            self.assertRaises(ValueError, sparse_map.fracdet_map, nside_fracdet)

    def compute_fracdet_map(self, nside_map, nside_fracdet, non_masked_px, nfine):
        bit_shift = healsparse.utils._compute_bitshift(nside_fracdet, nside_map)

        fracdet_map_orig = np.zeros(hpg.nside_to_npixel(nside_fracdet), dtype=np.float64)
        idx_frac = np.right_shift(np.arange(int(non_masked_px*nfine)), bit_shift)
        unique_idx_frac = np.unique(idx_frac)
        idx_counts = np.bincount(idx_frac, minlength=hpg.nside_to_npixel(nside_fracdet)).astype(np.float64)
        nfine_frac = (nside_map//nside_fracdet)**2
        fracdet_map_orig[unique_idx_frac] = idx_counts[unique_idx_frac]/nfine_frac

        return fracdet_map_orig


if __name__ == '__main__':
    unittest.main()
