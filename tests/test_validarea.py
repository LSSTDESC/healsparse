import unittest
import numpy.testing as testing
import numpy as np
import hpgeom as hpg
from numpy import random

import healsparse


class ValidAreaTestCase(unittest.TestCase):
    def test_valid_area(self):
        """
        Test getting the valid area of a map
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        n_rand = 1000
        r_indices = np.random.choice(np.arange(hpg.nside_to_npixel(nside_map)),
                                     size=n_rand, replace=False)

        for dt in [np.float64, np.int64]:
            # Create an empty map
            sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dt)

            # Check that the map is make_empty
            testing.assert_equal(sparse_map.get_valid_area(), 0)

            # Fill up the maps
            sparse_map.update_values_pix(r_indices, np.ones(n_rand, dtype=dt))
            testing.assert_equal(sparse_map.get_valid_area(),
                                 n_rand*hpg.nside_to_pixel_area(nside_map, degrees=True))
            testing.assert_equal(sparse_map.n_valid, n_rand)


if __name__ == '__main__':
    unittest.main()
