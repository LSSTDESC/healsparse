import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp

import healsparse


class GetFromHealpixCase(unittest.TestCase):
    def test_from_healpix_float(self):
        """
        Test converting healpix map (float type)
        """
        np.random.seed(12345)

        nside_coverage = 32
        nside_map = 128

        full_map = np.zeros(hp.nside2npix(nside_map)) + hp.UNSEEN
        full_map[0: 5000] = np.random.random(size=5000)

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage)
        testing.assert_almost_equal(full_map, sparse_map.get_values_pix(np.arange(full_map.size)))

        self.assertRaises(ValueError, healsparse.HealSparseMap, healpix_map=full_map,
                          nside_coverage=nside_coverage, sentinel=int(hp.UNSEEN))

    def test_from_healpix_int(self):
        """
        Test converting healpix map (int type)
        """
        np.random.seed(12345)

        nside_coverage = 32
        nside_map = 128

        full_map = np.zeros(hp.nside2npix(nside_map), dtype=np.int32) + np.iinfo(np.int32).min
        full_map[0: 5000] = np.random.poisson(size=5000)

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map,
                                              nside_coverage=nside_coverage,
                                              sentinel=np.iinfo(np.int32).min)
        testing.assert_array_equal(full_map,
                                   sparse_map.get_values_pix(np.arange(full_map.size)))

        self.assertRaises(ValueError, healsparse.HealSparseMap, healpix_map=full_map,
                          nside_coverage=nside_coverage)


if __name__ == '__main__':
    unittest.main()
