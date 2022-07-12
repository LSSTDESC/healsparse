import unittest
import numpy.testing as testing
import numpy as np
import hpgeom as hpg
from numpy import random
import pytest
import healsparse

try:
    import healpy as hp
    has_healpy = True
except ImportError:
    has_healpy = False


class UpgradeMapTestCase(unittest.TestCase):
    @pytest.mark.skipif(not has_healpy, reason="Requires healpy")
    def test_upgrade_map(self):
        """
        Test upgrade functionality with regular map.
        """
        random.seed(12345)
        nside_coverage = 32
        nside_map = 256
        nside_new = 1024
        full_map = random.random(hpg.nside_to_npixel(nside_map))

        # Generate sparse map
        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage,
                                              nside_sparse=nside_map)
        # Upgrade original map
        upg_map = hp.ud_grade(full_map, nside_out=nside_new, order_in='NESTED', order_out='NESTED')
        # Upgrade sparse map and compare
        new_map = sparse_map.upgrade(nside_out=nside_new)

        testing.assert_almost_equal(upg_map, new_map.generate_healpix_map())

    @pytest.mark.skipif(not has_healpy, reason="Requires healpy")
    def test_upgrade_map_outoforder(self):
        """
        Test upgrade functionality with an out-of-order map.
        """
        nside_coverage = 32
        nside_map = 256
        nside_new = 1024

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         np.float32)
        sparse_map[100000: 110000] = 1.0
        sparse_map[10000: 20000] = 2.0

        full_map = sparse_map.generate_healpix_map()

        deg_map = hp.ud_grade(full_map, nside_out=nside_new, order_in='NESTED', order_out='NESTED')

        new_map = sparse_map.upgrade(nside_out=nside_new)

        testing.assert_almost_equal(new_map.generate_healpix_map(), deg_map)

    @pytest.mark.skipif(not has_healpy, reason="Requires healpy")
    def test_upgrade_map_recarray(self):
        """
        Test upgrade functionality with a recarray.
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 256
        nside_new = 1024

        dtype = [('col1', 'f8'), ('col2', 'f8'), ('col3', 'i4')]
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype, primary='col1')
        pixel = np.arange(20000)
        values = np.zeros_like(pixel, dtype=dtype)
        values['col1'] = random.random(size=pixel.size)
        values['col2'] = random.random(size=pixel.size)
        values['col3'] = random.poisson(size=pixel.size, lam=2)
        sparse_map.update_values_pix(pixel, values)

        ra, dec = hpg.pixel_to_angle(nside_map, pixel)

        # Make the test values
        hpmap_col1 = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        hpmap_col2 = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        hpmap_col3 = np.zeros(hpg.nside_to_npixel(nside_map)) + hpg.UNSEEN
        hpmap_col1[pixel] = values['col1']
        hpmap_col2[pixel] = values['col2']
        hpmap_col3[pixel] = values['col3']

        # Upgrade healpix maps
        hpmap_col1 = hp.ud_grade(hpmap_col1, nside_out=nside_new, order_in='NESTED', order_out='NESTED')
        hpmap_col2 = hp.ud_grade(hpmap_col2, nside_out=nside_new, order_in='NESTED', order_out='NESTED')
        hpmap_col3 = hp.ud_grade(hpmap_col3, nside_out=nside_new, order_in='NESTED', order_out='NESTED')
        ipnest_test = hpg.angle_to_pixel(nside_new, ra, dec)

        # Upgrade the old map
        new_map = sparse_map.upgrade(nside_out=nside_new)
        testing.assert_almost_equal(new_map.get_values_pos(ra, dec, lonlat=True)['col1'],
                                    hpmap_col1[ipnest_test])
        testing.assert_almost_equal(new_map.get_values_pos(ra, dec, lonlat=True)['col2'],
                                    hpmap_col2[ipnest_test])
        testing.assert_almost_equal(new_map.get_values_pos(ra, dec, lonlat=True)['col3'],
                                    hpmap_col3[ipnest_test])

    def test_upgrade_map_widemask(self):
        """
        Test upgrade functionality with a wide mask.
        (not implemented)
        """
        nside_coverage = 32
        nside_map = 64
        nside_map2 = 256
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         healsparse.WIDE_MASK, wide_mask_maxbits=7)
        self.assertRaises(NotImplementedError, sparse_map.upgrade, nside_map2)


if __name__ == '__main__':
    unittest.main()
