import unittest
import numpy.testing as testing
import numpy as np
import hpgeom as hpg
import pytest
import healsparse

try:
    import healpy as hp
    has_healpy = True
except ImportError:
    has_healpy = False


class InterpolateMapTestCase(unittest.TestCase):
    @pytest.mark.skipif(not has_healpy, reason="Requires healpy")
    def test_interpolate_map_float(self):
        """
        Test interpolate_pos functionality with float quantities.
        """
        np.random.seed(12345)
        nside_coverage = 32
        nside_map = 64

        for dtype in [np.float32, np.float64]:

            sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype)
            npix = hpg.nside_to_npixel(nside_map)
            sparse_map[0: npix] = np.random.uniform(low=0.0, high=100.0, size=npix).astype(dtype)

            ra = np.random.uniform(low=0.0, high=360.0, size=10_000)
            dec = np.random.uniform(low=-90.0, high=90.0, size=10_000)

            interp_hsp = sparse_map.interpolate_pos(ra, dec)

            hpmap = sparse_map.generate_healpix_map()
            interp_hp = hp.get_interp_val(hpmap, ra, dec, lonlat=True, nest=True)

            testing.assert_array_almost_equal(interp_hsp, interp_hp)

    def test_interpolate_map_float_sentinel(self):
        """
        Test interpolate_pos with float quantities and a sentinel value.
        """
        np.random.seed(12345)
        nside_coverage = 32
        nside_map = 64

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.float64)
        npix = hpg.nside_to_npixel(nside_map)
        sparse_map[0: npix] = np.random.uniform(low=0.0, high=100.0, size=npix)

        ra = np.random.uniform(low=0.0, high=360.0, size=100)
        dec = np.random.uniform(low=-90.0, high=90.0, size=100)

        bad_pixel = 100
        sparse_map[bad_pixel] = None
        ra_unseen, dec_unseen = hpg.pixel_to_angle(nside_map, bad_pixel)

        ra[1] = ra_unseen + 0.01
        dec[1] = dec_unseen + 0.01

        vals = sparse_map.interpolate_pos(ra, dec)

        # Check that only the one is UNSEEN.
        testing.assert_almost_equal(vals[1], hpg.UNSEEN)
        ok, = np.where(vals != hpg.UNSEEN)
        self.assertEqual(len(ok), len(ra) - 1)

        # And check that if we call just the one pos it works.
        vals2 = sparse_map.interpolate_pos(ra[1], dec[1])
        self.assertEqual(len(vals2), 1)
        testing.assert_almost_equal(vals2[0], hpg.UNSEEN)

        # Check with allow_partial
        vals = sparse_map.interpolate_pos(ra, dec, allow_partial=True)
        ok, = np.where(vals != hpg.UNSEEN)
        self.assertEqual(len(ok), len(ra))

        # And check that if we call just the one pos it works.
        vals2 = sparse_map.interpolate_pos(ra[1], dec[1], allow_partial=True)
        self.assertEqual(len(vals2), 1)
        self.assertTrue(vals2[0] != hpg.UNSEEN)

        # And check that these values are equal and make sense with a by-hand computation.
        testing.assert_almost_equal(vals2[0], vals[1])
        pix, wgt = hpg.get_interpolation_weights(nside_map, ra[1], dec[1], lonlat=True)
        values = sparse_map[pix]
        values_valid = (values != hpg.UNSEEN)
        val_comp = np.sum(values[values_valid]*wgt[values_valid])/np.sum(wgt[values_valid])
        testing.assert_almost_equal(vals2[0], val_comp)

    def test_interpolate_map_float_all_sentinel(self):
        """
        Test interpolate_pos with float quantities and all sentinel values.
        """
        nside_coverage = 32
        nside_map = 64

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.float64)

        ra = np.random.uniform(low=0.0, high=360.0, size=100)
        dec = np.random.uniform(low=-90.0, high=90.0, size=100)

        vals = sparse_map.interpolate_pos(ra, dec)
        testing.assert_array_almost_equal(vals, hpg.UNSEEN)

        vals2 = sparse_map.interpolate_pos(ra, dec, allow_partial=True)
        testing.assert_array_almost_equal(vals2, hpg.UNSEEN)

    @pytest.mark.skipif(not has_healpy, reason="Requires healpy")
    def test_interpolate_map_int(self):
        """
        Test interpolate_pos functionality with int quantities.
        """
        np.random.seed(12345)
        nside_coverage = 32
        nside_map = 64

        for dtype in [np.int32, np.int64]:

            sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype)
            npix = hpg.nside_to_npixel(nside_map)
            sparse_map[0: npix] = np.random.poisson(size=npix, lam=10).astype(dtype)

            ra = np.random.uniform(low=0.0, high=360.0, size=10_000)
            dec = np.random.uniform(low=-90.0, high=90.0, size=10_000)

            interp_hsp = sparse_map.interpolate_pos(ra, dec)

            hpmap = sparse_map.generate_healpix_map()
            interp_hp = hp.get_interp_val(hpmap, ra, dec, lonlat=True, nest=True)

            testing.assert_array_almost_equal(interp_hsp, interp_hp)

    def test_interpolate_map_wide_mask(self):
        """
        Test interpolate_pos functionality with wide_mask quantities.
        """
        sparse_map = healsparse.HealSparseMap.make_empty(32, 64, healsparse.WIDE_MASK, wide_mask_maxbits=2)

        with self.assertRaises(NotImplementedError):
            sparse_map.interpolate_pos(0.0, 0.0)

    def test_interpolate_map_recarray(self):
        """
        Test interpolate_pos functionality with recarray quantities.
        """
        dtype = [("a", "f8"), ("b", "f4")]
        sparse_map = healsparse.HealSparseMap.make_empty(32, 64, dtype, primary="a")

        with self.assertRaises(NotImplementedError):
            sparse_map.interpolate_pos(0.0, 0.0)

    def test_interpolate_map_bool(self):
        """
        Test interpolate_pos functionality with bool quantities.
        """
        sparse_map = healsparse.HealSparseMap.make_empty(32, 64, bool, primary=False)

        with self.assertRaises(NotImplementedError):
            sparse_map.interpolate_pos(0.0, 0.0)


if __name__ == '__main__':
    unittest.main()
