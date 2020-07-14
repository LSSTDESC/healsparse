from __future__ import division, absolute_import, print_function

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
from numpy import random
import healsparse
from healsparse import WIDE_MASK


class DegradeMapTestCase(unittest.TestCase):
    def test_degrade_map_float(self):
        """
        Test HealSparse.degrade functionality with float quantities
        """
        random.seed(12345)
        nside_coverage = 32
        nside_map = 1024
        nside_new = 256
        full_map = random.random(hp.nside2npix(nside_map))

        # Generate sparse map

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage,
                                              nside_sparse=nside_map)

        # Degrade original HEALPix map

        deg_map = hp.ud_grade(full_map, nside_out=nside_new, order_in='NESTED', order_out='NESTED')

        # Degrade sparse map and compare to original

        new_map = sparse_map.degrade(nside_out=nside_new)

        # Test the coverage map generation and lookup

        testing.assert_almost_equal(deg_map, new_map.generate_healpix_map())

    def test_degrade_map_int(self):
        """
        Test HealSparse.degrade functionality with int quantities
        """

        random.seed(12345)
        nside_coverage = 32
        nside_map = 1024
        nside_new = 256
        full_map = random.poisson(size=hp.nside2npix(nside_map), lam=2)

        # Generate sparse map
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.int64)
        sparse_map.update_values_pix(np.arange(full_map.size), full_map)

        # Degrade original HEALPix map

        deg_map = hp.ud_grade(full_map.astype(np.float64), nside_out=nside_new,
                              order_in='NESTED', order_out='NESTED')

        # Degrade sparse map and compare to original
        new_map = sparse_map.degrade(nside_out=nside_new)

        # Test the coverage map generation and lookup

        testing.assert_almost_equal(deg_map, new_map.generate_healpix_map())

    def test_degrade_map_recarray(self):
        """
        Test HealSparse.degrade functionality with recarray quantities
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 1024
        nside_new = 256

        dtype = [('col1', 'f8'), ('col2', 'f8'), ('col3', 'i4')]
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype, primary='col1')
        pixel = np.arange(20000)
        values = np.zeros_like(pixel, dtype=dtype)
        values['col1'] = np.random.random(size=pixel.size)
        values['col2'] = np.random.random(size=pixel.size)
        values['col3'] = np.random.poisson(size=pixel.size, lam=2)
        sparse_map.update_values_pix(pixel, values)

        theta, phi = hp.pix2ang(nside_map, pixel, nest=True)
        ra = np.degrees(phi)
        dec = 90.0 - np.degrees(theta)

        # Make the test values
        hpmap_col1 = np.zeros(hp.nside2npix(nside_map)) + hp.UNSEEN
        hpmap_col2 = np.zeros(hp.nside2npix(nside_map)) + hp.UNSEEN
        hpmap_col3 = np.zeros(hp.nside2npix(nside_map)) + hp.UNSEEN
        hpmap_col1[pixel] = values['col1']
        hpmap_col2[pixel] = values['col2']
        hpmap_col3[pixel] = values['col3']

        # Degrade healpix maps
        hpmap_col1 = hp.ud_grade(hpmap_col1, nside_out=nside_new, order_in='NESTED', order_out='NESTED')
        hpmap_col2 = hp.ud_grade(hpmap_col2, nside_out=nside_new, order_in='NESTED', order_out='NESTED')
        hpmap_col3 = hp.ud_grade(hpmap_col3, nside_out=nside_new, order_in='NESTED', order_out='NESTED')
        ipnest_test = hp.ang2pix(nside_new, theta, phi, nest=True)

        # Degrade the old map
        newSparseMap = sparse_map.degrade(nside_out=nside_new)
        testing.assert_almost_equal(newSparseMap.get_values_pos(ra, dec, lonlat=True)['col1'],
                                    hpmap_col1[ipnest_test])
        testing.assert_almost_equal(newSparseMap.get_values_pos(ra, dec, lonlat=True)['col2'],
                                    hpmap_col2[ipnest_test])
        testing.assert_almost_equal(newSparseMap.get_values_pos(ra, dec, lonlat=True)['col3'],
                                    hpmap_col3[ipnest_test])

    def test_degrade_widemask_or(self):
        """
        Test HealSparse.degrade OR functionality with WIDE_MASK
        """

        nside_coverage = 32
        nside_map = 256
        nside_map2 = 64
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         WIDE_MASK, wide_mask_maxbits=7)
        sparse_map_or = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map2,
                                                            WIDE_MASK, wide_mask_maxbits=7)
        # Fill some pixels in the "high-resolution" map
        pixel = np.arange(4000, 8000)
        sparse_map.set_bits_pix(pixel, [4])

        # Check which pixels will be full in the "low-resolution" map and fill them
        pixel2 = np.unique(np.right_shift(pixel, healsparse.utils._compute_bitshift(nside_map2, nside_map)))
        sparse_map_or.set_bits_pix(pixel2, [4])

        # Degrade with or
        sparse_map_test = sparse_map.degrade(nside_map2, reduction='or')

        # Check the results
        testing.assert_almost_equal(sparse_map_or._sparse_map, sparse_map_test._sparse_map)

    def test_degrade_widemask_and(self):
        """
        Test HealSparse.degrade AND functionality with WIDE_MASK
        """

        nside_coverage = 32
        nside_map = 256
        nside_map2 = 64
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         WIDE_MASK, wide_mask_maxbits=7)
        sparse_map_and = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map2,
                                                             WIDE_MASK, wide_mask_maxbits=7)
        # Fill some pixels in the "high-resolution" map
        pixel = np.arange(0, 1024)
        pixel = np.concatenate([pixel[:512], pixel[512::3]]).ravel()
        sparse_map.set_bits_pix(pixel, [4])

        # Check which pixels will be full in the "low-resolution" map and fill them
        pixel2_all = np.unique(np.right_shift(pixel,
                               healsparse.utils._compute_bitshift(nside_map2, nside_map)))
        sparse_map_and.set_bits_pix(pixel2_all, [4])

        # Get the pixel number of the bad pixels
        pixel2_bad = np.unique(np.right_shift(pixel[512:],
                               healsparse.utils._compute_bitshift(nside_map2, nside_map)))
        sparse_map_and.clear_bits_pix(pixel2_bad, [4])

        # Degrade with and
        sparse_map_test = sparse_map.degrade(nside_map2, reduction='and')

        # Check the results
        testing.assert_almost_equal(sparse_map_and._sparse_map, sparse_map_test._sparse_map)

    def test_degrade_int_or(self):
        """
        Test HealSparse.degrade OR functionality with integer maps
        """

        nside_coverage = 32
        nside_map = 256
        nside_map2 = 64
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         np.int64)
        sparse_map_or = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map2,
                                                            np.int64)
        # Fill some pixels in the "high-resolution" map
        pixel = np.arange(4000, 8000)
        sparse_map.update_values_pix(pixel, pixel)

        # Check which pixels will be full in the "low-resolution" map and fill them
        pixel2 = np.unique(np.right_shift(pixel,
                           healsparse.utils._compute_bitshift(nside_map2, nside_map)))
        px2val = np.arange(4000+(nside_map//nside_map2)**2-1,
                           8000+(nside_map//nside_map2)**2-1,
                           (nside_map//nside_map2)**2)
        sparse_map_or.update_values_pix(pixel2, px2val)

        # Degrade with or
        sparse_map_test = sparse_map.degrade(nside_map2, reduction='or')

        # Check the results
        testing.assert_almost_equal(sparse_map_or._sparse_map, sparse_map_test._sparse_map)

    def test_degrade_int_and(self):
        """
        Test HealSparse.degrade AND functionality with integer maps
        """

        nside_coverage = 32
        nside_map = 256
        nside_map2 = 64
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         np.int64)
        sparse_map_and = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map2,
                                                             np.int64)
        # Fill some pixels in the "high-resolution" map
        pixel = np.arange(4000, 8000)
        sparse_map.update_values_pix(pixel, pixel)

        # Check which pixels will be full in the "low-resolution" map and fill them
        pixel2 = np.unique(np.right_shift(pixel,
                           healsparse.utils._compute_bitshift(nside_map2, nside_map)))
        px2val = np.arange(4000, 8000,
                           (nside_map//nside_map2)**2)
        sparse_map_and.update_values_pix(pixel2, px2val)

        # Degrade with or
        sparse_map_test = sparse_map.degrade(nside_map2, reduction='and')

        # Check the results
        testing.assert_almost_equal(sparse_map_and._sparse_map, sparse_map_test._sparse_map)


if __name__ == '__main__':
    unittest.main()
