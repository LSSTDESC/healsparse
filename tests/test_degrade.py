import unittest
import numpy.testing as testing
import numpy as np
import hpgeom as hpg
from numpy import random
import tempfile
import os
import shutil
import pytest
import healsparse
from healsparse import WIDE_MASK

try:
    import healpy as hp
    has_healpy = True
except ImportError:
    has_healpy = False


class DegradeMapTestCase(unittest.TestCase):
    @pytest.mark.skipif(not has_healpy, reason="Requires healpy")
    def test_degrade_map_float(self):
        """
        Test HealSparse.degrade functionality with float quantities
        """
        random.seed(12345)
        nside_coverage = 32
        nside_map = 1024
        nside_new = 256
        full_map = random.random(hpg.nside_to_npixel(nside_map))

        # Generate sparse map

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage,
                                              nside_sparse=nside_map)

        # Degrade original HEALPix map

        deg_map = hp.ud_grade(full_map, nside_out=nside_new, order_in='NESTED', order_out='NESTED')

        # Degrade sparse map and compare to original

        new_map = sparse_map.degrade(nside_out=nside_new)

        # Test the coverage map generation and lookup

        testing.assert_almost_equal(deg_map, new_map.generate_healpix_map())

        # Test degrade-on-read
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        fname = os.path.join(self.test_dir, 'test_float_degrade.hs')
        sparse_map.write(fname)

        new_map2 = healsparse.HealSparseMap.read(fname, degrade_nside=nside_new)

        testing.assert_almost_equal(deg_map, new_map2.generate_healpix_map())

    @pytest.mark.skipif(not has_healpy, reason="Requires healpy")
    def test_degrade_map_float_outoforder(self):
        """
        Test HealSparse.degrade functionality with float quantities and
        an out-of-order map.
        """
        nside_coverage = 32
        nside_map = 1024
        nside_new = 256

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         np.float32)
        sparse_map[100000: 110000] = 1.0
        sparse_map[10000: 20000] = 2.0

        full_map = sparse_map.generate_healpix_map()

        deg_map = hp.ud_grade(full_map, nside_out=nside_new, order_in='NESTED', order_out='NESTED')

        new_map = sparse_map.degrade(nside_out=nside_new)

        testing.assert_almost_equal(new_map.generate_healpix_map(), deg_map)

    @pytest.mark.skipif(not has_healpy, reason="Requires healpy")
    def test_degrade_map_int(self):
        """
        Test HealSparse.degrade functionality with int quantities
        """
        random.seed(12345)
        nside_coverage = 32
        nside_map = 1024
        nside_new = 256
        full_map = random.poisson(size=hpg.nside_to_npixel(nside_map), lam=2)

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

        # Test degrade-on-read
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        fname = os.path.join(self.test_dir, 'test_int_degrade.hs')
        sparse_map.write(fname)

        new_map2 = healsparse.HealSparseMap.read(fname, degrade_nside=nside_new)

        testing.assert_almost_equal(deg_map, new_map2.generate_healpix_map())

    @pytest.mark.skipif(not has_healpy, reason="Requires healpy")
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

        # Degrade healpix maps
        hpmap_col1 = hp.ud_grade(hpmap_col1, nside_out=nside_new, order_in='NESTED', order_out='NESTED')
        hpmap_col2 = hp.ud_grade(hpmap_col2, nside_out=nside_new, order_in='NESTED', order_out='NESTED')
        hpmap_col3 = hp.ud_grade(hpmap_col3, nside_out=nside_new, order_in='NESTED', order_out='NESTED')
        ipnest_test = hpg.angle_to_pixel(nside_new, ra, dec)

        # Degrade the old map
        new_map = sparse_map.degrade(nside_out=nside_new)
        testing.assert_almost_equal(new_map.get_values_pos(ra, dec, lonlat=True)['col1'],
                                    hpmap_col1[ipnest_test])
        testing.assert_almost_equal(new_map.get_values_pos(ra, dec, lonlat=True)['col2'],
                                    hpmap_col2[ipnest_test])
        testing.assert_almost_equal(new_map.get_values_pos(ra, dec, lonlat=True)['col3'],
                                    hpmap_col3[ipnest_test])

        # Test degrade-on-read
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        fname = os.path.join(self.test_dir, 'test_recarray_degrade.hs')
        sparse_map.write(fname)

        new_map2 = healsparse.HealSparseMap.read(fname, degrade_nside=nside_new)

        testing.assert_almost_equal(new_map2.get_values_pos(ra, dec, lonlat=True)['col1'],
                                    hpmap_col1[ipnest_test])
        testing.assert_almost_equal(new_map2.get_values_pos(ra, dec, lonlat=True)['col2'],
                                    hpmap_col2[ipnest_test])
        testing.assert_almost_equal(new_map2.get_values_pos(ra, dec, lonlat=True)['col3'],
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

        # Repeat for maxbits > 8
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         WIDE_MASK, wide_mask_maxbits=16)
        sparse_map_or = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map2,
                                                            WIDE_MASK, wide_mask_maxbits=16)
        # Fill some pixels in the "high-resolution" map
        pixel = np.arange(0, 1024)
        pixel = np.concatenate([pixel[:512], pixel[512::3]]).ravel()
        sparse_map.set_bits_pix(pixel, [4, 12])
        sparse_map.clear_bits_pix(pixel[:16], [4])  # set low value in the first pixel

        # Check which pixels will be full in the "low-resolution" map and fill them
        # Note that we are filling more than the ones that are going to be True
        # since we want to preserve the coverage_map
        pixel2_all = np.unique(np.right_shift(pixel,
                               healsparse.utils._compute_bitshift(nside_map2, nside_map)))
        sparse_map_or.set_bits_pix(pixel2_all, [4, 12])

        # Get the pixel number of the bad pixels
        pixel2_bad = np.array([0])
        sparse_map_or.clear_bits_pix(pixel2_bad, [4])  # set low value in the first pixel

        # Degrade with or
        sparse_map_test = sparse_map.degrade(nside_map2, reduction='or')

        # Check the results
        testing.assert_almost_equal(sparse_map_test._sparse_map, sparse_map_or._sparse_map)

        # Test degrade-on-read
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        fname = os.path.join(self.test_dir, 'test_wide_degrade.hs')
        sparse_map.write(fname)

        sparse_map_test2 = healsparse.HealSparseMap.read(fname, degrade_nside=nside_map2, reduction='or')
        testing.assert_almost_equal(sparse_map_test2._sparse_map, sparse_map_or._sparse_map)

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

        # Repeat for maxbits > 8
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         WIDE_MASK, wide_mask_maxbits=16)
        sparse_map_and = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map2,
                                                             WIDE_MASK, wide_mask_maxbits=16)
        # Fill some pixels in the "high-resolution" map
        pixel = np.arange(0, 1024)
        pixel = np.concatenate([pixel[:512], pixel[512::3]]).ravel()
        sparse_map.set_bits_pix(pixel, [4, 12])
        sparse_map.clear_bits_pix(pixel[:16], [4])  # set low value in the first pixel

        # Check which pixels will be full in the "low-resolution" map and fill them
        # Note that we are filling more than the ones that are going to be True
        # since we want to preserve the coverage_map
        pixel2_all = np.unique(np.right_shift(pixel,
                               healsparse.utils._compute_bitshift(nside_map2, nside_map)))
        sparse_map_and.set_bits_pix(pixel2_all, [4, 12])

        # Get the pixel number of the bad pixels
        pixel2_bad = np.unique(np.right_shift(pixel[512:],
                               healsparse.utils._compute_bitshift(nside_map2, nside_map)))
        sparse_map_and.clear_bits_pix(pixel2_bad, [4, 12])
        sparse_map_and.clear_bits_pix(pixel2_all[0], [4])  # set low value in the first pixel

        # Degrade with and
        sparse_map_test = sparse_map.degrade(nside_map2, reduction='and')

        # Check the results
        testing.assert_almost_equal(sparse_map_and._sparse_map, sparse_map_test._sparse_map)

        # Testing the degrade-on-read is redundant from the or case.

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
        testing.assert_almost_equal(sparse_map_test._sparse_map, sparse_map_or._sparse_map)

        # Test degrade-on-read
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        fname = os.path.join(self.test_dir, 'test_intor_degrade.hs')
        sparse_map.write(fname)

        sparse_map_test2 = healsparse.HealSparseMap.read(fname, degrade_nside=nside_map2, reduction='or')
        testing.assert_almost_equal(sparse_map_test2._sparse_map, sparse_map_or._sparse_map)

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

        # Testing the degrade-on-read is redundant from the or case.

    def test_degrade_map_float_prod(self):
        """
        Test HealSparse.degrade product with floats
        """
        nside_coverage = 32
        nside_map = 1024
        nside_new = 512
        full_map = np.full(hpg.nside_to_npixel(nside_map), 2.)
        # Generate sparse map

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage,
                                              nside_sparse=nside_map)

        # Degrade original HEALPix map

        deg_map = np.full(hpg.nside_to_npixel(nside_new), 2.**4)
        # Degrade sparse map and compare to original

        new_map = sparse_map.degrade(nside_out=nside_new, reduction='prod')

        # Test the coverage map generation and lookup

        testing.assert_almost_equal(deg_map, new_map.generate_healpix_map())

    def test_degrade_map_int_prod(self):
        """
        Test HealSparse.degrade product with integers
        """
        nside_coverage = 32
        nside_map = 1024
        nside_new = 512
        full_map = np.full(hpg.nside_to_npixel(nside_map), 2, dtype=np.int64)
        # Generate sparse map

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage,
                                              nside_sparse=nside_map, sentinel=0)

        # Degrade original HEALPix map

        deg_map = np.full(hpg.nside_to_npixel(nside_new), 2**4, dtype=np.int64)
        # Degrade sparse map and compare to original

        new_map = sparse_map.degrade(nside_out=nside_new, reduction='prod')
        # Test the coverage map generation and lookup

        testing.assert_almost_equal(deg_map, new_map.generate_healpix_map())

    def test_degrade_map_float_sum(self):
        """
        Test HealSparse.degrade sum with float
        """
        nside_coverage = 32
        nside_map = 1024
        nside_new = 512
        full_map = np.full(hpg.nside_to_npixel(nside_map), 1.)
        # Generate sparse map

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage,
                                              nside_sparse=nside_map)

        # Degrade original HEALPix map

        deg_map = np.full(hpg.nside_to_npixel(nside_new), 4.)
        # Degrade sparse map and compare to original

        new_map = sparse_map.degrade(nside_out=nside_new, reduction='sum')

        # Test the coverage map generation and lookup

        testing.assert_almost_equal(deg_map, new_map.generate_healpix_map())

    def test_degrade_map_int_sum(self):
        """
        Test HealSparse.degrade sum with integers
        """
        nside_coverage = 32
        nside_map = 1024
        nside_new = 512
        full_map = np.full(hpg.nside_to_npixel(nside_map), 1, dtype=np.int64)
        # Generate sparse map

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage,
                                              nside_sparse=nside_map, sentinel=0)

        # Degrade original HEALPix map

        deg_map = np.full(hpg.nside_to_npixel(nside_new), 4, dtype=np.int64)
        # Degrade sparse map and compare to original

        new_map = sparse_map.degrade(nside_out=nside_new, reduction='sum')
        # Test the coverage map generation and lookup

        testing.assert_almost_equal(deg_map, new_map.generate_healpix_map())

    def test_degrade_map_wmean(self):
        """
        Test HealSparse.degrade wmean functionality with float quantities
        """
        nside_coverage = 32
        nside_map = 1024
        nside_new = 256
        full_map = np.ones(hpg.nside_to_npixel(nside_map))
        weights = np.ones_like(full_map)
        full_map[::32] = 0.5  # We lower the value in 1 pixel every 32
        weights[::32] = 0.5  # We downweight 1 pixel every 32
        deg_map = np.ones(hpg.nside_to_npixel(nside_new))
        deg_map[::2] = 15.25/15.5
        # Generate sparse map

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage,
                                              nside_sparse=nside_map)
        weights = healsparse.HealSparseMap(healpix_map=weights, nside_coverage=nside_coverage,
                                           nside_sparse=nside_map)

        # Degrade sparse map and compare to original
        new_map = sparse_map.degrade(nside_out=nside_new, reduction='wmean', weights=weights)

        # Test the coverage map generation and lookup
        testing.assert_almost_equal(new_map.generate_healpix_map(), deg_map)

        # Test degrade-on-read with weights
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        fname = os.path.join(self.test_dir, 'test_float_degrade.hs')
        sparse_map.write(fname)
        fname_weight = os.path.join(self.test_dir, 'test_float_degrade_weights.hs')
        weights.write(fname_weight)

        new_map2 = healsparse.HealSparseMap.read(fname, degrade_nside=nside_new, reduction='wmean',
                                                 weightfile=fname_weight)
        testing.assert_almost_equal(new_map2.generate_healpix_map(), deg_map)

    def test_degrade_map_int_wmean(self):
        """
        Test HealSparse.degrade wmean with integers
        """
        nside_coverage = 32
        nside_map = 1024
        nside_new = 512
        full_map = np.full(hpg.nside_to_npixel(nside_map), 1, dtype=np.int64)
        # Generate sparse map

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage,
                                              nside_sparse=nside_map, sentinel=0)
        weights = healsparse.HealSparseMap(healpix_map=np.ones(len(full_map)), nside_coverage=nside_coverage,
                                           nside_sparse=nside_map)
        # Degrade original HEALPix map

        deg_map = np.full(hpg.nside_to_npixel(nside_new), 1, dtype=np.int64)
        # Degrade sparse map and compare to original

        new_map = sparse_map.degrade(nside_out=nside_new, reduction='wmean', weights=weights)
        # Test the coverage map generation and lookup
        testing.assert_almost_equal(deg_map, new_map.generate_healpix_map())

    def test_degrade_widemask_wmean(self):
        """
        Test HealSparse.degrade AND functionality with WIDE_MASK
        """

        nside_coverage = 32
        nside_map = 256
        nside_out = 64
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         WIDE_MASK, wide_mask_maxbits=7)

        weights = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.float32)

        testing.assert_raises(NotImplementedError,
                              sparse_map.degrade, nside_out=nside_out, reduction='wmean',
                              weights=weights)

    def test_degrade_lowres_float(self):
        """
        Test HealSparse.degrade in the case where the target resolution
        is smaller than the original coverage resolution (nside_out < nside_coverage)
        """
        random.seed(12345)
        nside_coverage = 32
        nside_map = 256
        nside_out = 8
        pxnums = np.arange(2000)
        pxvalues = random.random(size=2000).astype(np.float64)
        weights = 0.5+0.5*random.random(size=2000).astype(np.float64)
        methods = ['mean', 'std', 'max', 'mean', 'median', 'wmean', 'sum', 'prod']
        for method in methods:
            if method != 'wmean':
                wgt = None
                wgt2 = None
            else:
                wgt = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype=np.float64)
                wgt2 = healsparse.HealSparseMap.make_empty(nside_out, nside_map, dtype=np.float64)
                wgt[pxnums] = weights
                wgt2[pxnums] = weights
            sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype=np.float64)
            sparse_map2 = healsparse.HealSparseMap.make_empty(nside_out, nside_map, dtype=np.float64)
            sparse_map.update_values_pix(pxnums, pxvalues)
            sparse_map2.update_values_pix(pxnums, pxvalues)
            sparse_map2 = sparse_map2.degrade(nside_out, reduction=method, weights=wgt2)
            with self.assertWarns(ResourceWarning):
                sparse_map = sparse_map.degrade(nside_out, reduction=method, weights=wgt)
            testing.assert_almost_equal(sparse_map.coverage_map, sparse_map2.coverage_map)
            testing.assert_almost_equal(sparse_map._sparse_map, sparse_map2._sparse_map)

    def test_degrade_lowres_int(self):
        """
        Test HealSparse.degrade in the case where the target resolution
        is smaller than the original coverage resolution (nside_out < nside_coverage)
        """
        random.seed(12345)
        nside_coverage = 32
        nside_map = 256
        nside_out = 8
        pxnums = np.arange(2000)
        pxvalues = random.randint(1, 5, size=2000)
        weights = 0.5+0.5*random.random(size=2000).astype(np.float64)
        methods = ['mean', 'std', 'max', 'mean', 'median', 'wmean', 'sum', 'prod', 'and', 'or']
        for method in methods:
            if method != 'wmean':
                wgt = None
                wgt2 = None
            else:
                wgt = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype=np.float64)
                wgt2 = healsparse.HealSparseMap.make_empty(nside_out, nside_map, dtype=np.float64)
                wgt[pxnums] = weights
                wgt2[pxnums] = weights
            sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype=pxvalues.dtype)
            sparse_map2 = healsparse.HealSparseMap.make_empty(nside_out, nside_map, dtype=pxvalues.dtype)
            sparse_map.update_values_pix(pxnums, pxvalues)
            sparse_map2.update_values_pix(pxnums, pxvalues)
            sparse_map2 = sparse_map2.degrade(nside_out, reduction=method, weights=wgt2)
            with self.assertWarns(ResourceWarning):
                sparse_map = sparse_map.degrade(nside_out, reduction=method, weights=wgt)
            testing.assert_almost_equal(sparse_map.coverage_map, sparse_map2.coverage_map)
            testing.assert_almost_equal(sparse_map._sparse_map, sparse_map2._sparse_map)

    def test_degrade_lowres_wide(self):
        """
        Test HealSparse.degrade in the case where the target resolution
        is smaller than the original coverage resolution (nside_out < nside_coverage)
        """

        nside_coverage = 32
        nside_map = 256
        nside_out = 8
        pixel = np.arange(0, 1024)
        pixel = np.concatenate([pixel[:512], pixel[512::3]]).ravel()
        for method in ['and', 'or']:
            sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                             WIDE_MASK, wide_mask_maxbits=7)
            sparse_map2 = healsparse.HealSparseMap.make_empty(nside_out, nside_map,
                                                              WIDE_MASK, wide_mask_maxbits=7)

            sparse_map.set_bits_pix(pixel, [4])
            sparse_map2.set_bits_pix(pixel, [4])
            sparse_map2 = sparse_map2.degrade(nside_out, reduction=method)
            with self.assertWarns(ResourceWarning):
                sparse_map = sparse_map.degrade(nside_out, reduction=method)
            testing.assert_almost_equal(sparse_map.coverage_map, sparse_map2.coverage_map)
            testing.assert_almost_equal(sparse_map._sparse_map, sparse_map2._sparse_map)

    @pytest.mark.skipif(not has_healpy, reason="Requires healpy")
    def test_degrade_map_bool(self):
        """
        Test HealSparse.degrade functionality with bool quantities
        """
        random.seed(12345)
        nside_coverage = 32
        nside_map = 1024
        nside_new = 256

        full_map = np.zeros(hpg.nside_to_npixel(nside_map), dtype=bool)
        pixels = np.random.choice(full_map.size, size=full_map.size//4, replace=False)
        full_map[pixels] = True

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.bool_)
        sparse_map[pixels] = full_map[pixels]

        # Degrade original map
        test_map = full_map.astype(np.float64)
        test_map[~full_map] = hpg.UNSEEN
        deg_map = hp.ud_grade(test_map, nside_out=nside_new,
                              order_in='NESTED', order_out='NESTED')

        # Degrade sparse map and compare to original
        new_map = sparse_map.degrade(nside_out=nside_new)

        # Test the coverage map generation and lookup
        testing.assert_almost_equal(deg_map, new_map.generate_healpix_map())

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__ == '__main__':
    unittest.main()
