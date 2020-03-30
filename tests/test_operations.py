from __future__ import division, absolute_import, print_function

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
from numpy import random

import healsparse


class OperationsTestCase(unittest.TestCase):
    def test_sum(self):
        """
        Test map addition.
        """

        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        # Test adding two or three maps

        sparse_map1 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.float64)
        pixel1 = np.arange(4000, 20000)
        pixel1 = np.delete(pixel1, 15000)
        values1 = np.random.random(size=pixel1.size)
        sparse_map1.update_values_pix(pixel1, values1)
        hpmap1 = sparse_map1.generate_healpix_map()

        sparse_map2 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.float64)
        pixel2 = np.arange(15000, 25000)
        values2 = np.random.random(size=pixel2.size)
        sparse_map2.update_values_pix(pixel2, values2)
        hpmap2 = sparse_map2.generate_healpix_map()

        sparse_map3 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.float64)
        pixel3 = np.arange(16000, 25000)
        values3 = np.random.random(size=pixel3.size)
        sparse_map3.update_values_pix(pixel3, values3)
        hpmap3 = sparse_map3.generate_healpix_map()

        # Intersection addition

        # sum 2
        added_map_intersection = healsparse.sum_intersection([sparse_map1, sparse_map2])

        gd, = np.where((hpmap1 > hp.UNSEEN) & (hpmap2 > hp.UNSEEN))
        hpmap_sum_intersection = np.zeros_like(hpmap1) + hp.UNSEEN
        hpmap_sum_intersection[gd] = hpmap1[gd] + hpmap2[gd]

        testing.assert_almost_equal(hpmap_sum_intersection, added_map_intersection.generate_healpix_map())

        # sum 3
        added_map_intersection = healsparse.sum_intersection([sparse_map1, sparse_map2, sparse_map3])

        gd, = np.where((hpmap1 > hp.UNSEEN) & (hpmap2 > hp.UNSEEN) & (hpmap3 > hp.UNSEEN))
        hpmap_sum_intersection = np.zeros_like(hpmap1) + hp.UNSEEN
        hpmap_sum_intersection[gd] = hpmap1[gd] + hpmap2[gd] + hpmap3[gd]

        testing.assert_almost_equal(hpmap_sum_intersection, added_map_intersection.generate_healpix_map())

        # Union addition

        # sum 2
        added_map_union = healsparse.sum_union([sparse_map1, sparse_map2])

        gd, = np.where((hpmap1 > hp.UNSEEN) | (hpmap2 > hp.UNSEEN))
        hpmap_sum_union = np.zeros_like(hpmap1) + hp.UNSEEN
        # This hack works because we don't have summands going below zero...
        hpmap_sum_union[gd] = np.clip(hpmap1[gd], 0.0, None) + np.clip(hpmap2[gd], 0.0, None)

        testing.assert_almost_equal(hpmap_sum_union, added_map_union.generate_healpix_map())

        # sum 3
        added_map_union = healsparse.sum_union([sparse_map1, sparse_map2, sparse_map3])

        gd, = np.where((hpmap1 > hp.UNSEEN) | (hpmap2 > hp.UNSEEN) | (hpmap3 > hp.UNSEEN))
        hpmap_sum_union = np.zeros_like(hpmap1) + hp.UNSEEN
        # This hack works because we don't have summands going below zero...
        hpmap_sum_union[gd] = (np.clip(hpmap1[gd], 0.0, None) +
                               np.clip(hpmap2[gd], 0.0, None) +
                               np.clip(hpmap3[gd], 0.0, None))

        testing.assert_almost_equal(hpmap_sum_union, added_map_union.generate_healpix_map())

        # Test adding an int constant to a map

        added_map = sparse_map1 + 2

        hpmapAdd2 = np.zeros_like(hpmap1) + hp.UNSEEN
        gd, = np.where(hpmap1 > hp.UNSEEN)
        hpmapAdd2[gd] = hpmap1[gd] + 2

        testing.assert_almost_equal(hpmapAdd2, added_map.generate_healpix_map())

        # Test adding a float constant to a map

        added_map = sparse_map1 + 2.0

        hpmapAdd2 = np.zeros_like(hpmap1) + hp.UNSEEN
        gd, = np.where(hpmap1 > hp.UNSEEN)
        hpmapAdd2[gd] = hpmap1[gd] + 2.0

        testing.assert_almost_equal(hpmapAdd2, added_map.generate_healpix_map())

        # Test adding a float constant to a map, in place

        sparse_map1 += 2.0

        testing.assert_almost_equal(hpmapAdd2, sparse_map1.generate_healpix_map())

    def test_product(self):
        """
        Test map products.
        """

        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        # Test adding two or three maps

        sparse_map1 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.float64)
        pixel1 = np.arange(4000, 20000)
        pixel1 = np.delete(pixel1, 15000)
        values1 = np.random.random(size=pixel1.size)
        sparse_map1.update_values_pix(pixel1, values1)
        hpmap1 = sparse_map1.generate_healpix_map()

        sparse_map2 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.float64)
        pixel2 = np.arange(15000, 25000)
        values2 = np.random.random(size=pixel2.size)
        sparse_map2.update_values_pix(pixel2, values2)
        hpmap2 = sparse_map2.generate_healpix_map()

        sparse_map3 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.float64)
        pixel3 = np.arange(16000, 25000)
        values3 = np.random.random(size=pixel3.size)
        sparse_map3.update_values_pix(pixel3, values3)
        hpmap3 = sparse_map3.generate_healpix_map()

        # _intersection product

        # product of 2
        product_map_intersection = healsparse.product_intersection([sparse_map1, sparse_map2])

        gd, = np.where((hpmap1 > hp.UNSEEN) & (hpmap2 > hp.UNSEEN))
        hpmap_product_intersection = np.zeros_like(hpmap1) + hp.UNSEEN
        hpmap_product_intersection[gd] = hpmap1[gd] * hpmap2[gd]

        testing.assert_almost_equal(hpmap_product_intersection,
                                    product_map_intersection.generate_healpix_map())

        # product of 3
        product_map_intersection = healsparse.product_intersection([sparse_map1, sparse_map2, sparse_map3])

        gd, = np.where((hpmap1 > hp.UNSEEN) & (hpmap2 > hp.UNSEEN) & (hpmap3 > hp.UNSEEN))
        hpmap_product_intersection = np.zeros_like(hpmap1) + hp.UNSEEN
        hpmap_product_intersection[gd] = hpmap1[gd] * hpmap2[gd] * hpmap3[gd]

        testing.assert_almost_equal(hpmap_product_intersection,
                                    product_map_intersection.generate_healpix_map())

        # Union product

        # product of 2
        product_map_union = healsparse.product_union([sparse_map1, sparse_map2])

        gd, = np.where((hpmap1 > hp.UNSEEN) | (hpmap2 > hp.UNSEEN))
        hpmap_product_union = np.zeros_like(hpmap1) + hp.UNSEEN

        hpmap_product_union[gd] = 1.0
        gd1, = np.where(hpmap1[gd] > hp.UNSEEN)
        hpmap_product_union[gd[gd1]] *= hpmap1[gd[gd1]]
        gd2, = np.where(hpmap2[gd] > hp.UNSEEN)
        hpmap_product_union[gd[gd2]] *= hpmap2[gd[gd2]]

        testing.assert_almost_equal(hpmap_product_union, product_map_union.generate_healpix_map())

        # product 3
        product_map_union = healsparse.product_union([sparse_map1, sparse_map2, sparse_map3])

        gd, = np.where((hpmap1 > hp.UNSEEN) | (hpmap2 > hp.UNSEEN) | (hpmap3 > hp.UNSEEN))
        hpmap_product_union = np.zeros_like(hpmap1) + hp.UNSEEN

        hpmap_product_union[gd] = 1.0
        gd1, = np.where(hpmap1[gd] > hp.UNSEEN)
        hpmap_product_union[gd[gd1]] *= hpmap1[gd[gd1]]
        gd2, = np.where(hpmap2[gd] > hp.UNSEEN)
        hpmap_product_union[gd[gd2]] *= hpmap2[gd[gd2]]
        gd3, = np.where(hpmap3[gd] > hp.UNSEEN)
        hpmap_product_union[gd[gd3]] *= hpmap3[gd[gd3]]

        testing.assert_almost_equal(hpmap_product_union, product_map_union.generate_healpix_map())

        # Test multiplying an int constant to a map

        mult_map = sparse_map1 * 2

        hpmap_product2 = np.zeros_like(hpmap1) + hp.UNSEEN
        gd, = np.where(hpmap1 > hp.UNSEEN)
        hpmap_product2[gd] = hpmap1[gd] * 2

        testing.assert_almost_equal(hpmap_product2, mult_map.generate_healpix_map())

        # Test multiplying a float constant to a map

        mult_map = sparse_map1 * 2.0

        hpmap_product2 = np.zeros_like(hpmap1) + hp.UNSEEN
        gd, = np.where(hpmap1 > hp.UNSEEN)
        hpmap_product2[gd] = hpmap1[gd] * 2.0

        testing.assert_almost_equal(hpmap_product2, mult_map.generate_healpix_map())

        # Test adding a float constant to a map, in place

        sparse_map1 *= 2.0

        testing.assert_almost_equal(hpmap_product2, sparse_map1.generate_healpix_map())

    def test_product_integer(self):
        """
        Test map products.
        """

        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64
        sentinel = 0
        maxval = 100

        # Test adding two or three maps

        sparse_map1 = healsparse.HealSparseMap.make_empty(
            nside_coverage,
            nside_map,
            np.int64,
            sentinel=sentinel,
        )
        pixel1 = np.arange(4000, 20000)
        pixel1 = np.delete(pixel1, 15000)
        values1 = random.randint(low=1, high=maxval, size=pixel1.size)
        sparse_map1.update_values_pix(pixel1, values1)

        hpmap1 = np.zeros(hp.nside2npix(nside_map), dtype=np.int64)
        vpix = sparse_map1.valid_pixels
        hpmap1[vpix] = sparse_map1.get_values_pix(vpix)

        sparse_map2 = healsparse.HealSparseMap.make_empty(
            nside_coverage,
            nside_map,
            np.int64,
            sentinel=sentinel,
        )
        pixel2 = np.arange(15000, 25000)
        values2 = random.randint(low=1, high=maxval, size=pixel2.size)
        sparse_map2.update_values_pix(pixel2, values2)

        hpmap2 = np.zeros(hp.nside2npix(nside_map), dtype=np.int64)
        vpix = sparse_map2.valid_pixels
        hpmap2[vpix] = sparse_map2.get_values_pix(vpix)

        sparse_map3 = healsparse.HealSparseMap.make_empty(
            nside_coverage,
            nside_map,
            np.int64,
            sentinel=sentinel,
        )
        pixel3 = np.arange(16000, 25000)
        values3 = random.randint(low=1, high=maxval, size=pixel3.size)
        sparse_map3.update_values_pix(pixel3, values3)

        hpmap3 = np.zeros(hp.nside2npix(nside_map), dtype=np.int64)
        vpix = sparse_map3.valid_pixels
        hpmap3[vpix] = sparse_map3.get_values_pix(vpix)

        # _intersection product

        # product of 2
        product_map_intersection = healsparse.product_intersection([sparse_map1, sparse_map2])

        gd, = np.where((hpmap1 > sentinel) & (hpmap2 > sentinel))
        hpmap_product_intersection = np.zeros_like(hpmap1)
        hpmap_product_intersection[gd] = hpmap1[gd] * hpmap2[gd]

        pmap = np.zeros(hp.nside2npix(nside_map), dtype=np.int64)
        vpix = product_map_intersection.valid_pixels
        pmap[vpix] = product_map_intersection.get_values_pix(vpix)

        testing.assert_equal(hpmap_product_intersection, pmap)

        # product of 3
        product_map_intersection = healsparse.product_intersection([sparse_map1, sparse_map2, sparse_map3])

        gd, = np.where((hpmap1 > sentinel) & (hpmap2 > sentinel) & (hpmap3 > sentinel))
        hpmap_product_intersection = np.zeros_like(hpmap1)
        hpmap_product_intersection[gd] = hpmap1[gd] * hpmap2[gd] * hpmap3[gd]

        pmap = np.zeros(hp.nside2npix(nside_map), dtype=np.int64)
        vpix = product_map_intersection.valid_pixels
        pmap[vpix] = product_map_intersection.get_values_pix(vpix)

        testing.assert_equal(hpmap_product_intersection, pmap)

        # _union product

        # product of 2
        product_map_union = healsparse.product_union([sparse_map1, sparse_map2])

        gd, = np.where((hpmap1 > sentinel) | (hpmap2 > sentinel))
        hpmap_product_union = np.zeros_like(hpmap1)

        hpmap_product_union[gd] = 1
        gd1, = np.where(hpmap1[gd] > sentinel)
        hpmap_product_union[gd[gd1]] *= hpmap1[gd[gd1]]
        gd2, = np.where(hpmap2[gd] > sentinel)
        hpmap_product_union[gd[gd2]] *= hpmap2[gd[gd2]]

        pmap = np.zeros(hp.nside2npix(nside_map), dtype=np.int64)
        vpix = product_map_union.valid_pixels
        pmap[vpix] = product_map_union.get_values_pix(vpix)

        testing.assert_equal(hpmap_product_union, pmap)

        # product 3
        product_map_union = healsparse.product_union([sparse_map1, sparse_map2, sparse_map3])

        gd, = np.where((hpmap1 > sentinel) | (hpmap2 > sentinel) | (hpmap3 > sentinel))
        hpmap_product_union = np.zeros_like(hpmap1)

        hpmap_product_union[gd] = 1
        gd1, = np.where(hpmap1[gd] > sentinel)
        hpmap_product_union[gd[gd1]] *= hpmap1[gd[gd1]]
        gd2, = np.where(hpmap2[gd] > sentinel)
        hpmap_product_union[gd[gd2]] *= hpmap2[gd[gd2]]
        gd3, = np.where(hpmap3[gd] > sentinel)
        hpmap_product_union[gd[gd3]] *= hpmap3[gd[gd3]]

        pmap = np.zeros(hp.nside2npix(nside_map), dtype=np.int64)
        vpix = product_map_union.valid_pixels
        pmap[vpix] = product_map_union.get_values_pix(vpix)

        testing.assert_equal(hpmap_product_union, pmap)

        # Test multiplying an int constant to a map

        mult_map = sparse_map1 * 2

        hpmap_product2 = np.zeros_like(hpmap1)
        gd, = np.where(hpmap1 > sentinel)
        hpmap_product2[gd] = hpmap1[gd] * 2

        pmap = np.zeros(hp.nside2npix(nside_map), dtype=np.int64)
        vpix = mult_map.valid_pixels
        pmap[vpix] = mult_map.get_values_pix(vpix)

        testing.assert_equal(hpmap_product2, pmap)

    def test_or(self):
        """
        Test map bitwise or.
        """

        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        for dtype in [np.int64, np.uint64]:
            sparse_map1 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype)
            pixel1 = np.arange(4000, 20000)
            pixel1 = np.delete(pixel1, 15000)
            # Get a random list of integers
            values1 = np.random.poisson(size=pixel1.size, lam=2).astype(dtype)
            sparse_map1.update_values_pix(pixel1, values1)
            hpmap1 = sparse_map1.generate_healpix_map()

            sparse_map2 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype)
            pixel2 = np.arange(15000, 25000)
            values2 = np.random.poisson(size=pixel2.size, lam=2).astype(dtype)
            sparse_map2.update_values_pix(pixel2, values2)
            hpmap2 = sparse_map2.generate_healpix_map()

            sparse_map3 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype)
            pixel3 = np.arange(16000, 25000)
            values3 = np.random.poisson(size=pixel3.size, lam=2).astype(dtype)
            sparse_map3.update_values_pix(pixel3, values3)
            hpmap3 = sparse_map3.generate_healpix_map()

            # _intersection or

            # or 2
            or_map_intersection = healsparse.or_intersection([sparse_map1, sparse_map2])

            gd, = np.where((hpmap1 > hp.UNSEEN) & (hpmap2 > hp.UNSEEN))
            hpmap_or_intersection = np.zeros_like(hpmap1) + hp.UNSEEN
            hpmap_or_intersection[gd] = hpmap1[gd].astype(dtype) | hpmap2[gd].astype(dtype)

            testing.assert_almost_equal(hpmap_or_intersection, or_map_intersection.generate_healpix_map())

            # or 3
            or_map_intersection = healsparse.or_intersection([sparse_map1, sparse_map2, sparse_map3])

            gd, = np.where((hpmap1 > hp.UNSEEN) & (hpmap2 > hp.UNSEEN) & (hpmap3 > hp.UNSEEN))
            hpmap_or_intersection = np.zeros_like(hpmap1) + hp.UNSEEN
            hpmap_or_intersection[gd] = (hpmap1[gd].astype(dtype) |
                                         hpmap2[gd].astype(dtype) |
                                         hpmap3[gd].astype(dtype))

            testing.assert_almost_equal(hpmap_or_intersection, or_map_intersection.generate_healpix_map())

            # Union or

            # or 2
            or_map_union = healsparse.or_union([sparse_map1, sparse_map2])

            gd, = np.where((hpmap1 > hp.UNSEEN) | (hpmap2 > hp.UNSEEN))
            hpmap_or_union = np.zeros_like(hpmap1) + hp.UNSEEN
            hpmap_or_union[gd] = (np.clip(hpmap1[gd], 0.0, None).astype(dtype) |
                                  np.clip(hpmap2[gd], 0.0, None).astype(dtype))

            testing.assert_almost_equal(hpmap_or_union, or_map_union.generate_healpix_map())

            # or 3
            or_map_union = healsparse.or_union([sparse_map1, sparse_map2, sparse_map3])

            gd, = np.where((hpmap1 > hp.UNSEEN) | (hpmap2 > hp.UNSEEN) | (hpmap3 > hp.UNSEEN))
            hpmap_or_union = np.zeros_like(hpmap1) + hp.UNSEEN
            hpmap_or_union[gd] = (np.clip(hpmap1[gd], 0.0, None).astype(dtype) |
                                  np.clip(hpmap2[gd], 0.0, None).astype(dtype) |
                                  np.clip(hpmap3[gd], 0.0, None).astype(dtype))

            testing.assert_almost_equal(hpmap_or_union, or_map_union.generate_healpix_map())

            # Test orring an int constant to a map

            or_map = sparse_map1 | 2

            hpmap_or2 = np.zeros_like(hpmap1) + hp.UNSEEN
            gd, = np.where(hpmap1 > hp.UNSEEN)
            hpmap_or2[gd] = hpmap1[gd].astype(dtype) | 2
            testing.assert_almost_equal(hpmap_or2, or_map.generate_healpix_map())

            # Test orring an int constant to a map, in place

            sparse_map1 |= 2

            testing.assert_almost_equal(hpmap_or2, sparse_map1.generate_healpix_map())

    def test_and(self):
        """
        Test map bitwise and.
        """

        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        for dtype in [np.int64, np.uint64]:
            sparse_map1 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype)
            pixel1 = np.arange(4000, 20000)
            pixel1 = np.delete(pixel1, 15000)
            # Get a random list of integers
            values1 = np.random.poisson(size=pixel1.size, lam=2).astype(dtype)
            sparse_map1.update_values_pix(pixel1, values1)
            hpmap1 = sparse_map1.generate_healpix_map()

            sparse_map2 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype)
            pixel2 = np.arange(15000, 25000)
            values2 = np.random.poisson(size=pixel2.size, lam=2).astype(dtype)
            sparse_map2.update_values_pix(pixel2, values2)
            hpmap2 = sparse_map2.generate_healpix_map()

            sparse_map3 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype)
            pixel3 = np.arange(16000, 25000)
            values3 = np.random.poisson(size=pixel3.size, lam=2).astype(dtype)
            sparse_map3.update_values_pix(pixel3, values3)
            hpmap3 = sparse_map3.generate_healpix_map()

            # _intersection and

            # and 2
            and_map_intersection = healsparse.and_intersection([sparse_map1, sparse_map2])

            gd, = np.where((hpmap1 > hp.UNSEEN) & (hpmap2 > hp.UNSEEN))
            hpmap_and_intersection = np.zeros_like(hpmap1) + hp.UNSEEN
            hpmap_and_intersection[gd] = hpmap1[gd].astype(dtype) & hpmap2[gd].astype(dtype)
            if dtype == np.uint64:
                # For uint, we cannot tell the difference between 0 and UNSEEN
                bd, = np.where(hpmap_and_intersection == 0)
                hpmap_and_intersection[bd] = hp.UNSEEN

            testing.assert_almost_equal(hpmap_and_intersection, and_map_intersection.generate_healpix_map())

            # and 3
            and_map_intersection = healsparse.and_intersection([sparse_map1, sparse_map2, sparse_map3])

            gd, = np.where((hpmap1 > hp.UNSEEN) & (hpmap2 > hp.UNSEEN) & (hpmap3 > hp.UNSEEN))
            hpmap_and_intersection = np.zeros_like(hpmap1) + hp.UNSEEN
            hpmap_and_intersection[gd] = (hpmap1[gd].astype(dtype) &
                                          hpmap2[gd].astype(dtype) &
                                          hpmap3[gd].astype(dtype))
            if dtype == np.uint64:
                # For uint, we cannot tell the difference between 0 and UNSEEN
                bd, = np.where(hpmap_and_intersection == 0)
                hpmap_and_intersection[bd] = hp.UNSEEN

            testing.assert_almost_equal(hpmap_and_intersection, and_map_intersection.generate_healpix_map())

            # Union and

            # and 2
            and_map_union = healsparse.and_union([sparse_map1, sparse_map2])

            gd, = np.where((hpmap1 > hp.UNSEEN) | (hpmap2 > hp.UNSEEN))
            hpmap_and_union = np.zeros_like(hpmap1) + hp.UNSEEN

            hpmap_and_union[gd] = -1.0
            gd1, = np.where(hpmap1[gd] > hp.UNSEEN)
            hpmap_and_union[gd[gd1]] = (hpmap_and_union[gd[gd1]].astype(dtype) &
                                        hpmap1[gd[gd1]].astype(dtype))
            gd2, = np.where(hpmap2[gd] > hp.UNSEEN)
            hpmap_and_union[gd[gd2]] = (hpmap_and_union[gd[gd2]].astype(dtype) &
                                        hpmap2[gd[gd2]].astype(dtype))
            if dtype == np.uint64:
                # For uint, we cannot tell the difference between 0 and UNSEEN
                bd, = np.where(hpmap_and_union == 0)
                hpmap_and_union[bd] = hp.UNSEEN

            testing.assert_almost_equal(hpmap_and_union, and_map_union.generate_healpix_map())

            # and 3
            and_map_union = healsparse.and_union([sparse_map1, sparse_map2, sparse_map3])

            gd, = np.where((hpmap1 > hp.UNSEEN) | (hpmap2 > hp.UNSEEN) | (hpmap3 > hp.UNSEEN))
            hpmap_and_union = np.zeros_like(hpmap1) + hp.UNSEEN

            hpmap_and_union[gd] = -1.0
            gd1, = np.where(hpmap1[gd] > hp.UNSEEN)
            hpmap_and_union[gd[gd1]] = (hpmap_and_union[gd[gd1]].astype(dtype) &
                                        hpmap1[gd[gd1]].astype(dtype))
            gd2, = np.where(hpmap2[gd] > hp.UNSEEN)
            hpmap_and_union[gd[gd2]] = (hpmap_and_union[gd[gd2]].astype(dtype) &
                                        hpmap2[gd[gd2]].astype(dtype))
            gd3, = np.where(hpmap3[gd] > hp.UNSEEN)
            hpmap_and_union[gd[gd3]] = (hpmap_and_union[gd[gd3]].astype(dtype) &
                                        hpmap3[gd[gd3]].astype(dtype))
            if dtype == np.uint64:
                # For uint, we cannot tell the difference between 0 and UNSEEN
                bd, = np.where(hpmap_and_union == 0)
                hpmap_and_union[bd] = hp.UNSEEN

            testing.assert_almost_equal(hpmap_and_union, and_map_union.generate_healpix_map())

            # Test anding an int constant to a map

            and_map = sparse_map1 & 2

            hpmap_and2 = np.zeros_like(hpmap1) + hp.UNSEEN
            gd, = np.where(hpmap1 > hp.UNSEEN)
            hpmap_and2[gd] = hpmap1[gd].astype(dtype) & 2
            if dtype == np.uint64:
                # For uint, we cannot tell the difference between 0 and UNSEEN
                bd, = np.where(hpmap_and2 == 0)
                hpmap_and2[bd] = hp.UNSEEN

            testing.assert_almost_equal(hpmap_and2, and_map.generate_healpix_map())

            # Test anding an int constant to a map, in place

            sparse_map1 &= 2

            testing.assert_almost_equal(hpmap_and2, sparse_map1.generate_healpix_map())

    def test_xor(self):
        """
        Test map bitwise xor.
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        for dtype in [np.int64, np.uint64]:
            sparse_map1 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.int64)
            pixel1 = np.arange(4000, 20000)
            pixel1 = np.delete(pixel1, 15000)
            # Get a random list of integers
            values1 = np.random.poisson(size=pixel1.size, lam=2)
            sparse_map1.update_values_pix(pixel1, values1)
            hpmap1 = sparse_map1.generate_healpix_map()

            sparse_map2 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.int64)
            pixel2 = np.arange(15000, 25000)
            values2 = np.random.poisson(size=pixel2.size, lam=2)
            sparse_map2.update_values_pix(pixel2, values2)
            hpmap2 = sparse_map2.generate_healpix_map()

            sparse_map3 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.int64)
            pixel3 = np.arange(16000, 25000)
            values3 = np.random.poisson(size=pixel3.size, lam=2)
            sparse_map3.update_values_pix(pixel3, values3)
            hpmap3 = sparse_map3.generate_healpix_map()

            # _intersection xor

            # xor 2
            xor_map_intersection = healsparse.xor_intersection([sparse_map1, sparse_map2])

            gd, = np.where((hpmap1 > hp.UNSEEN) & (hpmap2 > hp.UNSEEN))
            hpmap_xor_intersection = np.zeros_like(hpmap1) + hp.UNSEEN
            hpmap_xor_intersection[gd] = hpmap1[gd].astype(np.int64) ^ hpmap2[gd].astype(np.int64)

            testing.assert_almost_equal(hpmap_xor_intersection, xor_map_intersection.generate_healpix_map())

            # xor 3
            xor_map_intersection = healsparse.xor_intersection([sparse_map1, sparse_map2, sparse_map3])

            gd, = np.where((hpmap1 > hp.UNSEEN) & (hpmap2 > hp.UNSEEN) & (hpmap3 > hp.UNSEEN))
            hpmap_xor_intersection = np.zeros_like(hpmap1) + hp.UNSEEN
            hpmap_xor_intersection[gd] = (hpmap1[gd].astype(np.int64) ^
                                          hpmap2[gd].astype(np.int64) ^
                                          hpmap3[gd].astype(np.int64))

            testing.assert_almost_equal(hpmap_xor_intersection, xor_map_intersection.generate_healpix_map())

            # Union xor

            # xor 2
            xor_map_union = healsparse.xor_union([sparse_map1, sparse_map2])

            gd, = np.where((hpmap1 > hp.UNSEEN) | (hpmap2 > hp.UNSEEN))
            hpmap_xor_union = np.zeros_like(hpmap1) + hp.UNSEEN

            hpmap_xor_union[gd] = 0.0
            gd1, = np.where(hpmap1[gd] > hp.UNSEEN)
            hpmap_xor_union[gd[gd1]] = (hpmap_xor_union[gd[gd1]].astype(np.int64) ^
                                        hpmap1[gd[gd1]].astype(np.int64))
            gd2, = np.where(hpmap2[gd] > hp.UNSEEN)
            hpmap_xor_union[gd[gd2]] = (hpmap_xor_union[gd[gd2]].astype(np.int64) ^
                                        hpmap2[gd[gd2]].astype(np.int64))

            testing.assert_almost_equal(hpmap_xor_union, xor_map_union.generate_healpix_map())

            # xor 3
            xor_map_union = healsparse.xor_union([sparse_map1, sparse_map2, sparse_map3])

            gd, = np.where((hpmap1 > hp.UNSEEN) | (hpmap2 > hp.UNSEEN) | (hpmap3 > hp.UNSEEN))
            hpmap_xor_union = np.zeros_like(hpmap1) + hp.UNSEEN

            hpmap_xor_union[gd] = 0.0
            gd1, = np.where(hpmap1[gd] > hp.UNSEEN)
            hpmap_xor_union[gd[gd1]] = (hpmap_xor_union[gd[gd1]].astype(np.int64) ^
                                        hpmap1[gd[gd1]].astype(np.int64))
            gd2, = np.where(hpmap2[gd] > hp.UNSEEN)
            hpmap_xor_union[gd[gd2]] = (hpmap_xor_union[gd[gd2]].astype(np.int64) ^
                                        hpmap2[gd[gd2]].astype(np.int64))
            gd3, = np.where(hpmap3[gd] > hp.UNSEEN)
            hpmap_xor_union[gd[gd3]] = (hpmap_xor_union[gd[gd3]].astype(np.int64) ^
                                        hpmap3[gd[gd3]].astype(np.int64))

            testing.assert_almost_equal(hpmap_xor_union, xor_map_union.generate_healpix_map())

            # Test xorring an int constant to a map

            xor_map = sparse_map1 ^ 2

            hpmap_xor2 = np.zeros_like(hpmap1) + hp.UNSEEN
            gd, = np.where(hpmap1 > hp.UNSEEN)
            hpmap_xor2[gd] = hpmap1[gd].astype(np.int64) ^ 2
            testing.assert_almost_equal(hpmap_xor2, xor_map.generate_healpix_map())

            # Test xorring an int constant to a map, in place

            sparse_map1 ^= 2

            testing.assert_almost_equal(hpmap_xor2, sparse_map1.generate_healpix_map())

    def test_miscellaneous_operations(self):
        """
        Test miscellaneous constant operations.
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 64

        sparse_map1 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.float64)
        pixel1 = np.arange(4000, 20000)
        pixel1 = np.delete(pixel1, 15000)
        values1 = np.random.random(size=pixel1.size)
        sparse_map1.update_values_pix(pixel1, values1)
        hpmap1 = sparse_map1.generate_healpix_map()

        # subtraction
        test_map = sparse_map1 - 2.0

        hpmap_test = np.zeros_like(hpmap1) + hp.UNSEEN
        gd, = np.where(hpmap1 > hp.UNSEEN)
        hpmap_test[gd] = hpmap1[gd] - 2.0

        testing.assert_almost_equal(hpmap_test, test_map.generate_healpix_map())

        test_map = sparse_map1.copy()
        test_map -= 2.0
        testing.assert_almost_equal(hpmap_test, test_map.generate_healpix_map())

        # division
        test_map = sparse_map1 / 2.0

        hpmap_test = np.zeros_like(hpmap1) + hp.UNSEEN
        gd, = np.where(hpmap1 > hp.UNSEEN)
        hpmap_test[gd] = hpmap1[gd] / 2.0

        testing.assert_almost_equal(hpmap_test, test_map.generate_healpix_map())

        test_map = sparse_map1.copy()
        test_map /= 2.0
        testing.assert_almost_equal(hpmap_test, test_map.generate_healpix_map())

        # power
        test_map = sparse_map1 ** 2.0

        hpmap_test = np.zeros_like(hpmap1) + hp.UNSEEN
        gd, = np.where(hpmap1 > hp.UNSEEN)
        hpmap_test[gd] = hpmap1[gd] ** 2.0

        testing.assert_almost_equal(hpmap_test, test_map.generate_healpix_map())

        test_map = sparse_map1.copy()
        test_map **= 2.0
        testing.assert_almost_equal(hpmap_test, test_map.generate_healpix_map())


if __name__ == '__main__':
    unittest.main()
