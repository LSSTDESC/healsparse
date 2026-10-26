import unittest
import numpy as np

import healsparse


class CropTestCase(unittest.TestCase):
    def test_crop_circle(self):
        m = healsparse.HealSparseMap.make_empty(32, 4096, bool, bit_packed=True)
        box0 = healsparse.Box(ra1=95.0, ra2=105.0, dec1=45.0, dec2=55.0, value=True)
        m |= box0

        valid_orig = m.valid_pixels
        ncov_orig = np.sum(m.coverage_mask)

        circle = healsparse.Circle(ra=100.0, dec=50.0, radius=2.0, value=True)
        # Try in_place=True
        m.crop(circle, in_place=True)

        valid_new = m.valid_pixels

        # We cropped pixels ...
        self.assertLess(len(valid_new), len(valid_orig))
        # ... and shrank the coverage map.
        self.assertLess(np.sum(m.coverage_mask), ncov_orig)
        np.testing.assert_array_equal(valid_new, circle.get_pixels(nside=m.nside_sparse))

    def test_crop_box(self):
        m = healsparse.HealSparseMap.make_empty(32, 4096, bool, bit_packed=True)
        box0 = healsparse.Box(ra1=95.0, ra2=105.0, dec1=45.0, dec2=55.0, value=True)
        m |= box0

        box = healsparse.Box(ra1=98.0, ra2=104.0, dec1=45.0, dec2=47.0, value=True)
        # Try in_place=False
        m2 = m.crop(box, in_place=False)

        valid_orig = m.valid_pixels
        valid_new = m2.valid_pixels

        # We cropped pixels ...
        self.assertLess(len(valid_new), len(valid_orig))
        # ... and shrank the coverage map.
        self.assertLess(np.sum(m2.coverage_mask), np.sum(m.coverage_mask))
        np.testing.assert_array_equal(valid_new, box.get_pixels(nside=m.nside_sparse))

    def test_crop_polygon(self):
        m = healsparse.HealSparseMap.make_empty(32, 4096, bool, bit_packed=True)
        box0 = healsparse.Box(ra1=95.0, ra2=105.0, dec1=45.0, dec2=55.0, value=True)
        m |= box0

        valid_orig = m.valid_pixels
        ncov_orig = np.sum(m.coverage_mask)

        poly = healsparse.Polygon(ra=[96.0, 97.0, 98.0, 96.0], dec=[46.0, 46.0, 47.0, 47.0], value=True)
        # Try in_place=True, defragment=False
        m.crop(poly, in_place=True, defragment=False)

        valid_new = m.valid_pixels

        # We cropped pixels ...
        self.assertLess(len(valid_new), len(valid_orig))
        # ... but did not shrink the coverage map.
        self.assertEqual(np.sum(m.coverage_mask), ncov_orig)
        np.testing.assert_array_equal(valid_new, poly.get_pixels(nside=m.nside_sparse))

    def test_crop_ellipse(self):
        m = healsparse.HealSparseMap.make_empty(32, 4096, bool, bit_packed=True)
        box0 = healsparse.Box(ra1=95.0, ra2=105.0, dec1=45.0, dec2=55.0, value=True)
        m |= box0

        ellipse = healsparse.Ellipse(
            ra=100.0,
            dec=50.0,
            semi_major=2.0,
            semi_minor=1.0,
            alpha=45.0,
            value=True,
        )
        # Try in_place=False, defragment=False
        m2 = m.crop(ellipse, in_place=False, defragment=False)

        valid_orig = m.valid_pixels
        valid_new = m2.valid_pixels

        # We cropped pixels ...
        self.assertLess(len(valid_new), len(valid_orig))
        # ... but did not shrink the coverage map.
        self.assertEqual(np.sum(m2.coverage_mask), np.sum(m.coverage_mask))
        np.testing.assert_array_equal(valid_new, ellipse.get_pixels(nside=m.nside_sparse))

    def test_crop_union(self):
        m = healsparse.HealSparseMap.make_empty(32, 4096, bool, bit_packed=True)
        box0 = healsparse.Box(ra1=95.0, ra2=105.0, dec1=45.0, dec2=55.0, value=True)
        m |= box0

        valid_orig = m.valid_pixels
        ncov_orig = np.sum(m.coverage_mask)

        circle1 = healsparse.Circle(ra=100.0, dec=50.0, radius=2.0, value=True)
        circle2 = healsparse.Circle(ra=102.0, dec=48.0, radius=1.0, value=True)
        union = healsparse.GeomUnion(geom_list=[circle1, circle2])
        m.crop(union, in_place=True)

        valid_new = m.valid_pixels

        # We cropped pixels ...
        self.assertLess(len(valid_new), len(valid_orig))
        # ... and shrank the coverage map.
        self.assertLess(np.sum(m.coverage_mask), ncov_orig)
        np.testing.assert_array_equal(valid_new, union.get_pixels(nside=m.nside_sparse))
