import unittest
import numpy as np
from numpy import random

import healsparse


class ClearTestCase(unittest.TestCase):

    def test_clear_pixels_single(self):
        """
        Test clear_pixels single value.
        """
        nside_coverage = 32
        nside_map = 4096

        m = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.bool_)

        pixels = np.arange(1000, 10000)
        m[pixels] = True

        size = m._sparse_map.size

        # Delete in range.
        m.clear_pixels(1100)
        pixels = np.delete(pixels, pixels == 1100)

        np.testing.assert_array_equal(m.valid_pixels, pixels)
        self.assertEqual(size, m._sparse_map.size)

        # Delete out of range.
        m.clear_pixels(11000)

        np.testing.assert_array_equal(m.valid_pixels, pixels)
        self.assertEqual(size, m._sparse_map.size)

    def test_clear_pixels_recarray_single(self):
        """
        Test clear_pixels single value from recarray.
        """
        nside_coverage = 32
        nside_map = 4096

        dtype = [('col1', 'f8'), ('col2', 'f8'), ('col3', 'i4')]
        m = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype, primary='col1')

        pixels = np.arange(1000, 10000)
        values = np.zeros_like(pixels, dtype=dtype)
        values['col1'] = random.random(size=pixels.size)
        values['col2'] = random.random(size=pixels.size)
        values['col3'] = np.ones(pixels.size, dtype=np.int32)

        m[pixels] = values

        size = m._sparse_map.size

        # Delete in range.
        m.clear_pixels(1100)
        pixels = np.delete(pixels, pixels == 1100)

        np.testing.assert_array_equal(m.valid_pixels, pixels)
        self.assertEqual(size, m._sparse_map.size)

        # Delete out of range.
        m.clear_pixels(11000)

        np.testing.assert_array_equal(m.valid_pixels, pixels)
        self.assertEqual(size, m._sparse_map.size)

    def test_clear_pixels_slice(self):
        """
        Test clear_pixels slice.
        """
        nside_coverage = 32
        nside_map = 4096

        m = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.bool_)

        pixels = np.arange(1000, 10000)
        m[pixels] = True

        size = m._sparse_map.size

        # Delete in range.
        m.clear_pixels(slice(1100, 1150))
        pixels = np.delete(pixels, (pixels >= 1100) & (pixels < 1150))

        np.testing.assert_array_equal(m.valid_pixels, pixels)
        self.assertEqual(size, m._sparse_map.size)

        # Delete out of range.
        m.clear_pixels(slice(11000, 12000))

        np.testing.assert_array_equal(m.valid_pixels, pixels)
        self.assertEqual(size, m._sparse_map.size)

    def test_clear_pixels_array(self):
        """
        Test clear_pixels array.
        """
        nside_coverage = 32
        nside_map = 4096

        m = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.bool_)

        pixels = np.arange(1000, 10000)
        m[pixels] = True

        size = m._sparse_map.size

        # Delete in range.
        m.clear_pixels(np.arange(1100, 1150))
        pixels = np.delete(pixels, (pixels >= 1100) & (pixels < 1150))

        np.testing.assert_array_equal(m.valid_pixels, pixels)
        self.assertEqual(size, m._sparse_map.size)

        # Delete out of range.
        m.clear_pixels(np.arange(11000, 12000))

        np.testing.assert_array_equal(m.valid_pixels, pixels)
        self.assertEqual(size, m._sparse_map.size)

    def test_clear_pixels_list(self):
        """
        Test clear_pixels list.
        """
        nside_coverage = 32
        nside_map = 4096

        m = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.bool_)

        pixels = np.arange(1000, 10000)
        m[pixels] = True

        size = m._sparse_map.size

        # Delete in range.
        m.clear_pixels(list(np.arange(1100, 1150)))
        pixels = np.delete(pixels, (pixels >= 1100) & (pixels < 1150))

        np.testing.assert_array_equal(m.valid_pixels, pixels)
        self.assertEqual(size, m._sparse_map.size)

        # Delete out of range.
        m.clear_pixels(list(np.arange(11000, 12000)))

        np.testing.assert_array_equal(m.valid_pixels, pixels)
        self.assertEqual(size, m._sparse_map.size)

    def test_clear_pixels_other(self):
        """
        Test clear_pixels other.
        """
        nside_coverage = 32
        nside_map = 4096

        m = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.bool_)

        pixels = np.arange(1000, 10000)
        m[pixels] = True

        with self.assertRaises(IndexError):
            m.clear_pixels(5.0)

    def test_clear_pixels_geom(self):
        """
        Test clear_pixels geom.
        """
        nside_coverage = 32
        nside_map = 4096

        m = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.bool_)

        circle1 = healsparse.Circle(ra=100.0, dec=20.0, radius=0.5, value=True)
        circle2 = healsparse.Circle(ra=100.0, dec=20.0, radius=0.1, value=True)

        m[circle1] = True

        size = m._sparse_map.size

        m.clear_pixels(circle2)

        pix1 = circle1.get_pixels(nside=m.nside_sparse)
        pix2 = circle2.get_pixels(nside=m.nside_sparse)

        a = np.searchsorted(pix1, pix2)
        pixels = np.delete(pix1, a)

        np.testing.assert_array_equal(m.valid_pixels, pixels)
        self.assertEqual(size, m._sparse_map.size)

        circle3 = healsparse.Circle(ra=180.0, dec=-20.0, radius=0.1, value=True)
        m.clear_pixels(circle3)

        np.testing.assert_array_equal(m.valid_pixels, pixels)
        self.assertEqual(size, m._sparse_map.size)
