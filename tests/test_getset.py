import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
from numpy import random

import healsparse


class GetSetTestCase(unittest.TestCase):
    def test_getitem_single(self):
        """
        Test __getitem__ single value
        """
        np.random.seed(12345)

        nside_coverage = 32
        nside_map = 128

        full_map = np.zeros(hp.nside2npix(nside_map)) + hp.UNSEEN
        full_map[0: 5000] = np.random.random(size=5000)

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage)

        # Grab a single item, in range
        testing.assert_almost_equal(sparse_map[100], full_map[100])

        # Grab a single item out of range
        testing.assert_almost_equal(sparse_map[6000], full_map[6000])

    def test_getitem_recarray_single(self):
        """
        Test __getitem__ from a recarray
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 128

        dtype = [('col1', 'f8'), ('col2', 'f8')]
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype, primary='col1')
        pixel = np.arange(5000)
        values = np.zeros_like(pixel, dtype=dtype)
        values['col1'] = np.random.random(size=pixel.size)
        values['col2'] = np.random.random(size=pixel.size)
        sparse_map.update_values_pix(pixel, values)

        # Test name access
        test = sparse_map['col1']
        testing.assert_array_almost_equal(test.get_values_pix(test.valid_pixels),
                                          values['col1'])

        # Test index access
        test_item = sparse_map[1000]
        testing.assert_almost_equal(test_item['col1'], values['col1'][1000])
        testing.assert_almost_equal(test_item['col2'], values['col2'][1000])

        test_item = sparse_map[10000]
        testing.assert_almost_equal(test_item['col1'], hp.UNSEEN)
        testing.assert_almost_equal(test_item['col2'], hp.UNSEEN)

    def test_getitem_slice(self):
        """
        Test __getitem__ using slices
        """
        np.random.seed(12345)

        nside_coverage = 32
        nside_map = 128

        full_map = np.zeros(hp.nside2npix(nside_map)) + hp.UNSEEN
        full_map[0: 5000] = np.random.random(size=5000)

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage)

        # Test in-range, overlap, out-of-range
        testing.assert_array_almost_equal(sparse_map[100: 500], full_map[100: 500])
        testing.assert_array_almost_equal(sparse_map[4500: 5500], full_map[4500: 5500])
        testing.assert_array_almost_equal(sparse_map[5500: 5600], full_map[5500: 5600])

        # Test stepped
        testing.assert_array_almost_equal(sparse_map[100: 500: 2], full_map[100: 500: 2])
        testing.assert_array_almost_equal(sparse_map[4500: 5500: 2], full_map[4500: 5500: 2])
        testing.assert_array_almost_equal(sparse_map[5500: 5600: 2], full_map[5500: 5600: 2])

        # Test all
        testing.assert_array_almost_equal(sparse_map[:], full_map[:])

    def test_getitem_array(self):
        """
        Test __getitem__ using an array
        """
        np.random.seed(12345)

        nside_coverage = 32
        nside_map = 128

        full_map = np.zeros(hp.nside2npix(nside_map)) + hp.UNSEEN
        full_map[0: 5000] = np.random.random(size=5000)

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage)

        indices = np.array([1, 2, 100, 500, 10000])
        testing.assert_array_almost_equal(sparse_map[indices], full_map[indices])
        testing.assert_almost_equal(sparse_map[indices[0]], full_map[indices[0]])

        indices = np.array([1., 2, 100, 500, 10000])
        self.assertRaises(IndexError, sparse_map.__getitem__, indices)
        self.assertRaises(IndexError, sparse_map.__getitem__, indices[0])

    def test_getitem_list(self):
        """
        Test __getitem__ using list/tuple
        """
        np.random.seed(12345)

        nside_coverage = 32
        nside_map = 128

        full_map = np.zeros(hp.nside2npix(nside_map)) + hp.UNSEEN
        full_map[0: 5000] = np.random.random(size=5000)

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage)

        indices = [1, 2, 100, 500, 10000]
        testing.assert_array_almost_equal(sparse_map[indices], full_map[indices])

        indices = [1.0, 2, 100, 500, 10000]
        self.assertRaises(IndexError, sparse_map.__getitem__, indices)

    def test_getitem_other(self):
        """
        Test __getitem__ using something else
        """
        np.random.seed(12345)

        nside_coverage = 32
        nside_map = 128

        full_map = np.zeros(hp.nside2npix(nside_map)) + hp.UNSEEN
        full_map[0: 5000] = np.random.random(size=5000)

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage)

        indices = (1, 2, 3, 4)
        self.assertRaises(IndexError, sparse_map.__getitem__, indices)

        indices = 5.0
        self.assertRaises(IndexError, sparse_map.__getitem__, indices)

    def test_setitem_single(self):
        """
        Test __setitem__ single value
        """
        np.random.seed(12345)

        nside_coverage = 32
        nside_map = 128

        full_map = np.zeros(hp.nside2npix(nside_map)) + hp.UNSEEN
        full_map[0: 5000] = np.random.random(size=5000)

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage)

        sparse_map[1000] = 1.0
        full_map[1000] = 1.0
        testing.assert_array_almost_equal(sparse_map.generate_healpix_map(),
                                          full_map)

        sparse_map[10000] = 1.0
        full_map[10000] = 1.0
        testing.assert_array_almost_equal(sparse_map.generate_healpix_map(),
                                          full_map)

    def test_setitem_recarray_single(self):
        """
        Test __setitem__ from recarray
        """
        random.seed(seed=12345)

        nside_coverage = 32
        nside_map = 128

        dtype = [('col1', 'f8'), ('col2', 'f8'), ('col3', 'i4')]
        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, dtype, primary='col1')
        pixel = np.arange(5000)
        values = np.zeros_like(pixel, dtype=dtype)
        values['col1'] = np.random.random(size=pixel.size)
        values['col2'] = np.random.random(size=pixel.size)
        values['col3'] = np.ones(pixel.size, dtype=np.int32)
        sparse_map.update_values_pix(pixel, values)

        value = np.zeros(1, dtype=dtype)
        value['col1'] = 1.0
        value['col2'] = 1.0
        value['col3'] = 10
        sparse_map[1000] = value
        testing.assert_almost_equal(sparse_map['col1'][1000], 1.0)
        testing.assert_almost_equal(sparse_map['col2'][1000], 1.0)
        self.assertEqual(sparse_map['col3'][1000], 10)
        testing.assert_almost_equal(sparse_map[1000]['col1'], 1.0)
        testing.assert_almost_equal(sparse_map[1000]['col2'], 1.0)
        self.assertEqual(sparse_map[1000]['col3'], 10)

        self.assertRaises(IndexError, sparse_map.__setitem__, 'col1', 1.0)

        # Try setting individual columns... test both ways of calling
        # although only the one works for setting
        sparse_map['col1'][100] = 100.0
        testing.assert_almost_equal(sparse_map['col1'][100], 100.0)
        testing.assert_almost_equal(sparse_map[100]['col1'], 100.0)

        sparse_map['col2'][100] = 100.0
        testing.assert_almost_equal(sparse_map['col2'][100], 100.0)
        testing.assert_almost_equal(sparse_map[100]['col2'], 100.0)

        sparse_map['col3'][100] = 100
        self.assertEqual(sparse_map['col3'][100], 100)
        self.assertEqual(sparse_map[100]['col3'], 100)

        sparse_map['col1'][100: 200] = np.zeros(100)
        testing.assert_array_almost_equal(sparse_map['col1'][100: 200], 0.0)
        testing.assert_array_almost_equal(sparse_map[100: 200]['col1'], 0.0)

        sparse_map['col2'][100: 200] = np.zeros(100)
        testing.assert_array_almost_equal(sparse_map['col2'][100: 200], 0.0)
        testing.assert_array_almost_equal(sparse_map[100: 200]['col2'], 0.0)

        sparse_map['col3'][100: 200] = np.zeros(100, dtype=np.int32)
        testing.assert_array_equal(sparse_map['col3'][100: 200], 0)
        testing.assert_array_equal(sparse_map[100: 200]['col3'], 0)

        # Finally, assert that we cannot set new pixels
        self.assertRaises(RuntimeError, sparse_map['col1'].__setitem__,
                          10000, 10.0)

    def test_setitem_slice(self):
        """
        Test __setitem__ slice
        """
        np.random.seed(12345)

        nside_coverage = 32
        nside_map = 128

        full_map = np.zeros(hp.nside2npix(nside_map)) + hp.UNSEEN
        full_map[0: 5000] = np.random.random(size=5000)

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage)

        # This needs to be accessed with an array of length 1 or same length.
        sparse_map[100: 500] = np.array([1.0])
        full_map[100: 500] = np.array([1.0])
        testing.assert_array_almost_equal(sparse_map.generate_healpix_map(),
                                          full_map)

        sparse_map[1000: 1500] = np.ones(500)
        full_map[1000: 1500] = np.ones(500)
        testing.assert_array_almost_equal(sparse_map.generate_healpix_map(),
                                          full_map)

        sparse_map[10000: 11000: 2] = np.ones(500)
        full_map[10000: 11000: 2] = np.ones(500)
        testing.assert_array_almost_equal(sparse_map.generate_healpix_map(),
                                          full_map)

        # Test all
        sparse_map[:] = np.array([1.0])
        full_map[:] = np.array([1.0])
        testing.assert_array_almost_equal(sparse_map.generate_healpix_map(),
                                          full_map)

    def test_setitem_array(self):
        """
        Test __setitem__ array
        """
        np.random.seed(12345)

        nside_coverage = 32
        nside_map = 128

        full_map = np.zeros(hp.nside2npix(nside_map)) + hp.UNSEEN
        full_map[0: 5000] = np.random.random(size=5000)

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage)

        indices = np.array([1, 2, 100, 500, 10000])
        sparse_map[indices] = np.array([1.0])
        full_map[indices] = np.array([1.0])
        testing.assert_array_almost_equal(sparse_map.generate_healpix_map(),
                                          full_map)

        # Simple in-place operation
        sparse_map[indices] += 1.0
        full_map[indices] += 1.0
        testing.assert_array_almost_equal(sparse_map.generate_healpix_map(),
                                          full_map)

        indices = np.array([1, 2, 100, 500, 10000]) + 100
        sparse_map[indices] = np.ones(len(indices))
        full_map[indices] = np.ones(len(indices))
        testing.assert_array_almost_equal(sparse_map.generate_healpix_map(),
                                          full_map)

        indices = np.array([1., 2, 100, 500, 10000])
        self.assertRaises(IndexError, sparse_map.__setitem__, indices, 1.0)

    def test_setitem_list(self):
        """
        Test __setitem__ list
        """
        np.random.seed(12345)

        nside_coverage = 32
        nside_map = 128

        full_map = np.zeros(hp.nside2npix(nside_map)) + hp.UNSEEN
        full_map[0: 5000] = np.random.random(size=5000)

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage)

        indices = [1, 2, 100, 500, 10000]
        sparse_map[indices] = np.array([1.0])
        full_map[indices] = np.array([1.0])
        testing.assert_array_almost_equal(sparse_map.generate_healpix_map(),
                                          full_map)

        indices = [101, 102, 200, 600, 10100]
        sparse_map[indices] = np.ones(len(indices))
        full_map[indices] = np.ones(len(indices))
        testing.assert_array_almost_equal(sparse_map.generate_healpix_map(),
                                          full_map)

        indices = [1., 2, 100, 500, 10000]
        self.assertRaises(IndexError, sparse_map.__setitem__, indices, 1.0)

    def test_setitem_other(self):
        """
        Test __setitem__ using something else
        """
        np.random.seed(12345)

        nside_coverage = 32
        nside_map = 128

        full_map = np.zeros(hp.nside2npix(nside_map)) + hp.UNSEEN
        full_map[0: 5000] = np.random.random(size=5000)

        sparse_map = healsparse.HealSparseMap(healpix_map=full_map, nside_coverage=nside_coverage)

        indices = (1, 2, 3, 4)
        self.assertRaises(IndexError, sparse_map.__setitem__, indices, 1.0)

        indices = 5.0
        self.assertRaises(IndexError, sparse_map.__setitem__, indices, 1.0)


if __name__ == '__main__':
    unittest.main()
