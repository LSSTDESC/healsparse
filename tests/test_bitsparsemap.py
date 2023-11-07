import unittest
import numpy.testing as testing
import numpy as np
import tempfile
import os
import shutil
import pytest

import healsparse
from healsparse import HealSparseMap, BitSparseMap


class BitSparseMapTestCase(unittest.TestCase):
    """Tests for BitSparseMap."""
    def test_create(self):
        m = BitSparseMap(size=0)
        self.assertEqual(m.size, 0)

        m = BitSparseMap(size=2**10)
        self.assertEqual(m.size, 2**10)
        self.assertEqual(len(m.data_array), 2**10 // 8)
        testing.assert_array_equal(m.data_array, 0)

        m = BitSparseMap(size=2**10, fill_value=True)
        self.assertEqual(m.size, 2**10)
        self.assertEqual(len(m.data_array), 2**10 // 8)
        testing.assert_array_equal(m.data_array, 255)

        m2 = BitSparseMap(data_buffer=m.data_array)
        self.assertEqual(m2.size, 2**10)
        self.assertEqual(len(m2.data_array), 2**10 // 8)
        testing.assert_array_equal(m2.data_array, 255)

        with self.assertRaises(ValueError):
            m = BitSparseMap(size=6)

    def test_resize(self):
        m = BitSparseMap(size=0)
        m.resize(2**10)
        self.assertEqual(m.size, 2**10)
        self.assertEqual(len(m.data_array), 2**10 // 8)
        testing.assert_array_equal(m.data_array, 0)

        with self.assertRaises(ValueError):
            m.resize(67)

    def test_copy(self):
        m = BitSparseMap(size=2**5)
        m[0] = True

        m2 = m.copy()
        testing.assert_array_equal(m2.data_array, m.data_array)

        m2[0] = False
        self.assertFalse(np.all(m2.data_array == m.data_array))

    def test_view(self):
        m = BitSparseMap(size=2**10)
        m[0] = True

        m2 = BitSparseMap(data_buffer=m.data_array)
        testing.assert_array_equal(m2.data_array, m.data_array)

        # This should change both.
        m2[0] = False
        testing.assert_array_equal(m2.data_array, m.data_array)

        # And another way to get a view.
        m3 = m[:]
        testing.assert_array_equal(m3.data_array, m.data_array)

        # This should change both.
        m2[0] = True
        testing.assert_array_equal(m3.data_array, m.data_array)

    def test_repr(self):
        m = BitSparseMap(size=2**10)

        self.assertEqual(repr(m), f"BitSparseMap(size={2**10})")
        self.assertEqual(str(m), f"BitSparseMap(size={2**10})")

    def test_setitem_single(self):
        # Test setting a single location.
        m = BitSparseMap(size=2**10)

        m[100] = True
        self.assertEqual(m[100], True)

    def test_setitem_slice_optimized_single(self):
        m = BitSparseMap(size=2**10)

        # Set True
        m[0: 64] = True
        testing.assert_array_equal(m[0: 64], True)
        testing.assert_array_equal(m[64:], False)

        # Set False
        m[16: 32] = False
        testing.assert_array_equal(m[0: 16], True)
        testing.assert_array_equal(m[16: 32], False)
        testing.assert_array_equal(m[64:], False)

    def test_setitiem_slice_unoptimized_single(self):
        m = BitSparseMap(size=2**10)

        # Set True
        m[0: 63] = True
        arr = np.array(m)
        testing.assert_array_equal(arr[0: 63], True)
        testing.assert_array_equal(arr[63:], False)

        # Set False
        m[17: 31] = False
        arr = np.array(m)
        testing.assert_array_equal(arr[0: 17], True)
        testing.assert_array_equal(arr[17: 31], False)
        testing.assert_array_equal(arr[63:], False)

    def test_setitem_slice_optimized_array(self):
        m = BitSparseMap(size=2**10)

        values = np.zeros(64, dtype=np.bool_)
        values[10: 20] = True

        m[0: 64] = values
        testing.assert_array_equal(m[0: 64], values)

    def test_setitiem_slice_unoptimized_array(self):
        m = BitSparseMap(size=2**10)

        values = np.zeros(62, dtype=np.bool_)
        values[10: 20] = True

        m[0: 62] = values
        arr = np.array(m)
        testing.assert_array_equal(arr[0: 62], values)

    def test_setgetitiem_indices(self):
        m = BitSparseMap(size=2**10)

        inds = np.array([1, 5, 10, 20])
        m[inds] = True

        testing.assert_array_equal(m[inds], True)

        values = np.array([True, False, True, True])
        m[inds] = values

        testing.assert_array_equal(m[inds], values)

    def test_setgetitem_list(self):
        m = BitSparseMap(size=2**10)

        inds = [1, 5, 10, 20]
        m[inds] = True

        testing.assert_array_equal(m[inds], True)

        values = np.array([True, False, True, True])
        m[tuple(inds)] = values

        testing.assert_array_equal(m[tuple(inds)], values)

    def test_getitem_single(self):
        m = BitSparseMap(size=2**10)

        m[10] = True

        self.assertEqual(m[10], True)

    def test_getitem_slice(self):
        m = BitSparseMap(size=2**10)

        m[16: 64] = True

        # This should return a view of the original map, and can be changed.
        m2 = m[16: 64]
        m2[0] = False
        self.assertEqual(m[16], False)

        with self.assertRaises(ValueError):
            _ = m[5: 32]
        with self.assertRaises(ValueError):
            _ = m[8: 31]
        with self.assertRaises(ValueError):
            _ = m[8: 32: 4]

    def test_sum(self):
        m = BitSparseMap(size=2**10)

        m[0] = True
        m[10] = True
        m[100] = True
        m[1000] = True

        self.assertEqual(m.sum(), 4)

        sum2 = m.sum(shape=(128, 8), axis=1)
        self.assertEqual(len(sum2), 128)

        # This is a very simple test of the summation output.
        self.assertEqual(sum2[0 // 8], 1)
        self.assertEqual(sum2[10 // 8], 1)
        self.assertEqual(sum2[100 // 8], 1)
        self.assertEqual(sum2[1000 // 8], 1)


class HealSparseBitMaskTestCase(unittest.TestCase):
    """Tests for HealSparseMap with bit_mask."""
    def test_make_bit_mask_map(self):
        nside_coverage = 32

        sparse_map = HealSparseMap.make_empty(nside_coverage, 2**15, np.bool_, bit_mask=True)
        self.assertTrue(sparse_map.is_bit_mask_map)
        self.assertEqual(sparse_map.sentinel, False)
        self.assertTrue(sparse_map.dtype, np.bool_)

        pixel = np.arange(4000, 200000)
        sparse_map[pixel] = True

        testing.assert_array_equal(sparse_map[pixel], True)
        testing.assert_array_equal(sparse_map.valid_pixels, pixel)
        testing.assert_equal(sparse_map.n_valid, len(pixel))

        with self.assertRaises(NotImplementedError):
            sparse_map = HealSparseMap.make_empty(
                nside_coverage,
                2**15,
                np.bool_,
                bit_mask=True,
                sentinel=True,
            )

        with self.assertRaises(ValueError):
            sparse_map = HealSparseMap.make_empty(nside_coverage, 64, np.bool_, bit_mask=True)

    def test_bit_mask_map_fits_io(self):
        nside_coverage = 32
        nside_map = 1024

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        sparse_map = HealSparseMap.make_empty(nside_coverage, nside_map, np.bool_, bit_mask=True)

        sparse_map[10_000_000: 11_000_000] = True

        fname = os.path.join(self.test_dir, f"healsparse_bitmask_{nside_map}.hsp")
        sparse_map.write(fname, clobber=True)
        sparse_map_in = HealSparseMap.read(fname)

        self.assertTrue(sparse_map_in.is_bit_mask_map)
        self.assertEqual(sparse_map_in.dtype, np.bool_)
        self.assertEqual(sparse_map_in.sentinel, False)

        testing.assert_array_equal(sparse_map_in.valid_pixels, sparse_map.valid_pixels)

        # Test reading back partial coverage.
        cov = healsparse.HealSparseCoverage.read(fname)
        covered_pixels, = np.where(cov.coverage_mask)

        sparse_map_in_partial = healsparse.HealSparseMap.read(
            fname,
            pixels=[covered_pixels[1], covered_pixels[10]],
        )
        self.assertTrue(sparse_map_in_partial.is_bit_mask_map)
        self.assertEqual(sparse_map_in_partial.dtype, np.bool_)
        self.assertEqual(sparse_map_in_partial.sentinel, False)

        cov_pixels = sparse_map._cov_map.cov_pixels(sparse_map.valid_pixels)
        pixel_sub = sparse_map.valid_pixels[(cov_pixels == covered_pixels[1]) |
                                            (cov_pixels == covered_pixels[10])]
        testing.assert_array_equal(sparse_map_in_partial.valid_pixels, pixel_sub)

    def test_bit_mask_map_fits_io_giant(self):
        # I don't know how to test this.
        nside_coverage = 32
        nside_map = 2**17

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')
        sparse_map = HealSparseMap.make_empty(
            nside_coverage,
            nside_map,
            np.bool_,
            bit_mask=True,
            cov_pixels=np.arange(1500),
        )

        sparse_map[1_000_000] = True
        sparse_map[100_000_000] = True

        fname = os.path.join(self.test_dir, f"healsparse_bitmask_{nside_map}_giant.hsp")
        sparse_map.write(fname, clobber=True)
        sparse_map_in = HealSparseMap.read(fname)

        # Confirm it's the correct type.
        self.assertTrue(sparse_map_in.is_bit_mask_map)
        self.assertEqual(sparse_map_in.dtype, np.bool_)
        self.assertEqual(sparse_map_in.sentinel, False)
        # Confirm that it was reshaped on write.
        self.assertIn('RESHAPED', sparse_map_in.metadata)
        self.assertTrue(sparse_map_in.metadata['RESHAPED'])

        self.assertEqual(sparse_map_in.n_valid, 2)

        # Test reading back partial coverage.
        cov_pixels = sparse_map._cov_map.cov_pixels([1_000_000, 100_000_000])

        sparse_map_in_partial = healsparse.HealSparseMap.read(
            fname,
            pixels=cov_pixels,
        )
        self.assertTrue(sparse_map_in_partial.is_bit_mask_map)
        self.assertEqual(sparse_map_in_partial.dtype, np.bool_)
        self.assertEqual(sparse_map_in_partial.sentinel, False)

        cov_pixels_partial, = np.where(sparse_map_in_partial.coverage_mask)
        testing.assert_array_equal(cov_pixels_partial, cov_pixels)
        testing.assert_array_equal(sparse_map_in_partial.valid_pixels, [1_000_000, 100_000_000])

    @pytest.mark.skipif(not healsparse.parquet_shim.use_pyarrow, reason='Requires pyarrow')
    def test_bit_mask_map_parquet_io(self):
        pass

    def test_bit_mask_map_fits_io_compression(self):
        nside_coverage = 32
        nside_map = 1024

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        sparse_map = HealSparseMap.make_empty(nside_coverage, nside_map, np.bool_, bit_mask=True)

        sparse_map[10_000_000: 11_000_000] = True

        fname = os.path.join(self.test_dir, f"healsparse_bitmask_{nside_map}.hsp")
        fname_nocomp = os.path.join(
            self.test_dir,
            f"healsparse_bitmask_{nside_map}_nocomp.hsp",
        )

        sparse_map.write(fname, nocompress=False)
        sparse_map.write(fname_nocomp, nocompress=True)

        sparse_map_in = HealSparseMap.read(fname)
        sparse_map_in_nocomp = HealSparseMap.read(fname_nocomp)

        self.assertTrue(sparse_map_in_nocomp.is_bit_mask_map)
        self.assertEqual(sparse_map_in_nocomp.dtype, np.bool_)
        self.assertEqual(sparse_map_in_nocomp.sentinel, False)

        testing.assert_array_equal(sparse_map_in_nocomp.valid_pixels, sparse_map_in.valid_pixels)

    def test_bit_mask_fracdet(self):
        nside_coverage = 32
        nside_map = 2**13

        # We compare the fracdet map generated with the bit_mask code to that
        # generated with a regular boolean map.
        sparse_map = HealSparseMap.make_empty(nside_coverage, nside_map, np.bool_, bit_mask=True)
        sparse_map_bool = HealSparseMap.make_empty(nside_coverage, nside_map, np.bool_)

        sparse_map[10000: 20000] = True
        sparse_map_bool[10000: 20000] = True
        sparse_map[1000000: 2000000] = True
        sparse_map_bool[1000000: 2000000] = True

        fracdet = sparse_map.fracdet_map(128)
        fracdet_bool = sparse_map.fracdet_map(128)

        self.assertEqual(fracdet.n_valid, fracdet_bool.n_valid)
        testing.assert_array_equal(fracdet.valid_pixels, fracdet_bool.valid_pixels)
        testing.assert_array_almost_equal(
            fracdet[fracdet.valid_pixels],
            fracdet_bool[fracdet_bool.valid_pixels],
        )

    def test_bit_mask_from_other_map(self):
        nside_coverage = 32

        for mode in ['regular', 'wide', 'recarray']:
            if mode == 'regular':
                sparse_map = HealSparseMap.make_empty(nside_coverage, 2**15, np.bool_)
                value = True
            elif mode == 'wide':
                sparse_map = HealSparseMap.make_empty(
                    nside_coverage,
                    2**15,
                    healsparse.WIDE_MASK,
                    wide_mask_maxbits=16,
                )
                value = np.zeros(2, dtype=np.uint8)
                value[0] = 1
            elif mode == 'recarray':
                dtype = [('a', 'f8'), ('b', 'i4')]
                sparse_map = HealSparseMap.make_empty(
                    nside_coverage,
                    2**15,
                    dtype,
                    primary='a',
                )
                value = np.zeros(1, dtype=dtype)
                value['a'] = 1.0
            elif mode == 'bitmask':
                sparse_map = HealSparseMap.make_empty(
                    nside_coverage,
                    2**15,
                    np.bool_,
                    bit_mask=True,
                )
                value = True

            sparse_map[0: 100] = value
            sparse_map[10_000_000: 11_000_000] = value

            bitmask_map = sparse_map.as_bit_mask_map()

            self.assertTrue(bitmask_map.is_bit_mask_map)
            self.assertEqual(bitmask_map.n_valid, sparse_map.n_valid)
            testing.assert_array_equal(bitmask_map.valid_pixels, sparse_map.valid_pixels)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__ == '__main__':
    unittest.main()
