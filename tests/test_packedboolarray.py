import unittest
import numpy.testing as testing
import numpy as np
import tempfile
import os
import shutil
import pytest

import healsparse
from healsparse import HealSparseMap
from healsparse.packedBoolArray import _PackedBoolArray


class PackedBoolArrayTestCase(unittest.TestCase):
    """Tests for _PackedBoolArray."""
    def _make_short_arrays(self):
        # This creates 4 short (<= 8) arrays:
        # First is the full 8 (start = 0, end = 8).
        # Second starts at 0, ends less than 8.
        # Third starts at >0, ends at 8.
        # Fourth starts at >0, ends at <8.

        short_arrays = []

        array = np.zeros(8, dtype=np.bool_)
        array[0] = True
        array[4] = True
        array[7] = True

        data = np.packbits(array, bitorder="little")

        short_arrays.append(_PackedBoolArray(data_buffer=data.copy(), start_index=0, stop_index=8))
        short_arrays.append(_PackedBoolArray(data_buffer=data.copy(), start_index=0, stop_index=6))
        short_arrays.append(_PackedBoolArray(data_buffer=data.copy(), start_index=1, stop_index=8))
        short_arrays.append(_PackedBoolArray(data_buffer=data.copy(), start_index=1, stop_index=6))

        return short_arrays

    def _make_middle_arrays(self):
        # This creates 4 middle (8 < size < 16) arrays:
        # First is the full 16 (start = 0, end = 16).
        # Second starts at 0, ends less than 16.
        # Third starts at >0, ends at 16.
        # Fourth starts at >0, ends at <16.

        middle_arrays = []

        array = np.zeros(16, dtype=np.bool_)
        array[0] = True
        array[4] = True
        array[7] = True
        array[12] = True
        array[15] = True

        data = np.packbits(array, bitorder="little")

        middle_arrays.append(_PackedBoolArray(data_buffer=data.copy(), start_index=0, stop_index=16))
        middle_arrays.append(_PackedBoolArray(data_buffer=data.copy(), start_index=0, stop_index=15))
        middle_arrays.append(_PackedBoolArray(data_buffer=data.copy(), start_index=1, stop_index=16))
        middle_arrays.append(_PackedBoolArray(data_buffer=data.copy(), start_index=1, stop_index=15))

        return middle_arrays

    def _make_long_arrays(self):
        # This creates 4 long (size > 16) arrays:
        # First is the full 32 (start = 0, end = 32).
        # Second starts at 0, ends less than 32.
        # Third starts at >0, ends at 32.
        # Fourth starts at >0, ends at <32.

        long_arrays = []

        array = np.zeros(32, dtype=np.bool_)
        array[0] = True
        array[4] = True
        array[7] = True
        array[12] = True
        array[15] = True
        array[20] = True
        array[24] = True
        array[29] = True
        array[31] = True

        data = np.packbits(array, bitorder="little")

        long_arrays.append(_PackedBoolArray(data_buffer=data.copy(), start_index=0, stop_index=32))
        long_arrays.append(_PackedBoolArray(data_buffer=data.copy(), start_index=0, stop_index=31))
        long_arrays.append(_PackedBoolArray(data_buffer=data.copy(), start_index=1, stop_index=32))
        long_arrays.append(_PackedBoolArray(data_buffer=data.copy(), start_index=1, stop_index=31))

        return long_arrays

    def test_create_size(self):
        # Create specifying nothing, should be 0 size.
        pba = _PackedBoolArray()
        self.assertEqual(pba.size, 0)
        testing.assert_array_equal(np.array(pba), np.zeros(0, dtype=np.bool_))

        # Specify a few sizes.
        for size in range(128):
            pba = _PackedBoolArray(size=size)
            self.assertEqual(pba.size, size)
            testing.assert_array_equal(np.array(pba), np.zeros(size, dtype=np.bool_))

        # Specify a few sizes with an offset.
        for size in range(128):
            pba = _PackedBoolArray(size=size, start_index=2)
            self.assertEqual(pba.size, size)
            testing.assert_array_equal(np.array(pba), np.zeros(size, dtype=np.bool_))

        with self.assertRaises(ValueError):
            pba = _PackedBoolArray(size=10, data_buffer=np.zeros(5, dtype=np.uint8))

        with self.assertRaises(ValueError):
            pba = _PackedBoolArray(size=20, start_index=-1)

        with self.assertRaises(ValueError):
            pba = _PackedBoolArray(size=20, start_index=10)

        with self.assertRaises(ValueError):
            pba = _PackedBoolArray(data_buffer=np.zeros(5, dtype=np.int64))

        with self.assertRaises(ValueError):
            pba = _PackedBoolArray(data_buffer=7)

        with self.assertRaises(ValueError):
            pba = _PackedBoolArray(data_buffer=np.zeros(5, dtype=np.int64), stop_index=10)

        with self.assertRaises(ValueError):
            pba = _PackedBoolArray(size=16, stop_index=10)

    def test_create_data_buffer(self):
        # Default, we can infer the size from the data buffer.
        for size in range(0, 128, 8):
            data_buffer = np.zeros(size // 8, dtype=np.uint8)
            pba = _PackedBoolArray(data_buffer=data_buffer)
            self.assertEqual(pba.size, size)
            testing.assert_array_equal(np.array(pba), np.zeros(size, dtype=np.bool_))

        # If we want a specific size, we must also specify the stop index.
        for size in range(0, 128, 8):
            for start_index in range(1, 8):
                data_buffer = np.zeros(size // 8 + 1, dtype=np.uint8)
                pba = _PackedBoolArray(
                    data_buffer=data_buffer,
                    start_index=start_index,
                    stop_index=start_index + size,
                )
                self.assertEqual(pba.size, size)
                testing.assert_array_equal(np.array(pba), np.zeros(size, dtype=np.bool_))

    def test_create_from_bool_array(self):
        for size in range(128):
            arr = np.ones(size, dtype=np.bool_)
            pba = _PackedBoolArray.from_boolean_array(arr)
            self.assertEqual(pba.size, size)
            testing.assert_array_equal(np.array(pba), arr)

        for size in range(128):
            for start_index in range(0, 8):
                arr = np.ones(size, dtype=np.bool_)
                pba = _PackedBoolArray.from_boolean_array(arr, start_index=start_index)
                self.assertEqual(pba.size, size)
                testing.assert_array_equal(np.array(pba), arr)

        with self.assertRaises(NotImplementedError):
            pba = _PackedBoolArray.from_boolean_array(7)

        with self.assertRaises(NotImplementedError):
            pba = _PackedBoolArray.from_boolean_array(np.zeros(10, dtype=np.int64))

    def test_resize(self):
        for newsize in range(16, 128):
            pba = _PackedBoolArray(size=8)
            pba[[0, 4, 6]] = True

            pba.resize(newsize)

            self.assertEqual(pba.size, newsize)
            comp_array = np.zeros(newsize, dtype=np.bool_)
            comp_array[[0, 4, 6]] = True
            testing.assert_array_equal(np.array(pba), comp_array)

        for newsize in range(16, 128):
            for start_index in range(8):
                pba = _PackedBoolArray(size=8, start_index=start_index)
                pba[[0, 4, 6]] = True

                pba.resize(newsize)

                self.assertEqual(pba.size, newsize)
                comp_array = np.zeros(newsize, dtype=np.bool_)
                comp_array[[0, 4, 6]] = True
                testing.assert_array_equal(np.array(pba), comp_array)

        with self.assertRaises(ValueError):
            pba = _PackedBoolArray(size=10)
            pba.resize(6)

    def test_copy(self):
        for mode in ("short", "middle", "long"):
            if mode == "short":
                pba_arrays = self._make_short_arrays()
            elif mode == "middle":
                pba_arrays = self._make_middle_arrays()
            elif mode == "long":
                pba_arrays = self._make_long_arrays()

            for pba in pba_arrays:
                pba_test = pba.copy()
                pba_array = np.array(pba)

                testing.assert_array_equal(np.array(pba_test), pba_array)
                self.assertEqual(pba_test._start_index, pba._start_index)
                self.assertEqual(pba_test._stop_index, pba._stop_index)

    def test_repr(self):
        pba = _PackedBoolArray(size=2**10)
        self.assertEqual(repr(pba), f"_PackedBoolArray(size={2**10})")
        self.assertEqual(str(pba), f"_PackedBoolArray(size={2**10})")

        pba = _PackedBoolArray(size=2**10, start_index=2)
        self.assertEqual(repr(pba), f"_PackedBoolArray(size={2**10}, start_index=2)")
        self.assertEqual(str(pba), f"_PackedBoolArray(size={2**10}, start_index=2)")

    def test_sum(self):
        # Do sums over all the array options.
        for mode in ("short", "middle", "long"):
            if mode == "short":
                pba_arrays = self._make_short_arrays()
            elif mode == "middle":
                pba_arrays = self._make_middle_arrays()
            elif mode == "long":
                pba_arrays = self._make_long_arrays()

            for pba in pba_arrays:
                self.assertEqual(pba.sum(), np.array(pba).sum())

        # Do sums over aligned arrays, with and without reshaping.
        for mode in ("short", "middle", "long"):
            if mode == "short":
                pba = self._make_short_arrays()[0]
            elif mode == "middle":
                pba = self._make_middle_arrays()[0]
            elif mode == "long":
                pba = self._make_long_arrays()[0]

            shape = (pba.size // 8, 8)
            testing.assert_array_equal(
                pba.sum(shape=shape),
                np.array(pba).reshape(shape).sum(),
            )

            shape = (pba.size // 8, 8)
            testing.assert_array_equal(
                pba.sum(shape=shape, axis=1),
                np.array(pba).reshape(shape).sum(axis=1),
            )

        with self.assertRaises(ValueError):
            pba = _PackedBoolArray(size=10)
            pba.sum(shape=(10, 1))

        with self.assertRaises(ValueError):
            pba = _PackedBoolArray(size=16, start_index=2)
            pba.sum(shape=(8, 2))

        with self.assertRaises(ValueError):
            pba = _PackedBoolArray(size=16)
            pba.sum(shape=(2, 8), axis=2)

        with self.assertRaises(ValueError):
            pba = _PackedBoolArray(size=16)
            pba.sum(shape=(8, 2), axis=1)

        with self.assertRaises(NotImplementedError):
            pba = _PackedBoolArray(size=16)
            pba.sum(shape=(2, 8), axis=0)

    def test_data_array(self):
        pba = _PackedBoolArray(size=16)
        testing.assert_array_equal(pba.data_array, pba._data)

        with self.assertRaises(NotImplementedError):
            pba = _PackedBoolArray(size=16, start_index=2)
            pba.data_array

    def test_view(self):
        for mode in ("short", "middle", "long"):
            if mode == "short":
                pba_arrays = self._make_short_arrays()
            elif mode == "middle":
                pba_arrays = self._make_middle_arrays()
            elif mode == "long":
                pba_arrays = self._make_long_arrays()

            for pba in pba_arrays:
                pba[0] = True

                if pba.start_index == 0:
                    # Only use this method of accessing data_array if
                    # start_index == 0.
                    pba2 = _PackedBoolArray(data_buffer=pba.data_array)
                    testing.assert_array_equal(pba2._data, pba._data)

                    # This should change both.
                    pba2[0] = False
                    testing.assert_array_equal(pba2._data, pba._data)

                # And another way to get a view.
                pba3 = pba[:]
                testing.assert_array_equal(pba3._data, pba._data)

                # This should change both.
                pba2[0] = True
                testing.assert_array_equal(pba3._data, pba._data)

    def test_setitem_short(self):
        short_arrays = self._make_short_arrays()

        for pba in short_arrays:
            # Test setting a single index to True or False.
            pba_test = pba.copy()
            pba_array = np.array(pba)

            pba_test[2] = True
            pba_array[2] = True
            testing.assert_array_equal(pba_test, pba_array)

            pba_test[2] = False
            pba_array[2] = False

            testing.assert_array_equal(pba_test, pba_array)

            # Test setting a list/array of indexes to True or False.
            pba_test = pba.copy()
            pba_array = np.array(pba)

            pba_test[[1, 2, 4]] = True
            pba_array[[1, 2, 4]] = True
            testing.assert_array_equal(pba_test, pba_array)

            pba_test[np.array([1, 2, 4])] = False
            pba_array[np.array([1, 2, 4])] = False
            testing.assert_array_equal(pba_test, pba_array)

            pba_test[np.array([], dtype=np.int64)] = True
            testing.assert_array_equal(pba_test, pba_array)

            pba_test[np.array([], dtype=np.int64)] = False
            testing.assert_array_equal(pba_test, pba_array)

            # Test setting a list/array of indices to an array.
            pba_test = pba.copy()
            pba_array = np.array(pba)

            pba_test[[1, 2, 4]] = np.array([True, True, False])
            pba_array[[1, 2, 4]] = np.array([True, True, False])
            testing.assert_array_equal(pba_test, pba_array)

            # Test setting slices: setting to True/False
            for full in (True, False):
                pba_test = pba.copy()
                pba_array = np.array(pba)

                if full:
                    s = slice(None, None, None)
                else:
                    s = slice(1, 4, None)

                pba_test[s] = True
                pba_array[s] = True
                testing.assert_array_equal(pba_test, pba_array)

                pba_test[s] = False
                pba_array[s] = False
                testing.assert_array_equal(pba_test, pba_array)

            # Test setting slices: setting to a numpy array.
            for full in (True, False):
                pba_test = pba.copy()
                pba_array = np.array(pba)

                if full:
                    s = slice(None, None, None)
                else:
                    s = slice(1, 4, None)

                target = np.ones(len(pba_test[s]), dtype=np.bool_)
                pba_test[s] = target
                pba_array[s] = target
                testing.assert_array_equal(pba_test, pba_array)

            # Test setting slices: setting to packed boolean array.
            for full in (True, False):
                pba_test = pba.copy()
                pba_array = np.array(pba)

                if full:
                    s = slice(None, None, None)
                else:
                    # The slice must be aligned.
                    s = slice(1, 4, None)

                # This gets us an "aligned" array.  Otherwise I think
                # we need to unpack prior to setting.
                target = pba_test[s].copy()
                target[:] = True

                pba_test[s] = target
                pba_array[s] = np.array(target)
                testing.assert_array_equal(pba_test, pba_array)

        with self.assertRaises(ValueError):
            pba = short_arrays[0].copy()
            pba[:] = np.zeros(len(pba), dtype=np.int64)

        with self.assertRaises(ValueError):
            pba = short_arrays[0].copy()
            pba[:] = np.zeros(len(pba) - 1, dtype=np.bool_)

        with self.assertRaises(ValueError):
            pba = short_arrays[0].copy()
            pba[:] = _PackedBoolArray(size=len(pba), start_index=2)

        with self.assertRaises(ValueError):
            pba = short_arrays[0].copy()
            pba[:] = 8

        with self.assertRaises(IndexError):
            pba = short_arrays[0].copy()
            pba[[5.0]] = True

        with self.assertRaises(IndexError):
            pba = short_arrays[0].copy()
            pba[5.0] = True

    def test_setitem_middle(self):
        middle_arrays = self._make_middle_arrays()

        for pba in middle_arrays:
            # Test setting a single index to True or False.
            pba_test = pba.copy()
            pba_array = np.array(pba)

            pba_test[10] = True
            pba_array[10] = True
            testing.assert_array_equal(pba_test, pba_array)

            pba_test[10] = False
            pba_array[10] = False

            testing.assert_array_equal(pba_test, pba_array)

            # Test setting a list/array of indexes to True or False.
            pba_test = pba.copy()
            pba_array = np.array(pba)

            pba_test[[1, 2, 4, 10]] = True
            pba_array[[1, 2, 4, 10]] = True
            testing.assert_array_equal(pba_test, pba_array)

            pba_test[np.array([1, 2, 4, 10])] = False
            pba_array[np.array([1, 2, 4, 10])] = False
            testing.assert_array_equal(pba_test, pba_array)

            # Test setting a list/array of indices to an array.
            pba_test = pba.copy()
            pba_array = np.array(pba)

            pba_test[[1, 2, 4, 10]] = np.array([True, True, False, False])
            pba_array[[1, 2, 4, 10]] = np.array([True, True, False, False])
            testing.assert_array_equal(pba_test, pba_array)

            # Test setting slices: setting to True/False
            for full in (True, False):
                pba_test = pba.copy()
                pba_array = np.array(pba)

                if full:
                    s = slice(None, None, None)
                else:
                    s = slice(6, 11, None)

                pba_test[s] = True
                pba_array[s] = True
                testing.assert_array_equal(pba_test, pba_array)

                pba_test[s] = False
                pba_array[s] = False
                testing.assert_array_equal(pba_test, pba_array)

            # Test setting slices: setting to a numpy array.
            for full in (True, False):
                pba_test = pba.copy()
                pba_array = np.array(pba)

                if full:
                    s = slice(None, None, None)
                else:
                    s = slice(6, 11, None)

                target = np.ones(len(pba_test[s]), dtype=np.bool_)
                pba_test[s] = target
                pba_array[s] = target
                testing.assert_array_equal(pba_test, pba_array)

            # Test setting slices: setting to packed boolean array.
            for full in (True, False):
                pba_test = pba.copy()
                pba_array = np.array(pba)

                if full:
                    s = slice(None, None, None)
                else:
                    # The slice must be aligned.
                    s = slice(6, 11, None)

                # This gets us an "aligned" array.
                target = pba_test[s].copy()
                target[:] = True

                pba_test[s] = target
                pba_array[s] = np.array(target)
                testing.assert_array_equal(pba_test, pba_array)

    def test_setitem_long(self):
        long_arrays = self._make_long_arrays()

        for pba in long_arrays:
            # Test setting a single index to True or False.
            pba_test = pba.copy()
            pba_array = np.array(pba)

            pba_test[20] = True
            pba_array[20] = True
            testing.assert_array_equal(pba_test, pba_array)

            pba_test[20] = False
            pba_array[20] = False

            testing.assert_array_equal(pba_test, pba_array)

            # Test setting a list/array of indexes to True or False.
            pba_test = pba.copy()
            pba_array = np.array(pba)

            pba_test[[1, 2, 4, 10, 20, 29]] = True
            pba_array[[1, 2, 4, 10, 20, 29]] = True
            testing.assert_array_equal(pba_test, pba_array)

            pba_test[np.array([1, 2, 4, 10, 20, 29])] = False
            pba_array[np.array([1, 2, 4, 10, 20, 29])] = False
            testing.assert_array_equal(pba_test, pba_array)

            # Test setting a list/array of indices to an array.
            pba_test = pba.copy()
            pba_array = np.array(pba)

            pba_test[[1, 2, 4, 10, 20, 29]] = np.array([True, True, False, False, False, True])
            pba_array[[1, 2, 4, 10, 20, 29]] = np.array([True, True, False, False, False, True])
            testing.assert_array_equal(pba_test, pba_array)

            # Test setting slices: setting to True/False
            for mode in ("full", "aligned", "unaligned"):
                pba_test = pba.copy()
                pba_array = np.array(pba)

                if mode == "full":
                    s = slice(None, None, None)
                elif mode == "aligned":
                    s = slice(8, 16, None)
                else:
                    s = slice(1, 30, None)

                pba_test[s] = True
                pba_array[s] = True
                testing.assert_array_equal(pba_test, pba_array)

                pba_test[s] = False
                pba_array[s] = False
                testing.assert_array_equal(pba_test, pba_array)

            # Test setting slices: setting to a numpy array.
            for mode in ("full", "aligned", "unaligned"):
                pba_test = pba.copy()
                pba_array = np.array(pba)

                if mode == "full":
                    s = slice(None, None, None)
                elif mode == "aligned":
                    s = slice(8, 16, None)
                else:
                    s = slice(1, 30, None)

                target = np.ones(len(pba_test[s]), dtype=np.bool_)
                pba_test[s] = target
                pba_array[s] = target
                testing.assert_array_equal(pba_test, pba_array)

            # Test setting slices: setting to packed boolean array.
            for mode in ("full", "aligned", "unaligned"):

                pba_test = pba.copy()
                pba_array = np.array(pba)

                if mode == "full":
                    s = slice(None, None, None)
                elif mode == "aligned":
                    s = slice(8, 16, None)
                else:
                    s = slice(1, 30, None)

                # This gets us an "aligned" array.  Otherwise I think
                # we need to unpack prior to setting.
                target = pba_test[s].copy()
                target[:] = True

                pba_test[s] = target
                pba_array[s] = np.array(target)
                testing.assert_array_equal(pba_test, pba_array)

    def test_getitem_short(self):
        short_arrays = self._make_short_arrays()

        for pba in short_arrays:
            # Test getting a single index.
            arr = np.array(pba)
            for i in range(len(pba)):
                self.assertEqual(pba[i], arr[i])

            # Test getting slices.
            # The full slice.
            testing.assert_array_equal(np.array(pba[:]), arr[:])
            testing.assert_array_equal(np.array(pba[0: len(pba)]), arr[0: len(pba)])

            # Start at 0, end at less than the last:
            testing.assert_array_equal(np.array(pba[: len(pba) - 2]), arr[: len(pba) - 2])
            testing.assert_array_equal(np.array(pba[0: len(pba) - 2]), arr[0: len(pba) - 2])

            # Start at 1, end at end:
            testing.assert_array_equal(np.array(pba[1:]), arr[1:])
            testing.assert_array_equal(np.array(pba[1: len(pba)]), arr[1: len(pba)])

            # Start at 1, end at less than the last:
            testing.assert_array_equal(np.array(pba[1: len(pba) - 2]), arr[1: len(pba) - 2])
            testing.assert_array_equal(np.array(pba[1: -2]), arr[1: -2])

            # And get an array of indices:
            inds = np.arange(len(pba))
            testing.assert_array_equal(pba[inds], arr[inds])
            testing.assert_array_equal(pba[list(inds)], arr[list(inds)])

            # And an empty array of indices.
            inds = np.array([], dtype=np.int64)
            testing.assert_array_equal(pba[inds], arr[inds])

        with self.assertRaises(IndexError):
            pba = short_arrays[0].copy()
            pba[[5.0]]

        with self.assertRaises(IndexError):
            pba = short_arrays[0].copy()
            pba[5.0]

        with self.assertRaises(NotImplementedError):
            pba = short_arrays[0].copy()
            pba[::2]

        with self.assertRaises(IndexError):
            pba = short_arrays[0].copy()
            pba[-1]

        with self.assertRaises(IndexError):
            pba = short_arrays[0].copy()
            pba[len(pba)]

    def test_getitem_middle(self):
        middle_arrays = self._make_middle_arrays()

        for pba in middle_arrays:
            # Test getting a single index.
            arr = np.array(pba)
            for i in range(len(pba)):
                self.assertEqual(pba[i], arr[i])

            # Test getting slices.
            # The full slice.
            testing.assert_array_equal(np.array(pba[:]), arr[:])
            testing.assert_array_equal(np.array(pba[0: len(pba)]), arr[0: len(pba)])

            # Start at 0, end at less than the last:
            testing.assert_array_equal(np.array(pba[: len(pba) - 2]), arr[: len(pba) - 2])
            testing.assert_array_equal(np.array(pba[0: len(pba) - 2]), arr[0: len(pba) - 2])

            # Start at 1, end at end:
            testing.assert_array_equal(np.array(pba[1:]), arr[1:])
            testing.assert_array_equal(np.array(pba[1: len(pba)]), arr[1: len(pba)])

            # Start at 1, end at less than the last:
            testing.assert_array_equal(np.array(pba[1: len(pba) - 2]), arr[1: len(pba) - 2])
            testing.assert_array_equal(np.array(pba[1: -2]), arr[1: -2])

            # Start at 8, end at end:
            testing.assert_array_equal(np.array(pba[8:]), arr[8:])
            testing.assert_array_equal(np.array(pba[8: len(pba)]), arr[8: len(pba)])

            # Start at 9, end at end:
            testing.assert_array_equal(np.array(pba[9:]), arr[9:])
            testing.assert_array_equal(np.array(pba[9: len(pba)]), arr[9: len(pba)])

            # And get an array of indices:
            inds = np.arange(len(pba))
            testing.assert_array_equal(pba[inds], arr[inds])
            testing.assert_array_equal(pba[list(inds)], arr[list(inds)])

    def test_getitem_long(self):
        long_arrays = self._make_long_arrays()

        for pba in long_arrays:
            # Test getting a single index.
            arr = np.array(pba)
            for i in range(len(pba)):
                self.assertEqual(pba[i], arr[i])

            # Test getting slices.
            # The full slice.
            testing.assert_array_equal(np.array(pba[:]), arr[:])
            testing.assert_array_equal(np.array(pba[0: len(pba)]), arr[0: len(pba)])

            # Start at 0, end at less than the last:
            testing.assert_array_equal(np.array(pba[: len(pba) - 2]), arr[: len(pba) - 2])
            testing.assert_array_equal(np.array(pba[0: len(pba) - 2]), arr[0: len(pba) - 2])

            # Start at 1, end at end:
            testing.assert_array_equal(np.array(pba[1:]), arr[1:])
            testing.assert_array_equal(np.array(pba[1: len(pba)]), arr[1: len(pba)])

            # Start at 1, end at less than the last:
            testing.assert_array_equal(np.array(pba[1: len(pba) - 2]), arr[1: len(pba) - 2])
            testing.assert_array_equal(np.array(pba[1: -2]), arr[1: -2])

            # Start at 16, end at end:
            testing.assert_array_equal(np.array(pba[16:]), arr[16:])
            testing.assert_array_equal(np.array(pba[16: len(pba)]), arr[16: len(pba)])

            # Start at 16, end at 24 (aligned, less than end):
            testing.assert_array_equal(np.array(pba[16: 24]), arr[16: 24])

            # Start at 17, end at end:
            testing.assert_array_equal(np.array(pba[17:]), arr[17:])
            testing.assert_array_equal(np.array(pba[17: len(pba)]), arr[17: len(pba)])

            # And get an array of indices:
            inds = np.arange(len(pba))
            testing.assert_array_equal(pba[inds], arr[inds])
            testing.assert_array_equal(pba[list(inds)], arr[list(inds)])

    def test_and(self):
        # This test checks the and operation with "already sliced"
        # test arrays.
        for mode in ("short", "middle", "long"):
            if mode == "short":
                pba_arrays = self._make_short_arrays()
            elif mode == "middle":
                pba_arrays = self._make_middle_arrays()
            elif mode == "long":
                pba_arrays = self._make_long_arrays()

            for pba in pba_arrays:
                # Test with constants.
                pba_test = pba.copy()
                pba_array = np.array(pba)

                pba_test2 = pba_test & True
                pba_array2 = pba_array & True
                testing.assert_array_equal(pba_test2, pba_array2)

                pba_test2 = pba_test & False
                pba_array2 = pba_array & False
                testing.assert_array_equal(pba_test2, pba_array2)

                pba_test &= True
                pba_array &= True
                testing.assert_array_equal(pba_test, pba_array)

                # Test with _PackedBoolArray.
                pba_test = pba.copy()
                pba_array = np.array(pba)

                other = pba_test.copy()
                other[:] = True
                other[1] = False

                pba_test2 = pba_test & other
                pba_array2 = pba_array & np.array(other)
                testing.assert_array_equal(pba_test2, pba_array2)

                pba_test &= other
                pba_array &= np.array(other)
                testing.assert_array_equal(pba_test, pba_array)

        pba = self._make_short_arrays()[0]
        with self.assertRaises(ValueError):
            pba & _PackedBoolArray(size=len(pba) - 1)

        with self.assertRaises(ValueError):
            pba & _PackedBoolArray(size=len(pba), start_index=1)

        with self.assertRaises(NotImplementedError):
            pba & 7

    def test_and_sliced(self):
        # This test checks the and operation while slicing
        # the test arrays.
        for mode in ("short", "middle", "long"):
            if mode == "short":
                pba_arrays = self._make_short_arrays()
            elif mode == "middle":
                pba_arrays = self._make_middle_arrays()
            elif mode == "long":
                pba_arrays = self._make_long_arrays()

            for pba in pba_arrays:
                # Test with constants.
                pba_test = pba.copy()
                pba_array = np.array(pba)

                pba_test2 = pba_test[1: -2] & True
                pba_array2 = pba_array[1: -2] & True
                testing.assert_array_equal(pba_test2, pba_array2)

                pba_test2 = pba_test[1: -2] & False
                pba_array2 = pba_array[1: -2] & False
                testing.assert_array_equal(pba_test2, pba_array2)

                pba_test[1: -2] &= True
                pba_array[1: -2] &= True
                testing.assert_array_equal(pba_test, pba_array)

                # Test with _PackedBoolArray.
                pba_test = pba.copy()
                pba_array = np.array(pba)

                other = pba_test.copy()
                other[:] = True
                other[1] = False

                pba_test2 = pba_test[1: -2] & other[1: -2]
                pba_array2 = pba_array[1: -2] & np.array(other)[1: -2]
                testing.assert_array_equal(pba_test2, pba_array2)

                pba_test[1: -2] &= other[1: -2]
                pba_array[1: -2] &= np.array(other)[1: -2]
                testing.assert_array_equal(pba_test, pba_array)

    def test_or(self):
        # This test checks the or operation with "already sliced"
        # test arrays.
        for mode in ("short", "middle", "long"):
            if mode == "short":
                pba_arrays = self._make_short_arrays()
            elif mode == "middle":
                pba_arrays = self._make_middle_arrays()
            elif mode == "long":
                pba_arrays = self._make_long_arrays()

            for pba in pba_arrays:
                # Test with constants.
                pba_test = pba.copy()
                pba_array = np.array(pba)

                pba_test2 = pba_test | True
                pba_array2 = pba_array | True
                testing.assert_array_equal(pba_test2, pba_array2)

                pba_test2 = pba_test | False
                pba_array2 = pba_array | False
                testing.assert_array_equal(pba_test2, pba_array2)

                pba_test |= True
                pba_array |= True
                testing.assert_array_equal(pba_test, pba_array)

                # Test with _PackedBoolArray.
                pba_test = pba.copy()
                pba_array = np.array(pba)

                other = pba_test.copy()
                other[:] = True
                other[1] = False

                pba_test2 = pba_test | other
                pba_array2 = pba_array | np.array(other)
                testing.assert_array_equal(pba_test2, pba_array2)

                pba_test |= other
                pba_array |= np.array(other)
                testing.assert_array_equal(pba_test, pba_array)

        pba = self._make_short_arrays()[0]
        with self.assertRaises(ValueError):
            pba | _PackedBoolArray(size=len(pba) - 1)

        with self.assertRaises(ValueError):
            pba | _PackedBoolArray(size=len(pba), start_index=1)

        with self.assertRaises(NotImplementedError):
            pba | 7

    def test_or_sliced(self):
        # This test checks the or operation while slicing
        # the test arrays.
        for mode in ("short", "middle", "long"):
            if mode == "short":
                pba_arrays = self._make_short_arrays()
            elif mode == "middle":
                pba_arrays = self._make_middle_arrays()
            elif mode == "long":
                pba_arrays = self._make_long_arrays()

            for pba in pba_arrays:
                # Test with constants.
                pba_test = pba.copy()
                pba_array = np.array(pba)

                pba_test2 = pba_test[1: -2] | True
                pba_array2 = pba_array[1: -2] | True
                testing.assert_array_equal(pba_test2, pba_array2)

                pba_test2 = pba_test[1: -2] | False
                pba_array2 = pba_array[1: -2] | False
                testing.assert_array_equal(pba_test2, pba_array2)

                pba_test[1: -2] |= True
                pba_array[1: -2] |= True
                testing.assert_array_equal(pba_test, pba_array)

                # Test with _PackedBoolArray.
                pba_test = pba.copy()
                pba_array = np.array(pba)

                other = pba_test.copy()
                other[:] = True
                other[1] = False

                pba_test2 = pba_test[1: -2] | other[1: -2]
                pba_array2 = pba_array[1: -2] | np.array(other)[1: -2]
                testing.assert_array_equal(pba_test2, pba_array2)

                pba_test[1: -2] |= other[1: -2]
                pba_array[1: -2] |= np.array(other)[1: -2]
                testing.assert_array_equal(pba_test, pba_array)

                # Should do nothing.
                pba_test[np.array([], dtype=np.int64)] |= True
                testing.assert_array_equal(pba_test, pba_array)

    def test_xor(self):
        # This test checks the xor operation with "already sliced"
        # test arrays.
        for mode in ("short", "middle", "long"):
            if mode == "short":
                pba_arrays = self._make_short_arrays()
            elif mode == "middle":
                pba_arrays = self._make_middle_arrays()
            elif mode == "long":
                pba_arrays = self._make_long_arrays()

            for pba in pba_arrays:
                # Test with constants.
                pba_test = pba.copy()
                pba_array = np.array(pba)

                pba_test2 = pba_test ^ True
                pba_array2 = pba_array ^ True
                testing.assert_array_equal(pba_test2, pba_array2)

                pba_test2 = pba_test ^ False
                pba_array2 = pba_array ^ False
                testing.assert_array_equal(pba_test2, pba_array2)

                pba_test ^= True
                pba_array ^= True
                testing.assert_array_equal(pba_test, pba_array)

                # Test with _PackedBoolArray.
                pba_test = pba.copy()
                pba_array = np.array(pba)

                other = pba_test.copy()
                other[:] = True
                other[1] = False

                pba_test2 = pba_test ^ other
                pba_array2 = pba_array ^ np.array(other)
                testing.assert_array_equal(pba_test2, pba_array2)

                pba_test ^= other
                pba_array ^= np.array(other)
                testing.assert_array_equal(pba_test, pba_array)

        pba = self._make_short_arrays()[0]
        with self.assertRaises(ValueError):
            pba ^ _PackedBoolArray(size=len(pba) - 1)

        with self.assertRaises(ValueError):
            pba ^ _PackedBoolArray(size=len(pba), start_index=1)

        with self.assertRaises(NotImplementedError):
            pba ^ 7

    def test_xor_sliced(self):
        # This test checks the xor operation while slicing
        # the test arrays.
        for mode in ("short", "middle", "long"):
            if mode == "short":
                pba_arrays = self._make_short_arrays()
            elif mode == "middle":
                pba_arrays = self._make_middle_arrays()
            elif mode == "long":
                pba_arrays = self._make_long_arrays()

            for pba in pba_arrays:
                # Test with constants.
                pba_test = pba.copy()
                pba_array = np.array(pba)

                pba_test2 = pba_test[1: -2] ^ True
                pba_array2 = pba_array[1: -2] ^ True
                testing.assert_array_equal(pba_test2, pba_array2)

                pba_test2 = pba_test[1: -2] ^ False
                pba_array2 = pba_array[1: -2] ^ False
                testing.assert_array_equal(pba_test2, pba_array2)

                pba_test[1: -2] ^= True
                pba_array[1: -2] ^= True
                testing.assert_array_equal(pba_test, pba_array)

                # Test with _PackedBoolArray.
                pba_test = pba.copy()
                pba_array = np.array(pba)

                other = pba_test.copy()
                other[:] = True
                other[1] = False

                pba_test2 = pba_test[1: -2] ^ other[1: -2]
                pba_array2 = pba_array[1: -2] ^ np.array(other)[1: -2]
                testing.assert_array_equal(pba_test2, pba_array2)

                pba_test[1: -2] ^= other[1: -2]
                pba_array[1: -2] ^= np.array(other)[1: -2]
                testing.assert_array_equal(pba_test, pba_array)

    def test_invert(self):
        # This test checks the and operation with "already sliced"
        # test arrays.
        for mode in ("short", "middle", "long"):
            if mode == "short":
                pba_arrays = self._make_short_arrays()
            elif mode == "middle":
                pba_arrays = self._make_middle_arrays()
            elif mode == "long":
                pba_arrays = self._make_long_arrays()

            for pba in pba_arrays:
                # Test with constants.
                pba_test = pba.copy()
                pba_array = np.array(pba)

                pba_test2 = ~pba_test
                pba_array2 = ~pba_array
                testing.assert_array_equal(pba_test2, pba_array2)

                pba_test.invert()
                pba_array = np.invert(pba_array)
                testing.assert_array_equal(pba_test, pba_array)

    def test_invert_slice(self):
        # This test checks the invert operation while slicing
        # the test arrays.
        for mode in ("short", "middle", "long"):
            if mode == "short":
                pba_arrays = self._make_short_arrays()
            elif mode == "middle":
                pba_arrays = self._make_middle_arrays()
            elif mode == "long":
                pba_arrays = self._make_long_arrays()

            for pba in pba_arrays:
                # Test with constants.
                pba_test = pba.copy()
                pba_array = np.array(pba)

                pba_test[1: -2] = ~pba_test[1: -2]
                pba_array[1: -2] = ~pba_array[1: -2]
                testing.assert_array_equal(pba_test, pba_array)

                pba_test[1: -2].invert()
                pba_array[1: -2] = np.invert(pba_array[1: -2])
                testing.assert_array_equal(pba_test, pba_array)


class HealSparseBitPackedTestCase(unittest.TestCase):
    """Tests for HealSparseMap with bit_packed."""
    def test_make_bit_packed_map(self):
        nside_coverage = 32

        sparse_map = HealSparseMap.make_empty(nside_coverage, 2**15, np.bool_, bit_packed=True)
        self.assertTrue(sparse_map.is_bit_packed_map)
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
                bit_packed=True,
                sentinel=True,
            )

        with self.assertRaises(ValueError):
            sparse_map = HealSparseMap.make_empty(nside_coverage, 64, np.bool_, bit_packed=True)

    def test_bit_packed_map_fits_io(self):
        nside_coverage = 32
        nside_map = 1024

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        sparse_map = HealSparseMap.make_empty(nside_coverage, nside_map, np.bool_, bit_packed=True)

        sparse_map[10_000_000: 11_000_000] = True

        fname = os.path.join(self.test_dir, f"healsparse_bitpacked_{nside_map}.hsp")
        sparse_map.write(fname, clobber=True)
        sparse_map_in = HealSparseMap.read(fname)

        self.assertTrue(sparse_map_in.is_bit_packed_map)
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
        self.assertTrue(sparse_map_in_partial.is_bit_packed_map)
        self.assertEqual(sparse_map_in_partial.dtype, np.bool_)
        self.assertEqual(sparse_map_in_partial.sentinel, False)

        cov_pixels = sparse_map._cov_map.cov_pixels(sparse_map.valid_pixels)
        pixel_sub = sparse_map.valid_pixels[(cov_pixels == covered_pixels[1]) |
                                            (cov_pixels == covered_pixels[10])]
        testing.assert_array_equal(sparse_map_in_partial.valid_pixels, pixel_sub)

    @pytest.mark.skipif("GITHUB_ACTIONS" in os.environ, reason='Giant test cannot be run on GHA')
    def notest_bit_packed_map_fits_io_giant(self):
        # I don't know how to test this.
        nside_coverage = 32
        nside_map = 2**17

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')
        sparse_map = HealSparseMap.make_empty(
            nside_coverage,
            nside_map,
            np.bool_,
            bit_packed=True,
            cov_pixels=np.arange(1500),
        )

        sparse_map[1_000_000] = True
        sparse_map[100_000_000] = True

        fname = os.path.join(self.test_dir, f"healsparse_bitpacked_{nside_map}_giant.hsp")
        sparse_map.write(fname, clobber=True)
        sparse_map_in = HealSparseMap.read(fname)

        # Confirm it's the correct type.
        self.assertTrue(sparse_map_in.is_bit_packed_map)
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
        self.assertTrue(sparse_map_in_partial.is_bit_packed_map)
        self.assertEqual(sparse_map_in_partial.dtype, np.bool_)
        self.assertEqual(sparse_map_in_partial.sentinel, False)

        cov_pixels_partial, = np.where(sparse_map_in_partial.coverage_mask)
        testing.assert_array_equal(cov_pixels_partial, cov_pixels)
        testing.assert_array_equal(sparse_map_in_partial.valid_pixels, [1_000_000, 100_000_000])

    @pytest.mark.skipif(not healsparse.parquet_shim.use_pyarrow, reason='Requires pyarrow')
    def test_bit_packed_map_parquet_io(self):
        nside_coverage = 32
        nside_map = 1024

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        sparse_map = HealSparseMap.make_empty(nside_coverage, nside_map, np.bool_, bit_packed=True)

        sparse_map[10_000_000: 11_000_000] = True

        fname = os.path.join(self.test_dir, f"healsparse_bitpacked_{nside_map}.hsp.parquet")
        sparse_map.write(fname, clobber=True, format="parquet")
        sparse_map_in = HealSparseMap.read(fname)

        self.assertTrue(sparse_map_in.is_bit_packed_map)
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
        self.assertTrue(sparse_map_in_partial.is_bit_packed_map)
        self.assertEqual(sparse_map_in_partial.dtype, np.bool_)
        self.assertEqual(sparse_map_in_partial.sentinel, False)

        cov_pixels = sparse_map._cov_map.cov_pixels(sparse_map.valid_pixels)
        pixel_sub = sparse_map.valid_pixels[(cov_pixels == covered_pixels[1]) |
                                            (cov_pixels == covered_pixels[10])]
        testing.assert_array_equal(sparse_map_in_partial.valid_pixels, pixel_sub)

    def test_bit_packed_map_fits_io_compression(self):
        nside_coverage = 32
        nside_map = 1024

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        sparse_map = HealSparseMap.make_empty(nside_coverage, nside_map, np.bool_, bit_packed=True)

        sparse_map[10_000_000: 11_000_000] = True

        fname = os.path.join(self.test_dir, f"healsparse_bitpacked_{nside_map}.hsp")
        fname_nocomp = os.path.join(
            self.test_dir,
            f"healsparse_bitpacked_{nside_map}_nocomp.hsp",
        )

        sparse_map.write(fname, nocompress=False)
        sparse_map.write(fname_nocomp, nocompress=True)

        sparse_map_in = HealSparseMap.read(fname)
        sparse_map_in_nocomp = HealSparseMap.read(fname_nocomp)

        self.assertTrue(sparse_map_in_nocomp.is_bit_packed_map)
        self.assertEqual(sparse_map_in_nocomp.dtype, np.bool_)
        self.assertEqual(sparse_map_in_nocomp.sentinel, False)

        testing.assert_array_equal(sparse_map_in_nocomp.valid_pixels, sparse_map_in.valid_pixels)

    def test_bit_packed_fracdet(self):
        nside_coverage = 32
        nside_map = 2**13

        # We compare the fracdet map generated with the bit_packed code to that
        # generated with a regular boolean map.
        sparse_map = HealSparseMap.make_empty(nside_coverage, nside_map, np.bool_, bit_packed=True)
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

    def test_bit_packed_from_other_map(self):
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
            elif mode == 'bitpacked':
                sparse_map = HealSparseMap.make_empty(
                    nside_coverage,
                    2**15,
                    np.bool_,
                    bit_packed=True,
                )
                value = True

            sparse_map[0: 100] = value
            sparse_map[10_000_000: 11_000_000] = value

            bitpacked_map = sparse_map.as_bit_packed_map()

            self.assertTrue(bitpacked_map.is_bit_packed_map)
            self.assertEqual(bitpacked_map.n_valid, sparse_map.n_valid)
            testing.assert_array_equal(bitpacked_map.valid_pixels, sparse_map.valid_pixels)

    def test_bit_packed_update_replace(self):
        nside_coverage = 32
        nside_map = 2**10

        bitpacked_map = HealSparseMap.make_empty(nside_coverage, nside_map, np.bool_, bit_packed=True)
        bool_map = HealSparseMap.make_empty(nside_coverage, nside_map, np.bool_)

        bitpacked_map.update_values_pix(np.arange(10000, 20000), True, operation="replace")
        bool_map.update_values_pix(np.arange(10000, 20000), True, operation="replace")

        testing.assert_array_equal(bitpacked_map.valid_pixels, bool_map.valid_pixels)

    def test_bit_packed_update_and(self):
        nside_coverage = 32
        nside_map = 2**10

        bitpacked_map = HealSparseMap.make_empty(nside_coverage, nside_map, np.bool_, bit_packed=True)
        bool_map = HealSparseMap.make_empty(nside_coverage, nside_map, np.bool_)

        bitpacked_map[10000: 20000] = True
        bool_map[10000: 20000] = True

        bitpacked_map.update_values_pix(np.arange(15000, 25000), True, operation="and")
        bool_map.update_values_pix(np.arange(15000, 25000), True, operation="and")

        testing.assert_array_equal(bitpacked_map.valid_pixels, bool_map.valid_pixels)

        bitpacked_map.update_values_pix(np.arange(15000, 25000), False, operation="and")
        bool_map.update_values_pix(np.arange(15000, 25000), False, operation="and")

        testing.assert_array_equal(bitpacked_map.valid_pixels, bool_map.valid_pixels)

    def test_bit_packed_update_or(self):
        nside_coverage = 32
        nside_map = 2**10

        bitpacked_map = HealSparseMap.make_empty(nside_coverage, nside_map, np.bool_, bit_packed=True)
        bool_map = HealSparseMap.make_empty(nside_coverage, nside_map, np.bool_)

        bitpacked_map[10000: 20000] = True
        bool_map[10000: 20000] = True

        bitpacked_map.update_values_pix(np.arange(15000, 25000), True, operation="or")
        bool_map.update_values_pix(np.arange(15000, 25000), True, operation="or")

        testing.assert_array_equal(bitpacked_map.valid_pixels, bool_map.valid_pixels)

        bitpacked_map.update_values_pix(np.arange(15000, 30000), False, operation="or")
        bool_map.update_values_pix(np.arange(15000, 30000), False, operation="or")

        testing.assert_array_equal(bitpacked_map.valid_pixels, bool_map.valid_pixels)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__ == '__main__':
    unittest.main()
