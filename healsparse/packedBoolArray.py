import numpy as np
import numbers

from .utils import is_integer_value


class _PackedBoolArray:
    """Bit-packed map to be used as a healsparse sparse_map.

    Parameters
    ----------
    size : `int`, optional
        Initial number of bits to allocate.
    data_buffer : `np.ndarray` of `np.uint8`, optional
        Data buffer of unsigned integers.  Overrides any other values
        if given.  This will not be a copy, so changes here will
        reflect in the source.
    start_index : `int`, optional
         Index at which the array starts (used for unaligned slices).
    stop_index : `int`, optional
         Index at which the array ends (used for unaligned slices and
         arrays which are not divisible by 8).
    """
    def __init__(self, size=0, data_buffer=None, start_index=None, stop_index=None):
        if data_buffer is not None:
            if not isinstance(data_buffer, np.ndarray) or data_buffer.dtype != np.uint8:
                raise ValueError("data_buffer must be a numpy array of type uint8")

            self._data = data_buffer

            if start_index is None:
                self._start_index = 0
            else:
                if start_index < 0 or start_index > 7:
                    raise ValueError("start_index must be between 0 and 7.")
                self._start_index = start_index

            if stop_index is None:
                self._stop_index = len(data_buffer)*8
            else:
                intrinsic_size = len(data_buffer)*8
                if stop_index < (intrinsic_size - 7) or stop_index > intrinsic_size:
                    raise ValueError("stop_index must be within 8 of the intrinsic size.")
                self._stop_index = stop_index
        else:
            if (size % 8) == 0:
                # This is aligned.
                data_len = size // 8
            else:
                # Unaligned; we need an extra data bin.
                data_len = size // 8 + 1

            self._data = np.zeros(data_len, dtype=np.uint8)

            self._start_index = 0
            self._stop_index = size

        # Reported dtype is numpy bool.
        self._dtype = np.dtype("bool")

        # Set up constants for bit counting
        self._s55 = np.uint8(0x55)
        self._s33 = np.uint8(0x33)
        self._s0F = np.uint8(0x0F)
        self._s01 = np.uint8(0x01)

        self._uint8_truefalse = {
            True: ~np.uint8(0),
            False: np.uint8(0),
        }

    @classmethod
    def from_boolean_array(cls, arr, start_index=None):
        """Create a _PackedBoolArray from a numpy boolean array.

        Parameters
        ----------
        arr : `np.ndarray`
            Numpy array; must be of np.bool_ dtype.
        start_index : `int`, optional
            Index at which the array starts (used for unaligned slices).

        Returns
        -------
        _PackedBoolArray
        """
        if arr.dtype != np.bool_:
            raise NotImplementedError("Can only use from_boolean_array with a boolean array.")

        if start_index is not None:
            data_buffer = np.packbits(
                np.concatenate((np.zeros(start_index, dtype=np.bool_), arr)),
                bitorder="little",
            )
        else:
            start_index = 0
            data_buffer = np.packbits(arr, bitorder="little")

        stop_index = start_index + len(arr)

        return cls(data_buffer=data_buffer, start_index=start_index, stop_index=stop_index)

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return (self.size, )

    @property
    def size(self):
        return self._stop_index - self._start_index

    def resize(self, newsize, refcheck=False):
        """Resize the PackedBoolArray.

        Parameters
        ----------
        newsize : `int`
            New size of the array.
        refcheck : `bool`, optional
            If False, reference count will not be checked.
        """
        if self._start_index != 0:
            raise NotImplementedError("Cannot resize a _PackedBoolArray with an offset start.")

        newsize_data = newsize // 8
        if (newsize % 8) != 0:
            newsize_data += 1

        # FIXME This is not correct...
        self._stop_index = newsize

        self._data.resize(newsize_data, refcheck=refcheck)

    def sum(self, shape=None, axis=None):
        if self._start_index != 0 or self._stop_index != self.size:
            # This will require special casing the first and last elements.
            raise NotImplementedError("Cannot yet do a sum of an offset _PackedBoolArray")

        if shape is None:
            # FIXME about masking first/last.
            return np.sum(self._bit_count(self._data), dtype=np.int64)
        else:
            if not isinstance(shape, (list, tuple)):
                raise ValueError("Shape must be a list or tuple.")
            if axis is not None and axis >= len(shape):
                raise ValueError(f"Axis {axis} is out of bounds for shape.")
            # The shape needs to (a) the last axis must be a multiple
            # of 8; the product needs to equal ths size.
            if np.prod(shape) != self.size:
                raise ValueError("Shape mismatch with array size.")
            if shape[-1] % 8 != 0:
                raise ValueError("Final shape index must be a multiple of 8.")

            new_shape = list(shape)
            new_shape[-1] //= 8
            temp = self._bit_count(self._data)
            return np.sum(temp.reshape(new_shape), axis=axis, dtype=np.int64)

    @property
    def data_array(self):
        if self._start_index != 0 or self._stop_index != self.size:
            raise NotImplementedError("Cannot yet get data array from offset array")

        return self._data

    def copy(self):
        """Return a copy.

        Returns
        -------
        copy : `_PackedBoolArray`
        """
        first_unpacked, mid_data, last_unpacked = self._extract_first_middle_last(mask_extra=True)
        new_buffer = self._data.copy()
        if first_unpacked[0] is not None:
            new_buffer[0] = np.packbits(first_unpacked[0], bitorder="little")
        if last_unpacked[0] is not None:
            new_buffer[-1] = np.packbits(last_unpacked[0], bitorder="little")

        return _PackedBoolArray(
            data_buffer=new_buffer,
            start_index=self._start_index,
            stop_index=self._stop_index,
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"_PackedBoolArray(size={self.size})"

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        # If it's a slice, we return a _PackedBoolArray view.
        if isinstance(key, slice):
            # Record the current start index.
            start_index = self._start_index
            stop_index = self._stop_index

            if key.start is not None:
                if key.start < 0 or key.start > self.size:
                    raise ValueError("Slice start out of range.")

                # We need to know both how to slice the data buffer and
                # recompute the start index relative to the slice.
                data_start = (key.start + self._start_index) // 8
                start_index = (key.start + self._start_index) % 8
                # Stop index also needs to be adjusted.
                stop_index -= data_start*8
            else:
                # Start at the beginning of the data buffer, but
                # use the original start_index.
                data_start = 0

            if key.stop is not None:
                if key.stop < 0:
                    _stop = key.stop + self.size
                else:
                    _stop = key.stop

                if _stop > self.size or _stop < start_index:
                    raise ValueError("Slice stop is out of range.")

                # We need to know how to slice the data buffer and
                # recompute the stop index relative to the slice.
                data_stop = (_stop + self._start_index) // 8 + 1
                stop_index = (data_stop - data_start - 1)*8 + (_stop + self._start_index) % 8
            else:
                # Stop at the end of the data buffer, and use
                # the original stop index (though this may have
                # been modified above if we cut off the data).
                data_stop = len(self._data)

            if key.step is not None:
                if key.step != 1:
                    raise NotImplementedError("Slicing with a step that is not 1 is not supported.")

            return _PackedBoolArray(
                data_buffer=self._data[data_start: data_stop],
                start_index=start_index,
                stop_index=stop_index,
            )

        elif isinstance(key, numbers.Integral):
            return self._test_bits_at_locs(np.atleast_1d(key))[0]
        elif isinstance(key, (np.ndarray, list, tuple)):
            indices = np.atleast_1d(key)
            if not is_integer_value(indices.dtype.type(0)):
                raise IndexError("Numpy array indices must be integers for __getitem__")
            return self._test_bits_at_locs(indices)
        else:
            raise IndexError("Illegal index type (%s) for __getitem__ in _PackedBoolArray." %
                             (key.__class__))

    def __setitem__(self, key, value):
        if isinstance(key, numbers.Integral):
            if value:
                return self._set_bits_at_locs(np.atleast_1d(key))
            else:
                return self._clear_bits_at_locs(np.atleast_1d(key))
        elif isinstance(key, slice):
            # We first compute a "view" slice of the data, which makes
            # our setting math a lot simpler.
            temp_pba = self[key]
            # Extract the parts so that we have an unpacked array at the
            # start and end (if necessary) and an efficient packed array
            # for the bulk of the array.
            # Note that the unpacked arrays are copies, not views, and
            # must be overwritten explicitly with packed data.
            first_unpacked, mid_data, last_unpacked = temp_pba._extract_first_middle_last(mask_extra=False)
            # Check the value.
            if isinstance(value, (bool, np.bool_)):
                if first_unpacked[0] is not None:
                    first_unpacked[0][slice(first_unpacked[1], first_unpacked[2], None)] = value
                    temp_pba._data[0] = np.packbits(first_unpacked[0], bitorder="little")[0]
                if last_unpacked[0] is not None:
                    last_unpacked[0][slice(last_unpacked[1], last_unpacked[2], None)] = value
                    temp_pba._data[-1] = np.packbits(last_unpacked[0], bitorder="little")[0]
                if mid_data is not None:
                    mid_data[:] = self._uint8_truefalse[bool(value)]
            elif isinstance(value, (_PackedBoolArray, np.ndarray)):
                if value.dtype != self._dtype:
                    raise ValueError("Can only set to array of bools")

                if isinstance(value, np.ndarray):
                    # Convert to a packed bool array, aligned with temp_pba.
                    if len(value) != len(temp_pba):
                        raise ValueError("Data length mismatch between slice and value.")
                    _value = _PackedBoolArray.from_boolean_array(
                        value,
                        start_index=temp_pba._start_index,
                    )
                else:
                    if temp_pba._start_index != value._start_index or \
                       temp_pba._stop_index != value._stop_index:
                        raise ValueError("Value _PackedBoolArray must be aligned with slice.")
                    _value = value

                value_first_unpacked, value_mid_data, value_last_unpacked = \
                    _value._extract_first_middle_last(mask_extra=False)

                if first_unpacked[0] is not None:
                    first_unpacked[0][slice(first_unpacked[1], first_unpacked[2], None)] = \
                        value_first_unpacked[0][slice(first_unpacked[1], first_unpacked[2], None)]
                    temp_pba._data[0] = np.packbits(first_unpacked[0], bitorder="little")[0]
                if last_unpacked[0] is not None:
                    last_unpacked[0][slice(last_unpacked[1], last_unpacked[2], None)] = \
                        value_last_unpacked[0][slice(last_unpacked[1], last_unpacked[2], None)]
                    temp_pba._data[-1] = np.packbits(last_unpacked[0], bitorder="little")[0]
                if mid_data is not None:
                    mid_data[:] = value_mid_data[:]
            else:
                raise ValueError("Can only set to bool or array of bools or _PackedBoolArray")

        elif isinstance(key, (np.ndarray, list, tuple)):
            indices = np.atleast_1d(key)
            if not is_integer_value(indices.dtype.type(0)):
                raise IndexError("Numpy array indices must be integers for __setitem__")
            if isinstance(value, (bool, np.bool_)):
                if value:
                    return self._set_bits_at_locs(np.atleast_1d(indices))
                else:
                    return self._clear_bits_at_locs(np.atleast_1d(indices))
            elif isinstance(value, np.ndarray):
                if value.dtype != self._dtype:
                    raise ValueError("Can only set to bool or array of bools")
                if len(value) != len(indices):
                    raise ValueError("Length mismatch")
                self._set_bits_at_locs(indices[value])
                self._clear_bits_at_locs(indices[~value])
            else:
                raise ValueError("Can only set to bool or array of bools")
        else:
            raise IndexError("Illegal index type (%s) for __setitem__ in _PackedBoolArray." %
                             (key.__class__))

    def __array__(self):
        array = np.unpackbits(self._data, bitorder="little").astype(np.bool_)
        return array[self._start_index: self._stop_index]

    def _extract_first_middle_last(self, mask_extra=False):
        """Extract the first/middle/last parts of the data buffer.

        The first and last are "unpacked" boolean arrays, of length
        at most 8. These are copies of the underlying data.
        The middle is a view into the data buffer, and can be
        operated on in bulk.

        Parameters
        ----------
        mask_extra : `bool,` optional
            Set all the extra padding to False when extracting.

        Returns
        -------
        first_unpacked : `tuple` (`np.ndarray`, int, int)
            Tuple with an unpacked array of boolean type, and two
            optional indices which are the "start" and "stop"
            values when re-setting the original array.  The array
            may be None if everything is aligned at the start.
        mid_data : `np.ndarray`
            View of data buffer.
        last_unpacked : `tuple` (`np.ndarray`, int, int)
            Tuple with an unpacked array of boolean type, and
            two optional indices which are the "start" and
            "stop" values when re-setting the original array.
            The array may be None if everything is aligned
            at the end.
        """
        ndata = len(self._data)

        if self._start_index == 0 and self._stop_index == ndata*8:
            # This is a fully aligned buffer.
            return (None, -1, -1), self._data, (None, -1, -1)
        elif self._start_index == 0:
            # This is aligned at 0.
            last_unpacked = np.unpackbits(self._data[-1], bitorder="little").astype(np.bool_)
            if mask_extra:
                last_unpacked[self._stop_index % 8:] = False

            if self._stop_index < 8:
                # This is a short array, less than 8 bits.
                return (None, -1, -1), None, (last_unpacked, None, self._stop_index % 8)
            else:
                # This is a longer array, more than 8 bits.
                return (None, -1, -1), self._data[0: -1], (last_unpacked, None, self._stop_index % 8)
        else:
            # This is not aligned at 0.
            first_unpacked = np.unpackbits(self._data[0], bitorder="little").astype(np.bool_)
            if mask_extra:
                first_unpacked[0: self._start_index] = False

            if self._stop_index == ndata*8:
                # Aligned at the end.
                if ndata == 1:
                    # This is a short array.
                    return (first_unpacked, self._start_index, None), None, (None, -1, -1)
                else:
                    # This is a longer array.
                    return (first_unpacked, self._start_index, None), self._data[1:], (None, -1, -1)
            else:
                # Unaligned at the end.
                if ndata == 1:
                    # This is a very short array that is unaligned at both
                    # the front and the back.
                    if mask_extra:
                        first_unpacked[self._stop_index:] = False

                    return (first_unpacked, self._start_index, self._stop_index), None, (None, -1, -1)
                else:
                    # This is a long array, we need the last unpacked.
                    last_unpacked = np.unpackbits(self._data[-1], bitorder="little").astype(np.bool_)
                    if mask_extra:
                        last_unpacked[self._stop_index % 8:] = False

                    mid_data = self._data[1: -1]
                    if len(mid_data) == 0:
                        mid_data = None
                    return (
                        (first_unpacked, self._start_index, None),
                        mid_data,
                        (last_unpacked, 0, self._stop_index % 8),
                    )

        # Should not get here.
        raise RuntimeError("Programmer mistake")

    def _set_bits_at_locs(self, locs):
        if locs.min() < 0 or locs.max() > self.size:
            raise ValueError("Location indices out of range.")

        _locs = locs + self._start_index

        np.bitwise_or.at(
            self._data,
            _locs // 8,
            1 << (_locs % 8).astype(np.uint8),
        )

    def _clear_bits_at_locs(self, locs):
        if locs.min() < 0 or locs.max() > self.size:
            raise ValueError("Location indices out of range.")

        _locs = locs + self._start_index

        np.bitwise_and.at(
            self._data,
            _locs // 8,
            ~(1 << (_locs % 8).astype(np.uint8)),
        )

    def _test_bits_at_locs(self, locs):
        if locs.min() < 0 or locs.max() > self.size:
            raise ValueError("Location indices out of range.")

        _locs = locs + self._start_index

        return self._data[_locs // 8] & (1 << (_locs % 8).astype(np.uint8)) != 0


class _PackedBoolArray0:
    """Bit-packed map to be used as a healsparse sparse_map.

    Parameters
    ----------
    size : `int`, optional
        Initial number of bits to allocate.  Must be multiple of 8.
    fill_value : `bool`, optional
        Initial fill value.  Only False is fully tested/supported.
    data_buffer : `np.ndarray` of `np.uint8`, optional
        Data buffer of unsigned integers.  Overrides any other values
        if given.  This will not be a copy, so changes here will
        reflect in the source.
    start_index : `int`, optional
        Index at which the array starts (used for unaligned slices).
    end_endex : `int`, optional
        Index at which the array ends (used for unaligned slices and
        arrays which are not divisible by 8).
    """
    def __init__(self, size=0, fill_value=False, data_buffer=None, start_index=None, end_index=None):
        if data_buffer is not None:
            if not isinstance(data_buffer, np.ndarray) or data_buffer.dtype != np.uint8:
                raise ValueError("data_buffer must be a numpy array of type uint8")

            self._data = data_buffer

            if start_index is None:
                self._start_index = 0
            else:
                if start_index < 0 or start_index > 7:
                    raise ValueError("start_index must be between 0 and 7.")
                self._start_index = start_index

            if end_index is None:
                self._end_index = len(data_buffer)*8
            else:
                if end_index < (len(data_buffer) - 7) or end_index > len(data_buffer):
                    raise ValueError("end_index must be within 8 of the intrinsic size.")
                self._end_index = end_index
        else:
            if (size % 8) == 0:
                # This is aligned.
                data_len = size // 8
            else:
                # Unaligned; we need an extra data bin.
                data_len = size // 8 + 1

            # Check if size is multiple of 8.
            if (size % 8) != 0:
                raise ValueError("_PackedBoolArray must have a size that is a multiple of 8.")

            self._data = np.zeros(size // 8, dtype=np.uint8)
            # We need to rething this part.
            if fill_value:
                # Set to all 1s
                self._data[:] = 255

            self._start_index = 0
            self._end_index = size

        # Reported dtype is numpy bool.
        self._dtype = np.dtype("bool")

        # Set up constants for bit counting
        self._s55 = np.uint8(0x55)
        self._s33 = np.uint8(0x33)
        self._s0F = np.uint8(0x0F)
        self._s01 = np.uint8(0x01)

        self._uint8_truefalse = {
            True: ~np.uint8(0),
            False: np.uint8(0),
        }

    @classmethod
    def from_boolean_array(cls, arr):
        """Create a _PackedBoolArray from a numpy boolean array.

        Parameters
        ----------
        arr : `np.ndarray`
            Numpy array; must be of np.bool_ dtype.

        Returns
        -------
        _PackedBoolArray
        """
        if arr.dtype != np.bool_:
            raise NotImplementedError("Can only use from_boolean_array with a boolean array.")

        return cls(data_buffer=np.packbits(arr, bitorder="little"), end_index=len(array))

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return (self.size, )

    @property
    def size(self):
        return self._end_index - self._start_index

    def resize(self, newsize, refcheck=False):
        """Docstring here.

        Parameters
        ----------
        newsize : `int`
            New size of the array.
        refcheck : `bool`, optional
            If False, reference count will not be checked.
        """
        if self._start_index != 0:
            raise NotImplementedError("Cannot resize a _PackedBoolArray with an offset start.")

        newsize_data = newsize // 8
        if (newsize % 8) != 0:
            newsize_data += 1

        self._end_index = newsize

        self._data.resize(newsize_data, refcheck=refcheck)

    def sum(self, shape=None, axis=None):
        if self._start_index != 0 or self._end_index != self.size:
            # This will require special casing the first and last elements.
            raise NotImplementedError("Cannot yet do a sum of an offset _PackedBoolArray")

        if shape is None:
            return np.sum(self._bit_count(self._data), dtype=np.int64)
        else:
            if not isinstance(shape, (list, tuple)):
                raise ValueError("Shape must be a list or tuple.")
            if axis is not None and axis >= len(shape):
                raise ValueError(f"Axis {axis} is out of bounds for shape.")
            # The shape needs to (a) the last axis must be a multiple
            # of 8; the product needs to equal ths size.
            if np.prod(shape) != self.size:
                raise ValueError("Shape mismatch with array size.")
            if shape[-1] % 8 != 0:
                raise ValueError("Final shape index must be a multiple of 8.")

            new_shape = list(shape)
            new_shape[-1] //= 8
            temp = self._bit_count(self._data)
            return np.sum(temp.reshape(new_shape), axis=axis, dtype=np.int64)

    @property
    def data_array(self):
        if self._start_index != 0 or self._end_index != self.size:
            raise NotImplementedError("Cannot yet get data array from offset array")

        return self._data

    def copy(self):
        """Return a copy.

        Returns
        -------
        copy : `_PackedBoolArray`
        """
        if self._slice_start_offset != 0 or self._slice_end_offset != 0:
            raise NotImplementedError("Cannot copy a sliced PackedBoolArray with offsets.")

        new_buffer = self._data.copy()
        if self._start_index > 0:
            first_arr = np.unpackbits(new_buffer[0], bitorder="little").astype(np.bool_)
            first_arr[0: self._start_index] = False
            new_buffer[0] = np.packbits(first_arr, bitorder="little")[0]
        if self._end_index < len(new_buffer) * 8:
            last_arr = np.unpackbits(new_buffer[-1], bitorder="little").astype(np.bool_)
            last_arr[len(new_buffer) * 8 - self._end_index: ] = False
            new_buffer[-1] = np.packbits(last_arr, bitorder="little")[0]

        return _PackedBoolArray(
            data_buffer=new_buffer,
            start_index=self._start_index,
            end_index=self._end_index,
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"_PackedBoolArray(size={self.size})"

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        # If it's a slice, we return a _PackedBoolArray view.
        if isinstance(key, slice):
            # print("getitem", key.start, key.stop)
            start_index = 0
            if key.start is not None:
                data_start = key.start // 8
                if key.start % 8 != 0:
                    start_index = key.start % 8
            else:
                data_start = 0
            if key.stop is not None:
                data_stop = key.stop // 8
                stop_index = key.stop
                if key.stop % 8 != 0:
                    data_stop += 1
            else:
                pass
            slice_start_offset = 0
            slice_end_offset = 0
            if key.start is not None:
                if key.start % 8 != 0:
                    slice_start_offset = key.start % 8
                start = key.start // 8
            else:
                start = 0
            if key.stop is not None:
                if key.stop % 8 != 0:
                    slice_end_offset = 8 - (key.stop % 8)
                stop = (key.stop + slice_end_offset) // 8
            else:
                stop = self._size // 8
            if key.step is not None:
                if key.step % 8 != 0:
                    raise ValueError("Slices of _PackedBoolArray must have a step multiple of 8.")
                step = key.step // 8
            else:
                step = None

            return _PackedBoolArray(
                data_buffer=self._data[slice(start, stop, step)],
                slice_start_offset = slice_start_offset,
                slice_end_offset = slice_end_offset,
            )
        elif isinstance(key, numbers.Integral):
            return self._test_bits_at_locs(key)[0]
        elif isinstance(key, np.ndarray):
            if not is_integer_value(key.dtype.type(0)):
                raise IndexError("Numpy array indices must be integers for __getitem__")
            return self._test_bits_at_locs(key)
        elif isinstance(key, (list, tuple)):
            arr = np.atleast_1d(key)
            if len(arr) > 0:
                if not is_integer_value(arr[0]):
                    raise IndexError("List array indices must be integers for __getitem__")
            return self._test_bits_at_locs(arr)
        else:
            raise IndexError("Illegal index type (%s) for __getitem__ in _PackedBoolArray." %
                             (key.__class__))

    def __setitem__(self, key, value):
        if isinstance(key, numbers.Integral):
            # Need to check that value is single-valued; and bool.
            if value:
                return self._set_bits_at_locs(key)
            else:
                return self._clear_bits_at_locs(key)
        elif isinstance(key, slice):
            print("setitem", key.start, key.stop)
            slice8 = True
            # We need this functionality to implement __iand__ and __ior__
            # with arbitrary ranges.  In the future, functionality
            # can be added to work in all cases.
            if isinstance(value, _PackedBoolArray):
                start_offset = value._slice_start_offset
                end_offset = value._slice_end_offset
            else:
                start_offset = 0
                end_offset = 0
            if key.start is not None:
                if (key.start - start_offset) % 8 != 0:
                    slice8 = False
                else:
                    start = (key.start - start_offset) // 8
            else:
                start = None
            if key.stop is not None:
                if (key.stop + end_offset) % 8 != 0:
                    slice8 = False
                else:
                    stop = (key.stop + end_offset) // 8
            else:
                stop = None
            if key.step is not None:
                if key.step % 8 != 0:
                    slice8 = False
                else:
                    step = key.step // 8
            else:
                step = None

            if slice8:
                # We can do optimized operations here.
                s8 = slice(start, stop, step)

                if isinstance(value, (bool, np.bool_)):
                    if value:
                        # This is all True.
                        self._data[s8] = np.array(-1).astype(np.uint8)
                    else:
                        # This is all False.
                        self._data[s8] = np.uint8(0)
                elif isinstance(value, np.ndarray):
                    if value.dtype != self._dtype:
                        raise ValueError("Can only set to array of bools")
                    if len(range(*s8.indices(len(self._data)))) != value.size // 8:
                        raise ValueError("Length of values does not match slice.")
                    self._data[s8] = np.packbits(value, bitorder="little")
                elif isinstance(value, _PackedBoolArray):
                    self._data[s8] = value._data
                else:
                    raise ValueError("Can only set to bool or array of bools or _PackedBoolArray")
            else:
                # Unoptimized operations
                start = key.start if key.start is not None else 0
                stop = key.stop if key.stop is not None else self.size
                step = key.step if key.step is not None else 1

                indices = np.arange(start, stop, step)

                # Need to check that value is single valued bool or array of bool with
                # the right length.
                if isinstance(value, (bool, np.bool_)):
                    if value:
                        return self._set_bits_at_locs(indices)
                    else:
                        return self._clear_bits_at_locs(indices)

                elif isinstance(value, np.ndarray):
                    if value.dtype != self._dtype:
                        raise ValueError("Can only set to array of bools")
                    if len(value) != len(indices):
                        raise ValueError("Length mismatch")
                    self._set_bits_at_locs(indices[value])
                    self._clear_bits_at_locs(indices[~value])
                else:
                    raise ValueError("Can only set to bool or array of bools")
        elif isinstance(key, np.ndarray):
            if not is_integer_value(key.dtype.type(0)):
                raise IndexError("Numpy array indices must be integers for __setitem__")
            if isinstance(value, (bool, np.bool_)):
                if value:
                    return self._set_bits_at_locs(key)
                else:
                    return self._clear_bits_at_locs(key)
            elif isinstance(value, np.ndarray):
                if value.dtype != self._dtype:
                    raise ValueError("Can only set to bool or array of bools")
                if len(value) != len(key):
                    raise ValueError("Length mismatch")
                self._set_bits_at_locs(key[value])
                self._clear_bits_at_locs(key[~value])
            else:
                raise ValueError("Can only set to bool or array of bools")
        elif isinstance(key, (list, tuple)):
            indices = np.atleast_1d(key)
            if len(indices) > 0:
                if not is_integer_value(indices[0]):
                    raise IndexError("List array indices must be integers for __setitem__")
            if isinstance(value, (bool, np.bool_)):
                if value:
                    return self._set_bits_at_locs(indices)
                else:
                    return self._clear_bits_at_locs(indices)
            elif isinstance(value, np.ndarray):
                if value.dtype != self._dtype:
                    raise ValueError("Can only set to bool or array of bools")
                if len(value) != len(key):
                    raise ValueError("Length mismatch")
                self._set_bits_at_locs(indices[value])
                self._clear_bits_at_locs(indices[~value])
            else:
                raise ValueError("Can only set to bool or array of bools")
        else:
            raise IndexError("Illegal index type (%s) for __setitem__ in _PackedBoolArray." %
                             (key.__class__))

    def __array__(self):
        array = np.unpackbits(self._data, bitorder="little").astype(np.bool_)
        start = self._slice_start_offset
        end = self.size - self._slice_end_offset
        return array[start: end]

    def __and__(self, other):
        if isinstance(other, (bool, np.bool_)):
            return _PackedBoolArray(data_buffer=(self._data & self._uint8_truefalse[other]))
        elif isinstance(other, _PackedBoolArray):
            return _PackedBoolArray(data_buffer=(self._data & other._data))
        else:
            raise NotImplementedError("and function only supports bool and _PackedBoolArray")

    def _sliced_operation_helper(self, operation, other):
        # We need to special case the first and/or last bits.
        full_start = 0
        full_end = len(self._data)

        # Do we need to special case fits within one?
        if self._slice_start_offset > 0:
            first = np.unpackbits(self._data[0], bitorder="little").astype(np.bool_)
            if operation == "and":
                first[self._slice_start_offset: ] &= other
            elif operation == "or":
                first[self._slice_start_offset: ] |= other
            elif operation == "xor":
                first[self._slice_start_offset: ] ^= other
            else:
                raise RuntimeError("Illegal operation for the helper.")
            self._data[0] = np.packbits(first, bitorder="little")[0]
            full_start = 1
        if self._slice_end_offset > 0:
            last = np.unpackbits(self._data[-1], bitorder="little").astype(np.bool_)
            if operation == "and":
                last[: 8 - self._slice_end_offset] &= other
            elif operation == "or":
                last[: 8 - self._slice_end_offset] |= other
            elif operation == "xor":
                last[: 8 - self._slice_end_offset] ^= other
            else:
                raise RuntimeError("Illegal operation for the helper.")
            self._data[-1] = np.packbits(last, bitorder="little")[0]
            full_end -= 1

        print(full_start, full_end)

        if operation == "and":
            self._data[full_start: full_end] &= self._uint8_truefalse[other]
        elif operation == "or":
            self._data[full_start: full_end] |= self._uint8_truefalse[other]
        elif operation == "xor":
            self._data[full_start: full_end] ^= self._uint8_truefalse[other]
        else:
            raise RuntimeError("Illegal operation for the helper")

    def __iand__(self, other):
        if isinstance(other, (bool, np.bool_)):
            if self._slice_start_offset > 0 or self._slice_end_offset > 0:
                self._sliced_operation_helper("and", other)
            else:
                # All aligned.
                self._data &= self._uint8_truefalse[other]
            return self
        elif isinstance(other, _PackedBoolArray):
            self._data &= other._data
            return self
        else:
            raise NotImplementedError("iand function only supports bool and _PackedBoolArray")

    def __or__(self, other):
        if isinstance(other, (bool, np.bool_)):
            return _PackedBoolArray(data_buffer=(self._data | self._uint8_truefalse[other]))
        elif isinstance(other, _PackedBoolArray):
            return _PackedBoolArray(data_buffer=(self._data | other._data))
        else:
            raise NotImplementedError("or function only supports bool and _PackedBoolArray")

    def __ior__(self, other):
        if isinstance(other, (bool, np.bool_)):
            if self._slice_start_offset > 0 or self._slice_end_offset > 0:
                print("Unaligned ior")
                self._sliced_operation_helper("or", other)
            else:
                # All aligned.
                print("aligned ior")
                self._data |= self._uint8_truefalse[other]
            return self
        elif isinstance(other, _PackedBoolArray):
            self._data |= other._data
            return self
        else:
            raise NotImplementedError("ior function only supports bool and _PackedBoolArray")

    def __xor__(self, other):
        if isinstance(other, (bool, np.bool_)):
            return _PackedBoolArray(data_buffer=(self._data ^ self._uint8_truefalse[other]))
        elif isinstance(other, _PackedBoolArray):
            return _PackedBoolArray(data_buffer=(self._data ^ other._data))
        else:
            raise NotImplementedError("xor function only supports bool and _PackedBoolArray")

    def __ixor__(self, other):
        if isinstance(other, (bool, np.bool_)):
            if self._slice_start_offset > 0 or self._slice_end_offset > 0:
                self._sliced_operation_helper("xor", other)
            else:
                # All aligned.
                self._data ^= self._uint8_truefalse[other]
            return self
        elif isinstance(other, _PackedBoolArray):
            self._data ^= other._data
            return self
        else:
            raise NotImplementedError("ixor function only supports bool and _PackedBoolArray")

    def __invert__(self):
        return _PackedBoolArray(data_buffer=~self._data)

    def invert(self):
        self._data = ~self._data

        return self

    def _set_bits_at_locs(self, locs):
        _locs = np.atleast_1d(locs)

        np.bitwise_or.at(
            self._data,
            _locs // 8,
            1 << (_locs % 8).astype(np.uint8),
        )

    def _clear_bits_at_locs(self, locs):
        # FIXME change this?  Require checked above?  Yes
        # But this can check bounds?
        _locs = np.atleast_1d(locs)

        np.bitwise_and.at(
            self._data,
            _locs // 8,
            ~(1 << (_locs % 8).astype(np.uint8)),
        )

    def _test_bits_at_locs(self, locs):
        # FIXME change this?  Require checked above?  Yes
        # But this can check bounds?
        _locs = np.atleast_1d(locs)

        return self._data[_locs // 8] & (1 << (_locs % 8).astype(np.uint8)) != 0


class _PackedBoolArray0:
    """Bit-packed map to be used as a healsparse sparse_map.

    Parameters
    ----------
    size : `int`, optional
        Initial number of bits to allocate.  Must be multiple of 8.
    fill_value : `bool`, optional
        Initial fill value.  Only False is fully tested/supported.
    data_buffer : `np.ndarray` of `np.uint8`, optional
        Data buffer of unsigned integers.  Overrides any other values
        if given.  This will not be a copy, so changes here will
        reflect in the source.
    start_index : `int`, optional
        Index at which the array starts (used for unaligned slices).
    end_endex : `int`, optional
        Index at which the array ends (used for unaligned slices and
        arrays which are not divisible by 8).
    """
    def __init__(self, size=0, fill_value=False, data_buffer=None, start_index=None, end_index=None):
        if data_buffer is not None:
            if not isinstance(data_buffer, np.ndarray) or data_buffer.dtype != np.uint8:
                raise ValueError("data_buffer must be a numpy array of type uint8")

            self._data = data_buffer

            if start_index is None:
                self._start_index = 0
            else:
                if start_index < 0 or start_index > 7:
                    raise ValueError("start_index must be between 0 and 7.")
                self._start_index = start_index

            if end_index is None:
                self._end_index = len(data_buffer)*8
            else:
                if end_index < (len(data_buffer) - 7) or end_index > len(data_buffer):
                    raise ValueError("end_index must be within 8 of the intrinsic size.")
                self._end_index = end_index
        else:
            if (size % 8) == 0:
                # This is aligned.
                data_len = size // 8
            else:
                # Unaligned; we need an extra data bin.
                data_len = size // 8 + 1

            # Check if size is multiple of 8.
            if (size % 8) != 0:
                raise ValueError("_PackedBoolArray must have a size that is a multiple of 8.")

            self._data = np.zeros(size // 8, dtype=np.uint8)
            # We need to rething this part.
            if fill_value:
                # Set to all 1s
                self._data[:] = 255

            self._start_index = 0
            self._end_index = size

        # Reported dtype is numpy bool.
        self._dtype = np.dtype("bool")

        # Set up constants for bit counting
        self._s55 = np.uint8(0x55)
        self._s33 = np.uint8(0x33)
        self._s0F = np.uint8(0x0F)
        self._s01 = np.uint8(0x01)

        self._uint8_truefalse = {
            True: ~np.uint8(0),
            False: np.uint8(0),
        }

    @classmethod
    def from_boolean_array(cls, arr):
        """Create a _PackedBoolArray from a numpy boolean array.

        Parameters
        ----------
        arr : `np.ndarray`
            Numpy array; must be of np.bool_ dtype.

        Returns
        -------
        _PackedBoolArray
        """
        if arr.dtype != np.bool_:
            raise NotImplementedError("Can only use from_boolean_array with a boolean array.")

        return cls(data_buffer=np.packbits(arr, bitorder="little"), end_index=len(array))

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return (self.size, )

    @property
    def size(self):
        return self._end_index - self._start_index

    def resize(self, newsize, refcheck=False):
        """Docstring here.

        Parameters
        ----------
        newsize : `int`
            New size of the array.
        refcheck : `bool`, optional
            If False, reference count will not be checked.
        """
        if self._start_index != 0:
            raise NotImplementedError("Cannot resize a _PackedBoolArray with an offset start.")

        newsize_data = newsize // 8
        if (newsize % 8) != 0:
            newsize_data += 1

        self._end_index = newsize

        self._data.resize(newsize_data, refcheck=refcheck)

    def sum(self, shape=None, axis=None):
        if self._start_index != 0 or self._end_index != self.size:
            # This will require special casing the first and last elements.
            raise NotImplementedError("Cannot yet do a sum of an offset _PackedBoolArray")

        if shape is None:
            return np.sum(self._bit_count(self._data), dtype=np.int64)
        else:
            if not isinstance(shape, (list, tuple)):
                raise ValueError("Shape must be a list or tuple.")
            if axis is not None and axis >= len(shape):
                raise ValueError(f"Axis {axis} is out of bounds for shape.")
            # The shape needs to (a) the last axis must be a multiple
            # of 8; the product needs to equal ths size.
            if np.prod(shape) != self.size:
                raise ValueError("Shape mismatch with array size.")
            if shape[-1] % 8 != 0:
                raise ValueError("Final shape index must be a multiple of 8.")

            new_shape = list(shape)
            new_shape[-1] //= 8
            temp = self._bit_count(self._data)
            return np.sum(temp.reshape(new_shape), axis=axis, dtype=np.int64)

    @property
    def data_array(self):
        if self._start_index != 0 or self._end_index != self.size:
            raise NotImplementedError("Cannot yet get data array from offset array")

        return self._data

    def copy(self):
        """Return a copy.

        Returns
        -------
        copy : `_PackedBoolArray`
        """
        if self._slice_start_offset != 0 or self._slice_end_offset != 0:
            raise NotImplementedError("Cannot copy a sliced PackedBoolArray with offsets.")

        new_buffer = self._data.copy()
        if self._start_index > 0:
            first_arr = np.unpackbits(new_buffer[0], bitorder="little").astype(np.bool_)
            first_arr[0: self._start_index] = False
            new_buffer[0] = np.packbits(first_arr, bitorder="little")[0]
        if self._end_index < len(new_buffer) * 8:
            last_arr = np.unpackbits(new_buffer[-1], bitorder="little").astype(np.bool_)
            last_arr[len(new_buffer) * 8 - self._end_index: ] = False
            new_buffer[-1] = np.packbits(last_arr, bitorder="little")[0]

        return _PackedBoolArray(
            data_buffer=new_buffer,
            start_index=self._start_index,
            end_index=self._end_index,
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"_PackedBoolArray(size={self.size})"

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        # If it's a slice, we return a _PackedBoolArray view.
        if isinstance(key, slice):
            # print("getitem", key.start, key.stop)
            start_index = 0
            if key.start is not None:
                data_start = key.start // 8
                if key.start % 8 != 0:
                    start_index = key.start % 8
            else:
                data_start = 0
            if key.stop is not None:
                data_stop = key.stop // 8
                stop_index = key.stop
                if key.stop % 8 != 0:
                    data_stop += 1
            else:
                pass
            slice_start_offset = 0
            slice_end_offset = 0
            if key.start is not None:
                if key.start % 8 != 0:
                    slice_start_offset = key.start % 8
                start = key.start // 8
            else:
                start = 0
            if key.stop is not None:
                if key.stop % 8 != 0:
                    slice_end_offset = 8 - (key.stop % 8)
                stop = (key.stop + slice_end_offset) // 8
            else:
                stop = self._size // 8
            if key.step is not None:
                if key.step % 8 != 0:
                    raise ValueError("Slices of _PackedBoolArray must have a step multiple of 8.")
                step = key.step // 8
            else:
                step = None

            return _PackedBoolArray(
                data_buffer=self._data[slice(start, stop, step)],
                slice_start_offset = slice_start_offset,
                slice_end_offset = slice_end_offset,
            )
        elif isinstance(key, numbers.Integral):
            return self._test_bits_at_locs(key)[0]
        elif isinstance(key, np.ndarray):
            if not is_integer_value(key.dtype.type(0)):
                raise IndexError("Numpy array indices must be integers for __getitem__")
            return self._test_bits_at_locs(key)
        elif isinstance(key, (list, tuple)):
            arr = np.atleast_1d(key)
            if len(arr) > 0:
                if not is_integer_value(arr[0]):
                    raise IndexError("List array indices must be integers for __getitem__")
            return self._test_bits_at_locs(arr)
        else:
            raise IndexError("Illegal index type (%s) for __getitem__ in _PackedBoolArray." %
                             (key.__class__))

    def __setitem__(self, key, value):
        if isinstance(key, numbers.Integral):
            # Need to check that value is single-valued; and bool.
            if value:
                return self._set_bits_at_locs(key)
            else:
                return self._clear_bits_at_locs(key)
        elif isinstance(key, slice):
            print("setitem", key.start, key.stop)
            slice8 = True
            # We need this functionality to implement __iand__ and __ior__
            # with arbitrary ranges.  In the future, functionality
            # can be added to work in all cases.
            if isinstance(value, _PackedBoolArray):
                start_offset = value._slice_start_offset
                end_offset = value._slice_end_offset
            else:
                start_offset = 0
                end_offset = 0
            if key.start is not None:
                if (key.start - start_offset) % 8 != 0:
                    slice8 = False
                else:
                    start = (key.start - start_offset) // 8
            else:
                start = None
            if key.stop is not None:
                if (key.stop + end_offset) % 8 != 0:
                    slice8 = False
                else:
                    stop = (key.stop + end_offset) // 8
            else:
                stop = None
            if key.step is not None:
                if key.step % 8 != 0:
                    slice8 = False
                else:
                    step = key.step // 8
            else:
                step = None

            if slice8:
                # We can do optimized operations here.
                s8 = slice(start, stop, step)

                if isinstance(value, (bool, np.bool_)):
                    if value:
                        # This is all True.
                        self._data[s8] = np.array(-1).astype(np.uint8)
                    else:
                        # This is all False.
                        self._data[s8] = np.uint8(0)
                elif isinstance(value, np.ndarray):
                    if value.dtype != self._dtype:
                        raise ValueError("Can only set to array of bools")
                    if len(range(*s8.indices(len(self._data)))) != value.size // 8:
                        raise ValueError("Length of values does not match slice.")
                    self._data[s8] = np.packbits(value, bitorder="little")
                elif isinstance(value, _PackedBoolArray):
                    self._data[s8] = value._data
                else:
                    raise ValueError("Can only set to bool or array of bools or _PackedBoolArray")
            else:
                # Unoptimized operations
                start = key.start if key.start is not None else 0
                stop = key.stop if key.stop is not None else self.size
                step = key.step if key.step is not None else 1

                indices = np.arange(start, stop, step)

                # Need to check that value is single valued bool or array of bool with
                # the right length.
                if isinstance(value, (bool, np.bool_)):
                    if value:
                        return self._set_bits_at_locs(indices)
                    else:
                        return self._clear_bits_at_locs(indices)

                elif isinstance(value, np.ndarray):
                    if value.dtype != self._dtype:
                        raise ValueError("Can only set to array of bools")
                    if len(value) != len(indices):
                        raise ValueError("Length mismatch")
                    self._set_bits_at_locs(indices[value])
                    self._clear_bits_at_locs(indices[~value])
                else:
                    raise ValueError("Can only set to bool or array of bools")
        elif isinstance(key, np.ndarray):
            if not is_integer_value(key.dtype.type(0)):
                raise IndexError("Numpy array indices must be integers for __setitem__")
            if isinstance(value, (bool, np.bool_)):
                if value:
                    return self._set_bits_at_locs(key)
                else:
                    return self._clear_bits_at_locs(key)
            elif isinstance(value, np.ndarray):
                if value.dtype != self._dtype:
                    raise ValueError("Can only set to bool or array of bools")
                if len(value) != len(key):
                    raise ValueError("Length mismatch")
                self._set_bits_at_locs(key[value])
                self._clear_bits_at_locs(key[~value])
            else:
                raise ValueError("Can only set to bool or array of bools")
        elif isinstance(key, (list, tuple)):
            indices = np.atleast_1d(key)
            if len(indices) > 0:
                if not is_integer_value(indices[0]):
                    raise IndexError("List array indices must be integers for __setitem__")
            if isinstance(value, (bool, np.bool_)):
                if value:
                    return self._set_bits_at_locs(indices)
                else:
                    return self._clear_bits_at_locs(indices)
            elif isinstance(value, np.ndarray):
                if value.dtype != self._dtype:
                    raise ValueError("Can only set to bool or array of bools")
                if len(value) != len(key):
                    raise ValueError("Length mismatch")
                self._set_bits_at_locs(indices[value])
                self._clear_bits_at_locs(indices[~value])
            else:
                raise ValueError("Can only set to bool or array of bools")
        else:
            raise IndexError("Illegal index type (%s) for __setitem__ in _PackedBoolArray." %
                             (key.__class__))

    def __array__(self):
        array = np.unpackbits(self._data, bitorder="little").astype(np.bool_)
        start = self._slice_start_offset
        end = self.size - self._slice_end_offset
        return array[start: end]

    def __and__(self, other):
        if isinstance(other, (bool, np.bool_)):
            return _PackedBoolArray(data_buffer=(self._data & self._uint8_truefalse[other]))
        elif isinstance(other, _PackedBoolArray):
            return _PackedBoolArray(data_buffer=(self._data & other._data))
        else:
            raise NotImplementedError("and function only supports bool and _PackedBoolArray")

    def _sliced_operation_helper(self, operation, other):
        # We need to special case the first and/or last bits.
        full_start = 0
        full_end = len(self._data)

        # Do we need to special case fits within one?
        if self._slice_start_offset > 0:
            first = np.unpackbits(self._data[0], bitorder="little").astype(np.bool_)
            if operation == "and":
                first[self._slice_start_offset: ] &= other
            elif operation == "or":
                first[self._slice_start_offset: ] |= other
            elif operation == "xor":
                first[self._slice_start_offset: ] ^= other
            else:
                raise RuntimeError("Illegal operation for the helper.")
            self._data[0] = np.packbits(first, bitorder="little")[0]
            full_start = 1
        if self._slice_end_offset > 0:
            last = np.unpackbits(self._data[-1], bitorder="little").astype(np.bool_)
            if operation == "and":
                last[: 8 - self._slice_end_offset] &= other
            elif operation == "or":
                last[: 8 - self._slice_end_offset] |= other
            elif operation == "xor":
                last[: 8 - self._slice_end_offset] ^= other
            else:
                raise RuntimeError("Illegal operation for the helper.")
            self._data[-1] = np.packbits(last, bitorder="little")[0]
            full_end -= 1

        print(full_start, full_end)

        if operation == "and":
            self._data[full_start: full_end] &= self._uint8_truefalse[other]
        elif operation == "or":
            self._data[full_start: full_end] |= self._uint8_truefalse[other]
        elif operation == "xor":
            self._data[full_start: full_end] ^= self._uint8_truefalse[other]
        else:
            raise RuntimeError("Illegal operation for the helper")

    def __iand__(self, other):
        if isinstance(other, (bool, np.bool_)):
            if self._slice_start_offset > 0 or self._slice_end_offset > 0:
                self._sliced_operation_helper("and", other)
            else:
                # All aligned.
                self._data &= self._uint8_truefalse[other]
            return self
        elif isinstance(other, _PackedBoolArray):
            self._data &= other._data
            return self
        else:
            raise NotImplementedError("iand function only supports bool and _PackedBoolArray")

    def __or__(self, other):
        if isinstance(other, (bool, np.bool_)):
            return _PackedBoolArray(data_buffer=(self._data | self._uint8_truefalse[other]))
        elif isinstance(other, _PackedBoolArray):
            return _PackedBoolArray(data_buffer=(self._data | other._data))
        else:
            raise NotImplementedError("or function only supports bool and _PackedBoolArray")

    def __ior__(self, other):
        if isinstance(other, (bool, np.bool_)):
            if self._slice_start_offset > 0 or self._slice_end_offset > 0:
                print("Unaligned ior")
                self._sliced_operation_helper("or", other)
            else:
                # All aligned.
                print("aligned ior")
                self._data |= self._uint8_truefalse[other]
            return self
        elif isinstance(other, _PackedBoolArray):
            self._data |= other._data
            return self
        else:
            raise NotImplementedError("ior function only supports bool and _PackedBoolArray")

    def __xor__(self, other):
        if isinstance(other, (bool, np.bool_)):
            return _PackedBoolArray(data_buffer=(self._data ^ self._uint8_truefalse[other]))
        elif isinstance(other, _PackedBoolArray):
            return _PackedBoolArray(data_buffer=(self._data ^ other._data))
        else:
            raise NotImplementedError("xor function only supports bool and _PackedBoolArray")

    def __ixor__(self, other):
        if isinstance(other, (bool, np.bool_)):
            if self._slice_start_offset > 0 or self._slice_end_offset > 0:
                self._sliced_operation_helper("xor", other)
            else:
                # All aligned.
                self._data ^= self._uint8_truefalse[other]
            return self
        elif isinstance(other, _PackedBoolArray):
            self._data ^= other._data
            return self
        else:
            raise NotImplementedError("ixor function only supports bool and _PackedBoolArray")

    def __invert__(self):
        return _PackedBoolArray(data_buffer=~self._data)

    def invert(self):
        self._data = ~self._data

        return self

    def _set_bits_at_locs(self, locs):
        # FIXME change this?  Require checked above?  Yes
        # But this can check bounds?
        _locs = np.atleast_1d(locs)

        np.bitwise_or.at(
            self._data,
            _locs // 8,
            1 << (_locs % 8).astype(np.uint8),
        )

    def _clear_bits_at_locs(self, locs):
        # FIXME change this?  Require checked above?  Yes
        # But this can check bounds?
        _locs = np.atleast_1d(locs)

        np.bitwise_and.at(
            self._data,
            _locs // 8,
            ~(1 << (_locs % 8).astype(np.uint8)),
        )

    def _test_bits_at_locs(self, locs):
        # FIXME change this?  Require checked above?  Yes
        # But this can check bounds?
        _locs = np.atleast_1d(locs)

        return self._data[_locs // 8] & (1 << (_locs % 8).astype(np.uint8)) != 0

    def _bit_count(self, arr):
        # Deal with limits?
        arr = arr - ((arr >> 1) & self._s55)
        arr = (arr & self._s33) + ((arr >> 2) & self._s33)
        arr = (arr + (arr >> 4)) & self._s0F
        return arr * self._s01
