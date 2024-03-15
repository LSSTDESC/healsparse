import numpy as np
import numbers

from .utils import is_integer_value


class _PackedBoolArray:
    """Bit-packed map to be used as a healsparse sparse_map.

    Either size or data_buffer may be specified but not both.
    The start_index may be specified with size or with data_buffer.
    The stop_index may only be specified with data_buffer.

    Parameters
    ----------
    size : `int`, optional
        Initial number of bits to allocate.
    data_buffer : `np.ndarray` of `np.uint8`, optional
        Data buffer of unsigned integers.  Overrides any other values
        if given.  This will not be a copy, so changes here will
        reflect in the source.
    start_index : `int`, optional
        Index at which the array starts (used for unaligned slices/arrays).
    stop_index : `int`, optional
        Exclusive index for the end of the array (following Python
        conventions where start is inclusive and stop is exclusive
        ([start: stop]). Used for unaligned slices and arrays with
        length not divisible by 8.
    """
    def __init__(self, size=None, data_buffer=None, start_index=None, stop_index=None):
        if size is not None and data_buffer is not None:
            raise ValueError("May only specify one of size or data_buffer.")

        if start_index is None:
            self._start_index = 0
        else:
            if start_index < 0 or start_index > 7:
                raise ValueError("start_index must be between 0 and 7.")
            self._start_index = start_index

        if data_buffer is not None:
            if not isinstance(data_buffer, np.ndarray) or data_buffer.dtype != np.uint8:
                raise ValueError("data_buffer must be a numpy array of type uint8")

            self._data = data_buffer

            if stop_index is None:
                self._stop_index = len(data_buffer)*8
            else:
                intrinsic_size = len(data_buffer)*8
                if stop_index < (intrinsic_size - 7) or stop_index > intrinsic_size:
                    raise ValueError("stop_index must be within 8 of the intrinsic size.")
                self._stop_index = stop_index
        else:
            if size is None:
                size = 0

            if stop_index is not None:
                raise ValueError("stop_index may not be specified without data_buffer.")

            if ((size + self._start_index) % 8) == 0:
                # This is aligned.
                offset_value = 0
            else:
                # Unaligned; we need an extra data bin.
                offset_value = 1
            data_len = (size + self._start_index) // 8 + offset_value

            self._data = np.zeros(data_len, dtype=np.uint8)

            self._stop_index = size + self._start_index

        # Reported dtype is numpy bool.
        self._dtype = np.dtype("bool")

        # We will need a lookup table for bit counting.
        self.LUT = None

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
        if not isinstance(arr, np.ndarray) or arr.dtype != np.bool_:
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

    @property
    def start_index(self):
        return self._start_index

    def resize(self, newsize, refcheck=False):
        """Resize the PackedBoolArray.

        Parameters
        ----------
        newsize : `int`
            New size of the array.
        refcheck : `bool`, optional
            If False, reference count will not be checked.
        """
        if newsize < self.size:
            raise ValueError("resize can only be used to enlarge the data buffer.")
        elif newsize == self.size:
            # Nothing to do.
            return

        newsize_data = (newsize + self._start_index) // 8
        if (newsize + self._start_index) % 8 != 0:
            newsize_data += 1

        self._stop_index = newsize + self._start_index

        self._data.resize(newsize_data, refcheck=refcheck)

    def sum(self, shape=None, axis=None):
        if shape is None:
            summand = np.int64(0)
            # This is a straight sum, can be done on any alignment.
            first_unpacked, mid_data, last_unpacked = self._extract_first_middle_last(mask_extra=True)
            if first_unpacked[0] is not None:
                summand += np.sum(first_unpacked[0], dtype=np.int64)
            if last_unpacked[0] is not None:
                summand += np.sum(last_unpacked[0], dtype=np.int64)
            if mid_data is not None:
                summand += np.sum(self._bit_count(mid_data), dtype=np.int64)

            return summand
        else:
            if self._start_index != 0 or (self._stop_index % 8) != 0:
                raise ValueError("Reshaped summation can only be done on aligned _PackedBoolArrays.")
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
            if axis == 0:
                raise NotImplementedError("axis=0 is not supported for summation.")

            new_shape = list(shape)
            new_shape[-1] //= 8
            temp = self._bit_count(self._data)
            return np.sum(temp.reshape(new_shape), axis=axis, dtype=np.int64)

    @property
    def data_array(self):
        if self._start_index != 0 or self._stop_index != self.size:
            raise NotImplementedError("_PackedBoolArray does not support extracting the "
                                      "data_array from an unaligned _PackedBoolArray.")

        return self._data

    def copy(self):
        """Return a copy.

        Returns
        -------
        copy : `_PackedBoolArray`
        """
        # Prior to copying the array, we mask any extra padding bits
        # on either side of the data array. This requires unpacking
        # the first and/or last parts of the data buffer and then
        # repackaging the masked tails.
        first_unpacked, mid_data, last_unpacked = self._extract_first_middle_last(mask_extra=True)
        new_buffer = self._data.copy()
        if first_unpacked[0] is not None:
            new_buffer[0] = np.packbits(first_unpacked[0], bitorder="little")[0]
        if last_unpacked[0] is not None:
            new_buffer[-1] = np.packbits(last_unpacked[0], bitorder="little")[0]

        return _PackedBoolArray(
            data_buffer=new_buffer,
            start_index=self._start_index,
            stop_index=self._stop_index,
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        st = f"_PackedBoolArray(size={self.size}"
        if self._start_index > 0:
            st += f", start_index={self._start_index}"
        return st + ")"

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        # If it's a slice, we return a _PackedBoolArray view.
        if isinstance(key, slice):
            # Record the current start index.
            start_index = self._start_index
            stop_index = self._stop_index
            key_start = 0

            if key.start is not None:
                if key.start < 0 or key.start > self.size:
                    raise ValueError("Slice start out of range.")
                key_start = key.start

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
                size = _stop - key_start
                if ((size + start_index) % 8) == 0:
                    # This is aligned.
                    offset_value = 0
                else:
                    offset_value = 1

                data_stop = (_stop + self._start_index) // 8 + offset_value
                stop_index = (data_stop - data_start - offset_value)*8 + (_stop + self._start_index) % 8
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

            if len(temp_pba) == 0:
                return

            # Extract the parts so that we have an unpacked array at the
            # start and end (if necessary) and an efficient packed array
            # for the bulk of the array.
            # Note that the unpacked arrays are copies, not views, and
            # must be overwritten explicitly with packed data.
            first_unpacked, mid_data, last_unpacked = temp_pba._extract_first_middle_last(mask_extra=False)
            # Check the value.
            if isinstance(value, (bool, np.bool_)):
                if first_unpacked[0] is not None:
                    first_unpacked[0][first_unpacked[1]: first_unpacked[2]] = value
                    temp_pba._data[0] = np.packbits(first_unpacked[0], bitorder="little")[0]
                if last_unpacked[0] is not None:
                    last_unpacked[0][last_unpacked[1]: last_unpacked[2]] = value
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
                    first_unpacked[0][first_unpacked[1]: first_unpacked[2]] = \
                        value_first_unpacked[0][first_unpacked[1]: first_unpacked[2]]
                    temp_pba._data[0] = np.packbits(first_unpacked[0], bitorder="little")[0]
                if last_unpacked[0] is not None:
                    last_unpacked[0][last_unpacked[1]: last_unpacked[2]] = \
                        value_last_unpacked[0][last_unpacked[1]: last_unpacked[2]]
                    temp_pba._data[-1] = np.packbits(last_unpacked[0], bitorder="little")[0]
                if mid_data is not None:
                    mid_data[:] = value_mid_data[:]
            else:
                raise ValueError("Can only set to bool or array of bools or _PackedBoolArray")

        elif isinstance(key, (np.ndarray, list, tuple)):
            indices = np.atleast_1d(key)
            if not is_integer_value(indices.dtype.type(0)):
                raise IndexError("Numpy array indices must be integers for __setitem__")
            if len(indices) == 0:
                return
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

    def __iand__(self, other):
        if isinstance(other, (bool, np.bool_)):
            self._operation_helper_bool("and", other)
        elif isinstance(other, _PackedBoolArray):
            if len(other) != len(self):
                raise ValueError("RHS _PackedBoolArray must have the same size.")
            if other._start_index != self._start_index or other._stop_index != other._stop_index:
                raise ValueError("RHS _PackedBoolArray must have matched alignment.")
            self._operation_helper_pba("and", other)
        else:
            raise NotImplementedError("and function only supports bool and _PackedBoolArray")
        return self

    def __and__(self, other):
        new = self.copy()
        new &= other
        return new

    def __ior__(self, other):
        if isinstance(other, (bool, np.bool_)):
            self._operation_helper_bool("or", other)
        elif isinstance(other, _PackedBoolArray):
            if len(other) != len(self):
                raise ValueError("RHS _PackedBoolArray must have the same size.")
            if other._start_index != self._start_index or other._stop_index != other._stop_index:
                raise ValueError("RHS _PackedBoolArray must have matched alignment.")
            self._operation_helper_pba("or", other)
        else:
            raise NotImplementedError("or function only supports bool and _PackedBoolArray")
        return self

    def __or__(self, other):
        new = self.copy()
        new |= other
        return new

    def __ixor__(self, other):
        if isinstance(other, (bool, np.bool_)):
            self._operation_helper_bool("xor", other)
        elif isinstance(other, _PackedBoolArray):
            if len(other) != len(self):
                raise ValueError("RHS _PackedBoolArray must have the same size.")
            if other._start_index != self._start_index or other._stop_index != other._stop_index:
                raise ValueError("RHS _PackedBoolArray must have matched alignment.")
            self._operation_helper_pba("xor", other)
        else:
            raise NotImplementedError("xor function only supports bool and _PackedBoolArray")
        return self

    def __xor__(self, other):
        new = self.copy()
        new ^= other
        return new

    def __invert__(self):
        new = self.copy()
        new.invert()
        return new

    def invert(self):
        self._operation_helper_bool("invert", True)

        return self

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

    def _operation_helper_bool(self, operation, other):
        """Helper function for performing an operation with a boolean.

        Operates on self._data in-place.

        Parameters
        ----------
        operation : `str`
            Must be one of ``and``, ``or``, ``xor``, or ``invert``.
        other : `bool`
            Boolean value.
        """
        first_unpacked, mid_data, last_unpacked = self._extract_first_middle_last(mask_extra=False)

        if first_unpacked[0] is not None:
            if operation == "and":
                first_unpacked[0][first_unpacked[1]: first_unpacked[2]] &= other
            elif operation == "or":
                first_unpacked[0][first_unpacked[1]: first_unpacked[2]] |= other
            elif operation == "xor":
                first_unpacked[0][first_unpacked[1]: first_unpacked[2]] ^= other
            elif operation == "invert":
                first_unpacked[0][first_unpacked[1]: first_unpacked[2]] = \
                    ~first_unpacked[0][first_unpacked[1]: first_unpacked[2]]
            self._data[0] = np.packbits(first_unpacked[0], bitorder="little")[0]
        if last_unpacked[0] is not None:
            if operation == "and":
                last_unpacked[0][last_unpacked[1]: last_unpacked[2]] &= other
            elif operation == "or":
                last_unpacked[0][last_unpacked[1]: last_unpacked[2]] |= other
            elif operation == "xor":
                last_unpacked[0][last_unpacked[1]: last_unpacked[2]] ^= other
            elif operation == "invert":
                last_unpacked[0][last_unpacked[1]: last_unpacked[2]] = \
                    ~last_unpacked[0][last_unpacked[1]: last_unpacked[2]]
            self._data[-1] = np.packbits(last_unpacked[0], bitorder="little")[0]
        if mid_data is not None:
            if operation == "and":
                mid_data[:] &= self._uint8_truefalse[bool(other)]
            elif operation == "or":
                mid_data[:] |= self._uint8_truefalse[bool(other)]
            elif operation == "xor":
                mid_data[:] ^= self._uint8_truefalse[bool(other)]
            elif operation == "invert":
                mid_data[:] = ~mid_data[:]

    def _operation_helper_pba(self, operation, other):
        """Helper function for performing an operation with a _PackedBoolArray.

        Operates on self._data in-place.

        Parameters
        ----------
        operation : `str`
            Must be one of ``and``, ``or``, or ``xor``.
        other : `_PackedBoolArray`
            _PackedBoolArray; must be aligned.
        """
        first_unpacked, mid_data, last_unpacked = self._extract_first_middle_last(mask_extra=False)
        o_first_unpacked, o_mid_data, o_last_unpacked = other._extract_first_middle_last(mask_extra=False)

        if first_unpacked[0] is not None:
            if operation == "and":
                first_unpacked[0][first_unpacked[1]: first_unpacked[2]] &= \
                    o_first_unpacked[0][first_unpacked[1]: first_unpacked[2]]
            elif operation == "or":
                first_unpacked[0][first_unpacked[1]: first_unpacked[2]] |= \
                    o_first_unpacked[0][first_unpacked[1]: first_unpacked[2]]
            elif operation == "xor":
                first_unpacked[0][first_unpacked[1]: first_unpacked[2]] ^= \
                    o_first_unpacked[0][first_unpacked[1]: first_unpacked[2]]
            self._data[0] = np.packbits(first_unpacked[0], bitorder="little")[0]
        if last_unpacked[0] is not None:
            if operation == "and":
                last_unpacked[0][last_unpacked[1]: last_unpacked[2]] &= \
                    o_last_unpacked[0][last_unpacked[1]: last_unpacked[2]]
            elif operation == "or":
                last_unpacked[0][last_unpacked[1]: last_unpacked[2]] |= \
                    o_last_unpacked[0][last_unpacked[1]: last_unpacked[2]]
            elif operation == "xor":
                last_unpacked[0][last_unpacked[1]: last_unpacked[2]] ^= \
                    o_last_unpacked[0][last_unpacked[1]: last_unpacked[2]]
            self._data[-1] = np.packbits(last_unpacked[0], bitorder="little")[0]
        if mid_data is not None:
            if operation == "and":
                mid_data[:] &= o_mid_data[:]
            elif operation == "or":
                mid_data[:] |= o_mid_data[:]
            elif operation == "xor":
                mid_data[:] ^= o_mid_data[:]

    def _set_bits_at_locs(self, locs):
        if len(locs) == 0:
            return

        if locs.min() < 0 or locs.max() >= self.size:
            raise IndexError("Location indices out of range.")

        _locs = locs + self._start_index

        np.bitwise_or.at(
            self._data,
            _locs // 8,
            1 << (_locs % 8).astype(np.uint8),
        )

    def _clear_bits_at_locs(self, locs):
        if len(locs) == 0:
            return

        if locs.min() < 0 or locs.max() >= self.size:
            raise IndexError("Location indices out of range.")

        _locs = locs + self._start_index

        np.bitwise_and.at(
            self._data,
            _locs // 8,
            ~(1 << (_locs % 8).astype(np.uint8)),
        )

    def _test_bits_at_locs(self, locs):
        if len(locs) == 0:
            return np.zeros([], dtype=np.bool_)

        if locs.min() < 0 or locs.max() >= self.size:
            raise IndexError("Location indices out of range.")

        _locs = locs + self._start_index

        return self._data[_locs // 8] & (1 << (_locs % 8).astype(np.uint8)) != 0

    def _bit_count(self, arr):
        if self.LUT is None:
            _s55 = np.uint8(0x55)
            _s33 = np.uint8(0x33)
            _s0F = np.uint8(0x0F)
            _s01 = np.uint8(0x01)
            self.LUT = np.arange(2**8, dtype=np.uint8)
            self.LUT = self.LUT - ((self.LUT >> 1) & _s55)
            self.LUT = (self.LUT & _s33) + ((self.LUT >> 2) & _s33)
            self.LUT = (self.LUT + (self.LUT >> 4)) & _s0F
            self.LUT *= _s01

        return self.LUT[arr]
