import numpy as np
import numbers

from .utils import is_integer_value


class _PackedBoolArray:
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
    """
    def __init__(self, size=0, fill_value=False, data_buffer=None):
        if data_buffer is not None:
            if not isinstance(data_buffer, np.ndarray) or data_buffer.dtype != np.uint8:
                raise ValueError("data_buffer must be a numpy array of type uint8")

            self._data = data_buffer
        else:
            # Check if size is multiple of 8.
            if (size % 8) != 0:
                raise ValueError("_PackedBoolArray must have a size that is a multiple of 8.")

            self._data = np.zeros(size // 8, dtype=np.uint8)
            # We need to rething this part.
            if fill_value:
                # Set to all 1s
                self._data[:] = 255

        # Reported dtype is numpy bool.
        self._dtype = np.dtype("bool")

        # Set up constants for bit counting
        self._s55 = np.uint8(0x55)
        self._s33 = np.uint8(0x33)
        self._s0F = np.uint8(0x0F)
        self._s01 = np.uint8(0x01)

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return (self.size, )

    @property
    def size(self):
        return len(self._data) * 8

    def resize(self, newsize, refcheck=False):
        if (newsize % 8) != 0:
            raise ValueError("_PackedBoolArray must have a size that is a multiple of 8.")

        self._data.resize(newsize // 8, refcheck=refcheck)

    def sum(self, shape=None, axis=None):
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
        return self._data

    def copy(self):
        """Return a copy.

        Returns
        -------
        copy : `_PackedBoolArray`
        """
        return _PackedBoolArray(data_buffer=self._data.copy())

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"_PackedBoolArray(size={self.size})"

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        # If it's a slice, we return a _PackedBoolArray view.
        if isinstance(key, slice):
            if key.start is not None:
                if key.start % 8 != 0:
                    raise ValueError("Slices of _PackedBoolArray must start with a multiple of 8.")
                start = key.start // 8
            else:
                start = None
            if key.stop is not None:
                if key.stop % 8 != 0:
                    raise ValueError("Slices of _PackedBoolArray must end with a multiple of 8.")
                stop = key.stop // 8
            else:
                stop = None
            if key.step is not None:
                if key.step % 8 != 0:
                    raise ValueError("Slices of _PackedBoolArray must have a step multiple of 8.")
                step = key.step // 8
            else:
                step = None

            return _PackedBoolArray(data_buffer=self._data[slice(start, stop, step)])
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
        # FIXME:
        # - consolidate code
        # - check bounds
        if isinstance(key, numbers.Integral):
            # Need to check that value is single-valued; and bool.
            if value:
                return self._set_bits_at_locs(key)
            else:
                return self._clear_bits_at_locs(key)
        elif isinstance(key, slice):
            slice8 = True
            if key.start is not None:
                if key.start % 8 != 0:
                    slice8 = False
                else:
                    start = key.start // 8
            else:
                start = None
            if key.stop is not None:
                if key.stop % 8 != 0:
                    slice8 = False
                else:
                    stop = key.stop // 8
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
                        self._data[s8] = np.uint8(-1)
                    else:
                        # This is all False.
                        self._data[s8] = np.uint8(0)
                elif isinstance(value, np.ndarray):
                    if value.dtype != self._dtype:
                        raise ValueError("Can only set to array of bools")
                    if len(range(*s8.indices(len(self._data)))) != value.size // 8:
                        raise ValueError("Length of values does not match slice.")
                    self._data[s8] = np.packbits(value, bitorder="little")
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
                    return self._clear_bits(self._data, key)
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
        return np.unpackbits(self._data, bitorder="little")

    def __and__(self, other):
        raise NotImplementedError("and function not supported for _PackedBoolArray")

    def __iand__(self, other):
        raise NotImplementedError("and function not supported for _PackedBoolArray")

    def __or__(self, other):
        raise NotImplementedError("or function not supported for _PackedBoolArray")

    def __ior__(self, other):
        raise NotImplementedError("or function not supported for _PackedBoolArray")

    def __xor__(self, other):
        raise NotImplementedError("xor function not supported for _PackedBoolArray")

    def __ixor__(self, other):
        raise NotImplementedError("xor function not supported for _PackedBoolArray")

    def _set_bits_at_locs(self, locs):
        _locs = np.atleast_1d(locs)

        np.bitwise_or.at(
            self._data,
            _locs // 8,
            1 << (_locs % 8).astype(np.uint8),
        )

    def _clear_bits_at_locs(self, locs):
        _locs = np.atleast_1d(locs)

        np.bitwise_and.at(
            self._data,
            _locs // 8,
            ~(1 << (_locs % 8).astype(np.uint8)),
        )

    def _test_bits_at_locs(self, locs):
        _locs = np.atleast_1d(locs)

        return self._data[_locs // 8] & (1 << (_locs % 8).astype(np.uint8)) != 0

    def _bit_count(self, arr):
        arr = arr - ((arr >> 1) & self._s55)
        arr = (arr & self._s33) + ((arr >> 2) & self._s33)
        arr = (arr + (arr >> 4)) & self._s0F
        return arr * self._s01
