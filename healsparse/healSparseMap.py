import numpy as np
import hpgeom as hpg
import numbers

from .healSparseCoverage import HealSparseCoverage
from .utils import reduce_array, check_sentinel, _bitvals_to_packed_array
from .utils import WIDE_NBIT, WIDE_MASK, PIXEL_RANGE_THRESHOLD
from .utils import is_integer_value, _compute_bitshift
from .io_map import _read_map, _write_map, _write_moc
from .packedBoolArray import _PackedBoolArray
from .geom import GeomBase
import warnings


class HealSparseMap(object):
    """
    Class to define a HealSparseMap
    """

    def __init__(self, cov_map=None, cov_index_map=None, sparse_map=None, nside_sparse=None,
                 healpix_map=None, nside_coverage=None, primary=None, sentinel=None,
                 nest=True, metadata=None, _is_view=False):
        """
        Instantiate a HealSparseMap.

        Can be created with cov_index_map, sparse_map, and nside_sparse; or with
        healpix_map, nside_coverage.  Also see `HealSparseMap.read()`,
        `HealSparseMap.make_empty()`, `HealSparseMap.make_empty_like()`.

        Parameters
        ----------
        cov_map : `HealSparseCoverage`, optional
           Coverage map object
        cov_index_map : `np.ndarray`, optional
           Coverage index map, will be deprecated
        sparse_map : `np.ndarray`, optional
           Sparse map
        nside_sparse : `int`, optional
           Healpix nside for sparse map
        healpix_map : `np.ndarray`, optional
           Input healpix map to convert to a sparse map
        nside_coverage : `int`, optional
           Healpix nside for coverage map
        primary : `str`, optional
           Primary key for recarray, required if dtype has fields.
        sentinel : `int` or `float`, optional
           Sentinel value.  Default is `UNSEEN` for floating-point types,
           minimum int for int types, and False for bool types.
        nest : `bool`, optional
           If input healpix map is in nest format.  Default is True.
        metadata : `dict`-like, optional
           Map metadata that can be stored in FITS header format.
        _is_view : `bool`, optional
           This healSparse map is a view into another healsparse map.
           Not all features will be available.  (Internal usage)

        Returns
        -------
        healSparseMap : `HealSparseMap`
        """
        if cov_index_map is not None and cov_map is not None:
            raise RuntimeError('Cannot specify both cov_index_map and cov_map')
        if cov_index_map is not None:
            warnings.warn("cov_index_map deprecated", DeprecationWarning, stacklevel=2)
            cov_map = HealSparseCoverage(cov_index_map, nside_sparse)

        if cov_map is not None and sparse_map is not None and nside_sparse is not None:
            # this is a sparse map input
            self._cov_map = cov_map
            self._sparse_map = sparse_map
        elif healpix_map is not None and nside_coverage is not None:
            # this is a healpix_map input
            if sentinel is None:
                sentinel = hpg.UNSEEN
            if is_integer_value(healpix_map[0]) and not is_integer_value(sentinel):
                raise ValueError("The sentinel must be set to an integer value with an integer healpix_map")
            elif not is_integer_value(healpix_map[0]) and is_integer_value(sentinel):
                raise ValueError("The sentinel must be set to an float value with an float healpix_map")

            self._cov_map, self._sparse_map = self.convert_healpix_map(healpix_map,
                                                                       nside_coverage=nside_coverage,
                                                                       nest=nest,
                                                                       sentinel=sentinel)
            nside_sparse = hpg.npixel_to_nside(healpix_map.size)
        else:
            raise RuntimeError("Must specify either cov_map/sparse_map or healpix_map/nside_coverage")

        self._nside_sparse = nside_sparse

        self._n_valid = None
        self._is_rec_array = False
        self._is_wide_mask = False
        self._is_bit_packed = False
        self._wide_mask_width = 0
        self._primary = primary
        self.metadata = metadata
        self._is_view = _is_view
        if self._sparse_map.dtype.fields is not None:
            self._is_rec_array = True
            if self._primary is None:
                raise RuntimeError("Must specify `primary` field when using a recarray for the sparse_map.")

            self._sentinel = check_sentinel(self._sparse_map[self._primary].dtype.type, sentinel)
        else:
            if ((self._sparse_map.dtype.type == WIDE_MASK) and len(self._sparse_map.shape) == 2):
                self._is_wide_mask = True
                self._wide_mask_width = self._sparse_map.shape[1]
                self._wide_mask_maxbits = WIDE_NBIT * self._wide_mask_width
            elif isinstance(self._sparse_map, _PackedBoolArray):
                self._is_bit_packed = True
                if sentinel is not False:
                    raise NotImplementedError("Can only use False sentinel for bit_packed maps.")
                if (self._cov_map.nfine_per_cov % 8) != 0:
                    raise ValueError("Can only create a bit_packed map at least two "
                                     "healpix levels between coverage and mask.")

            self._sentinel = check_sentinel(self._sparse_map.dtype.type, sentinel)

    @classmethod
    def read(cls, filename, nside_coverage=None, pixels=None, header=False,
             degrade_nside=None, weightfile=None, reduction='mean',
             use_threads=False):
        """
        Read in a HealSparseMap.

        Parameters
        ----------
        filename : `str`
           Name of the file to read.  May be either a regular HEALPIX
           map or a HealSparseMap
        nside_coverage : `int`, optional
           Nside of coverage map to generate if input file is healpix map.
        pixels : `list`, optional
           List of coverage map pixels to read.  Only used if input file
           is a HealSparseMap
        header : `bool`, optional
           Return the fits header metadata as well as map?  Default is False.
        degrade_nside : `int`, optional
           Degrade map to this nside on read.  None means leave as-is.
           Not yet implemented for parquet files.
        weightfile : `str`, optional
           Floating-point map to supply weights for degrade wmean.  Must
           be a HealSparseMap (weighted degrade not supported for
           healpix degrade-on-read).
           Not yet implemented for parquet files.
        reduction : `str`, optional
           Reduction method with degrade-on-read.
           (mean, median, std, max, min, and, or, sum, prod, wmean).
           Not yet implemented for parquet files.
        use_threads : `bool`, optional
           Use multithreaded reading for parquet files.

        Returns
        -------
        healSparseMap : `HealSparseMap`
           HealSparseMap from file, covered by pixels
        header : `fitsio.FITSHDR` or `astropy.io.fits` (if header=True)
           Fits header for the map file.
        """
        return _read_map(cls, filename, nside_coverage=nside_coverage, pixels=pixels,
                         header=header, degrade_nside=degrade_nside,
                         weightfile=weightfile, reduction=reduction, use_threads=use_threads)

    @classmethod
    def make_empty(cls, nside_coverage, nside_sparse, dtype, primary=None, sentinel=None,
                   wide_mask_maxbits=None, metadata=None, cov_pixels=None, bit_packed=False):
        """
        Make an empty map with nothing in it.

        Parameters
        ----------
        nside_coverage : `int`
            Nside for the coverage map
        nside_sparse : `int`
            Nside for the sparse map
        dtype : `str` or `list` or `np.dtype`
            Datatype, any format accepted by numpy.
        primary : `str`, optional
            Primary key for recarray, required if dtype has fields.
        sentinel : `int` or `float`, optional
            Sentinel value.  Default is `UNSEEN` for floating-point types,
            and minimum int for int types.
        wide_mask_maxbits : `int`, optional
            Create a "wide bit mask" map, with this many bits.
        metadata : `dict`-like, optional
            Map metadata that can be stored in FITS header format.
        cov_pixels : `np.ndarray` or `list`
            List of integer coverage pixels to pre-allocate
        bit_packed : `bool`, optional
            Use bit-packed array for boolean mask?  (dtype must be boolean).

        Returns
        -------
        healSparseMap : `HealSparseMap`
           HealSparseMap filled with sentinel values.
        """
        test_arr = np.zeros(1, dtype=dtype)

        if wide_mask_maxbits is not None:
            if test_arr.dtype != WIDE_MASK:
                raise ValueError("Must use dtype=healsparse.WIDE_MASK to use a wide_mask")
            if sentinel is not None:
                if sentinel != 0:
                    raise ValueError("Sentinel must be 0 for wide_mask")
            nbitfields = (wide_mask_maxbits - 1) // WIDE_NBIT + 1
        if bit_packed:
            if dtype not in (np.bool_, bool):
                raise ValueError("Must use dtype=np.bool_ or bool to use bit_packed")

        if cov_pixels is None:
            cov_map = HealSparseCoverage.make_empty(nside_coverage, nside_sparse)
            # One pixel is the overflow pixel of a truly empty map
            npix = 1
        else:
            cov_pixels = np.atleast_1d(cov_pixels)
            cov_map = HealSparseCoverage.make_from_pixels(nside_coverage, nside_sparse,
                                                          cov_pixels)
            # We need to allocate the overflow pixel
            npix = cov_pixels.size + 1

        if wide_mask_maxbits is not None:
            # The sentinel is always zero
            _sentinel = 0
            sparse_map = np.zeros((cov_map.nfine_per_cov*npix, nbitfields), dtype=dtype)
        elif bit_packed:
            _sentinel = check_sentinel(test_arr.dtype.type, sentinel)
            if (cov_map.nfine_per_cov % 8) != 0:
                raise ValueError("Can only create a bit_packed mask at least two "
                                 "healpix levels between coverage and mask.")
            if _sentinel:
                raise NotImplementedError("Can only create a bit_packed map with False sentinel value.")
            sparse_map = _PackedBoolArray(size=cov_map.nfine_per_cov*npix)
        elif test_arr.dtype.fields is None:
            # Non-recarray
            _sentinel = check_sentinel(test_arr.dtype.type, sentinel)
            sparse_map = np.full(cov_map.nfine_per_cov*npix, _sentinel, dtype=dtype)
        else:
            # Recarray type
            if primary is None:
                raise RuntimeError("Must specify 'primary' field when using a recarray for the sparse_map.")

            primary_found = False
            for name in test_arr.dtype.names:
                if name == primary:
                    _sentinel = check_sentinel(test_arr[name].dtype.type, sentinel)
                    test_arr[name] = _sentinel
                    primary_found = True
                else:
                    test_arr[name] = check_sentinel(test_arr[name].dtype.type, None)

            if not primary_found:
                raise RuntimeError("Primary field not found in input dtype of recarray.")

            sparse_map = np.full(cov_map.nfine_per_cov*npix, test_arr, dtype=dtype)

        return cls(cov_map=cov_map, sparse_map=sparse_map,
                   nside_sparse=nside_sparse, primary=primary, sentinel=_sentinel,
                   metadata=metadata)

    @classmethod
    def make_empty_like(cls, sparsemap, nside_coverage=None, nside_sparse=None, dtype=None,
                        primary=None, sentinel=None, wide_mask_maxbits=None, metadata=None,
                        cov_pixels=None, bit_packed=False):
        """
        Make an empty map with the same parameters as an existing map.

        Parameters
        ----------
        sparsemap : `HealSparseMap`
           Sparse map to use as basis for new empty map.
        nside_coverage : `int`, optional
           Coverage nside, default to sparsemap.nside_coverage
        nside_sparse : `int`, optional
           Sparse map nside, default to sparsemap.nside_sparse
        dtype : `str` or `list` or `np.dtype`, optional
           Datatype, any format accepted by numpy.  Default is sparsemap.dtype
        primary : `str`, optional
           Primary key for recarray.  Default is sparsemap.primary
        sentinel : `int` or `float`, optional
           Sentinel value.  Default is sparsemap._sentinel
        wide_mask_maxbits : `int`, optional
           Create a "wide bit mask" map, with this many bits.
        metadata : `dict`-like, optional
           Map metadata that can be stored in FITS header format.
        cov_pixels : `np.ndarray` or `list`
           List of integer coverage pixels to pre-allocate
        bit_packed : `bool`, optional
            Use bit-packed array for boolean mask?  (dtype must be boolean).

        Returns
        -------
        healSparseMap : `HealSparseMap`
           HealSparseMap filled with sentinel values.
        """
        if nside_coverage is None:
            nside_coverage = sparsemap.nside_coverage
        if nside_sparse is None:
            nside_sparse = sparsemap.nside_sparse
        if dtype is None:
            dtype = sparsemap.dtype
        if primary is None:
            primary = sparsemap.primary
        if sentinel is None:
            sentinel = sparsemap._sentinel
        if wide_mask_maxbits is None:
            if sparsemap._is_wide_mask:
                wide_mask_maxbits = sparsemap._wide_mask_maxbits
        if bit_packed is None:
            bit_packed = sparsemap._is_bit_packed
        if metadata is None:
            metadata = sparsemap._metadata

        return cls.make_empty(nside_coverage, nside_sparse, dtype, primary=primary,
                              sentinel=sentinel, wide_mask_maxbits=wide_mask_maxbits,
                              metadata=metadata, cov_pixels=cov_pixels, bit_packed=bit_packed)

    @staticmethod
    def convert_healpix_map(healpix_map, nside_coverage, nest=True, sentinel=hpg.UNSEEN):
        """
        Convert a healpix map to a healsparsemap.

        Parameters
        ----------
        healpix_map : `np.ndarray`
           Numpy array that describes a healpix map.
        nside_coverage : `int`
           Nside for the coverage map to construct
        nest : `bool`, optional
           Is the input map in nest format?  Default is True.
        sentinel : `float`, optional
           Sentinel value for null values in the sparse_map.

        Returns
        -------
        cov_map : `HealSparseCoverage`
           Coverage map with pixel indices
        sparse_map : `np.ndarray`
           Sparse map of input values.
        """
        if not nest:
            healpix_map = hpg.reorder(healpix_map, ring_to_nest=True)

        # Compute the coverage map...
        # Note that this is coming from a standard healpix map so the sentinel
        # is always hpg.UNSEEN
        ipnest, = np.where(healpix_map > hpg.UNSEEN)

        nside_sparse = hpg.npixel_to_nside(healpix_map.size)
        cov_map = HealSparseCoverage.make_empty(nside_coverage, nside_sparse)

        ipnest_cov = cov_map.cov_pixels(ipnest)
        cov_pix = np.unique(ipnest_cov)

        cov_map.initialize_pixels(cov_pix)

        sparse_map = np.full((cov_pix.size + 1)*cov_map.nfine_per_cov,
                             sentinel, dtype=healpix_map.dtype)
        sparse_map[ipnest + cov_map[ipnest_cov]] = healpix_map[ipnest]

        return cov_map, sparse_map

    def write(self, filename, clobber=False, nocompress=False, format='fits', nside_io=4):
        """
        Write a HealSparseMap to a file.  Use the `metadata` property from
        the map to persist additional information in the fits header.

        Parameters
        ----------
        filename : `str`
            Name of file to save
        clobber : `bool`, optional
            Clobber existing file?  Default is False.
        nocompress : `bool`, optional
            If this is False, then integer maps will be compressed losslessly.
            Note that `np.int64` maps cannot be compressed in the FITS standard.
            This option only applies if format=``fits``.
        nside_io : `int`, optional
            The healpix nside to partition the output map files in parquet.
            Must be less than or equal to nside_coverage, and not greater than 16.
            This option only applies if format=``parquet``.
        format : `str`, optional
            File format.  May be ``fits``, ``parquet``, or ``healpix``. Note that
            the ``healpix`` EXPLICIT format does not maintain all metadata and
            coverage information.

        Raises
        ------
        NotImplementedError if file format is not supported.
        ValueError if nside_io is out of range.
        """
        _write_map(self, filename, clobber=clobber, nocompress=nocompress, format=format,
                   nside_io=nside_io)

    def write_moc(self, filename, clobber=False):
        """
        Write the valid pixels of a HealSparseMap to a multi-order component (MOC)
        file.  Note that values of the pixels are not persisted in MOC format.

        Parameters
        ----------
        filename : `str`
            Name of file to save
        clobber : `bool`, optional
            Clobber existing file?  Default is False.
        """
        _write_moc(self, filename, clobber=clobber)

    def _reserve_cov_pix(self, new_cov_pix):
        """
        Reserve new coverage pixels.  This routine does no checking, it should
        be done by the caller.

        Parameters
        ----------
        new_cov_pix : `np.ndarray`
           Integer array of new coverage pixels
        """

        new_cov_map = self._cov_map.append_pixels(len(self._sparse_map), new_cov_pix, check=False)
        self._cov_map = new_cov_map

        # Use resizing
        oldsize = len(self._sparse_map)
        newsize = oldsize + new_cov_pix.size*self._cov_map.nfine_per_cov

        if self._is_wide_mask:
            self._sparse_map.resize((newsize, self._wide_mask_width), refcheck=False)
        else:
            self._sparse_map.resize(newsize, refcheck=False)

        # Fill with blank values
        self._sparse_map[oldsize:] = self._sparse_map[0]

    def update_values_pos(self, ra_or_theta, dec_or_phi, values,
                          lonlat=True, operation='replace'):
        """
        Update the values in the sparsemap for a list of positions.

        Parameters
        ----------
        ra_or_theta : `float`, array-like
            Angular coordinates of points on a sphere.
        dec_or_phi : `float`, array-like
            Angular coordinates of points on a sphere.
        values : `np.ndarray` or `None`
            Value or Array of values.  Must be same type as sparse_map.
            If None, then the pixels will be set to the sentinel map value.
            If None is selected then no additional coverage pixels will be
            added as a result of the operation.
        lonlat : `bool`, optional
            If True, input angles are longitude and latitude in degrees.
            Otherwise, they are co-latitude and longitude in radians.
        operation : `str`, optional
            Operation to use to update values.  May be 'replace' (default);
            'add'; 'or', or 'and' (for bit masks).

        Raises
        ------
        ValueError
            If positions do not resolve to unique positions and operation
            is 'replace', or if values is None and operation is not 'replace'.

        Notes
        -----
        During the 'add' operation, if the default sentinel map value is not
        equal to 0, then any default values will be set to 0 prior to addition.
        """
        return self.update_values_pix(hpg.angle_to_pixel(self._nside_sparse,
                                                         ra_or_theta,
                                                         dec_or_phi,
                                                         lonlat=lonlat),
                                      values,
                                      operation=operation)

    def update_values_pix(self, pixels, values, nest=True, operation='replace'):
        """
        Update the values in the sparsemap for a list of pixels.
        The list of pixels must be unique if the operation is 'replace'.

        Parameters
        ----------
        pixels : `np.ndarray` (M,) or (M, 2)
            Integer array of sparse_map pixel values.  If this is a 2D array
            of shape (M, 2), this is interpreted as pixel ranges where each
            row is [start, end).
        values : `np.ndarray` or `None`
            Value or Array of values.  Must be same type as sparse_map.
            If None, then the pixels will be set to the sentinel map value.
            If None is selected then no additional coverage pixels will be
            added as a result of the operation.
        operation : `str`, optional
            Operation to use to update values.  May be 'replace' (default);
            'add'; 'or', or 'and' (for bit masks).

        Raises
        ------
        ValueError
            Raised if pixels are not unique and operation is 'replace', or if
            operation is not 'replace' on a recarray map, or if values is
            None and operation is not 'replace'.

        Notes
        -----
        During the 'add' operation, if the default sentinel map value is not
        equal to 0, then any default values will be set to 0 prior to addition.
        """
        # We invalidate the n_valid cache here.
        self._n_valid = None

        # When None is specified, we use the sentinel value.
        no_append = False
        if values is None:
            if operation != 'replace':
                raise ValueError("Can only use 'None' with 'replace' operation.")

            if self._is_wide_mask:
                values = np.full(self._wide_mask_width, self._sentinel)
            elif self._is_rec_array:
                values = np.zeros(1, dtype=self._sparse_map.dtype)
                values[self._primary] = self._sentinel
            else:
                values = self._sentinel
            no_append = True

        if operation != 'replace':
            if self.dtype == np.bool_:
                if operation not in ['or', 'and']:
                    raise NotImplementedError("Booleam maps Can only use replace/and/or operations.")
            elif operation in ['or', 'and']:
                if not self.is_integer_map or self._sentinel != 0:
                    raise ValueError("Can only use and/or with integer map with 0 sentinel")
            elif operation == 'add':
                if self._is_rec_array:
                    raise ValueError("Cannot use 'add' operation with a recarray map.")
            else:
                raise ValueError("Only 'replace', 'add', 'or', and 'and' are supported operations")

        # If _not_ recarray, we can use a single int/float
        is_single_value = False
        _values = values
        if not self._is_rec_array:
            if self._is_wide_mask:
                # Special for wide_mask
                if not isinstance(values, np.ndarray):
                    raise ValueError("Wide mask must be set with a numpy ndarray")
                if len(values) == self._wide_mask_width and len(values.shape) == 1:
                    is_single_value = True
                    # Reshape so we can use the 0th entry below
                    _values = _values.reshape((1, self._wide_mask_width))
            else:
                # Non wide_mask
                if isinstance(values, numbers.Integral):
                    if not self.is_integer_map:
                        raise ValueError("Cannot set non-integer map with an integer")
                    is_single_value = True
                    _values = np.array([values], dtype=self.dtype)
                elif isinstance(values, numbers.Real):
                    if self.is_integer_map:
                        raise ValueError("Cannot set non-floating point map with a floating point.")
                    is_single_value = True
                    _values = np.array([values], dtype=self.dtype)
                elif isinstance(values, (bool, np.bool_)):
                    is_single_value = True
                    _values = np.array([values], dtype=bool)

        if isinstance(values, np.ndarray) and len(values) == 1:
            is_single_value = True

        # First, check if these are the same type
        if not is_single_value and not isinstance(_values, np.ndarray):
            raise ValueError("Values are not a numpy ndarray")

        if hasattr(pixels, "__len__") and len(pixels) == 0:
            if len(_values) > 1:
                warnings.warn("Shape mismatch: using a non-zero-length array of values "
                              "to set a zero-length list of pixels.",
                              UserWarning)
            # Nothing to do
            return

        # Check numpy data type for everything but wide_mask single value
        if not self._is_wide_mask or (self._is_wide_mask and not is_single_value):
            if self._is_rec_array:
                if self._sparse_map.dtype != _values.dtype:
                    raise ValueError("Data-type mismatch between sparse_map and values")
            elif self._sparse_map.dtype.type != _values.dtype.type:
                raise ValueError("Data-type mismatch between sparse_map and values")

        if operation == 'replace':
            # Check for unique pixel positions
            if hasattr(pixels, "__len__"):
                if len(np.unique(pixels)) < len(pixels):
                    raise ValueError("List of pixels must be unique if operation='replace'")

        if pixels.ndim == 2 and pixels.shape[1] == 2:
            # These are pixel ranges.
            if not is_single_value:
                raise ValueError("Can only use a single value with pixel ranges (N, 2) input.")
            if not nest:
                raise ValueError("Can only use pixel ranges with nest ordering.")

            # At the risk of premature optimization, we only call the special function
            # if the number of pixels is above some threshold.
            pixels_to_set = np.sum(pixels[:, 1] - pixels[:, 0])
            if pixels_to_set > PIXEL_RANGE_THRESHOLD:
                return self._update_values_pixel_ranges(pixels, _values[0], operation, no_append)
            else:
                _pix = hpg.pixel_ranges_to_pixels(pixels)
        elif not nest:
            _pix = hpg.ring_to_nest(self._nside_sparse, pixels)
        else:
            _pix = pixels

        # Check array lengths
        if not is_single_value and len(_values) != pixels.size:
            raise ValueError("Length of values must be same length as pixels (or length 1)")

        if self._is_view:
            # Check that we are not setting new pixels
            if np.any(self.get_values_pix(_pix) == self._sentinel):
                raise RuntimeError("This API cannot be used to set new pixels in the map.")

        # Compute the coverage pixels
        ipnest_cov = self._cov_map.cov_pixels(_pix)

        # Check which pixels are in the coverage map
        cov_mask = self.coverage_mask
        in_cov = cov_mask[ipnest_cov]
        out_cov = ~cov_mask[ipnest_cov]

        # This little internal function is used by several modes below
        # and it is much clearer to pull it out.
        def _do_operation_on_sparse_map(operation, sparse_map, indices, values):
            if operation == "replace":
                sparse_map[indices] = values
            elif operation == "add":
                # Put in a check to reset uncovered pixels to 0
                if self._sentinel != 0:
                    sparse_map[indices[sparse_map[indices] == self._sentinel]] = 0
                np.add.at(sparse_map, indices, values)
            elif operation == "or":
                if self._is_bit_packed:
                    sparse_map[indices] |= values
                else:
                    np.bitwise_or.at(sparse_map, indices, values)
            elif operation == "and":
                if self._is_bit_packed:
                    sparse_map[indices] &= values
                else:
                    np.bitwise_and.at(sparse_map, indices, values)

        # Replace values for those pixels in the coverage map
        _indices = _pix[in_cov] + self._cov_map[ipnest_cov[in_cov]]
        if is_single_value:
            _do_operation_on_sparse_map(operation, self._sparse_map, _indices, _values[0])
        else:
            _do_operation_on_sparse_map(operation, self._sparse_map, _indices, _values[in_cov])

        # Update the coverage map for the rest of the pixels (if necessary)
        if out_cov.sum() > 0 and not no_append:
            # New version to minimize data copying

            # Faster trick for getting unique values
            new_cov_temp = np.zeros(cov_mask.size, dtype=np.int8)
            new_cov_temp[ipnest_cov[out_cov]] = 1
            new_cov_pix, = np.where(new_cov_temp > 0)

            # Reserve the memory here
            oldsize = len(self._sparse_map)
            self._reserve_cov_pix(new_cov_pix)

            _indices = _pix[out_cov] + self._cov_map[ipnest_cov[out_cov]] - oldsize
            if is_single_value:
                _do_operation_on_sparse_map(operation, self._sparse_map[oldsize:], _indices, _values[0])
            else:
                _do_operation_on_sparse_map(operation, self._sparse_map[oldsize:], _indices, _values[out_cov])

    def _update_values_pixel_ranges(self, pixel_ranges, value, operation, no_append):
        """
        Update a set of pixel ranges with a given (single) value.

        All inputs should be validated prior to calling this internal routine.

        Parameters
        ----------
        pixel_ranges : `np.ndarray` (M, 2)
            2D array of pixel ranges.
        value : `int` or `float` or `bool`
            Single value to set.
        operation : `str`
            Operation to apply.  Must be ``replace``, ``and``, ``or`` or ``add``.
        no_append : `bool`
            If True, no coverage pixels will be appended.
        """
        # Compute the coverage pixels.
        cov_pix_ranges = np.right_shift(pixel_ranges, self._cov_map.bit_shift)
        # After the bit shift these pixel ranges are inclusive, not exclusive.
        cov_pix_to_set = hpg.pixel_ranges_to_pixels(cov_pix_ranges, inclusive=True)
        cov_pix_to_set = np.unique(cov_pix_to_set)

        cov_mask = self.coverage_mask

        new_cov_pixels = cov_pix_to_set[~cov_mask[cov_pix_to_set]]

        if not no_append and len(new_cov_pixels) > 0:
            # Reserve more storage for new coverage pixels.
            self._reserve_cov_pix(new_cov_pixels)

        # Check which of these ranges cover more than one pixel?
        delta_pix = pixel_ranges[:, 1] - pixel_ranges[:, 0]
        delta_covpix = cov_pix_ranges[:, 1] - cov_pix_ranges[:, 0]

        covpix_start_values = (self._cov_map[cov_pix_ranges.ravel()] +
                               self._cov_map.nfine_per_cov*cov_pix_ranges.ravel()
                               ).reshape(cov_pix_ranges.shape)

        covpix_offset_values = self._cov_map[self._cov_map.cov_pixels_from_index(
            covpix_start_values.ravel()
        )].reshape(cov_pix_ranges.shape)

        def _do_operation_on_sparse_map_range(operation, sparse_map, start, stop, value):
            # Note that start: stop will not have overlapping pixels, so we do
            # not need to use ufunc.at() to perform operations.
            if operation == "replace":
                sparse_map[start: stop] = value
            elif operation == "add":
                # Put in a check to reset uncovered pixels to 0
                if self._sentinel != 0:
                    sparse_map[start: stop][sparse_map[start: stop] == self._sentinel] = 0
                sparse_map[start: stop] += value
            elif operation == "or":
                sparse_map[start: stop] |= value
            elif operation == "and":
                sparse_map[start: stop] &= value

        # Loop over ranges.
        for i in range(pixel_ranges.shape[0]):
            if delta_covpix[i] > 0:
                # This range overlaps multiple coverage pixels.
                if no_append and not cov_mask[cov_pix_ranges[i, 0]]:
                    # Nothing to set here.
                    pass
                else:
                    # The first coverage pixel will be partly covered.
                    start = pixel_ranges[i, 0] + covpix_offset_values[i, 0]
                    stop = (
                        self._cov_map[cov_pix_ranges[i, 0]] +
                        self._cov_map.nfine_per_cov*(cov_pix_ranges[i, 0] + 1)
                    )
                    _do_operation_on_sparse_map_range(operation, self._sparse_map, start, stop, value)

                # The middle coverage pixels will be fully covered.
                for cov_pix_full in range(cov_pix_ranges[i, 0] + 1, cov_pix_ranges[i, 1]):
                    if no_append and not cov_mask[cov_pix_full]:
                        # Nothing to set here.
                        continue
                    start = (self._cov_map[cov_pix_full] + self._cov_map.nfine_per_cov*cov_pix_full)
                    stop = start + self._cov_map.nfine_per_cov
                    _do_operation_on_sparse_map_range(operation, self._sparse_map, start, stop, value)

                if no_append and not cov_mask[cov_pix_ranges[i, 1]]:
                    # Nothing to set here.
                    pass
                else:
                    # The final coverage pixel will be partly covered.
                    start = (self._cov_map[cov_pix_ranges[i, 1]] +
                             self._cov_map.nfine_per_cov*(cov_pix_ranges[i, 1])
                             )
                    stop = pixel_ranges[i, 1] + covpix_offset_values[i, 1]
                    _do_operation_on_sparse_map_range(operation, self._sparse_map, start, stop, value)
            else:
                if no_append and not cov_mask[cov_pix_ranges[i, 0]]:
                    # Nothing to set here.
                    continue

                # This range is fully contained in a coverage pixel.
                start = pixel_ranges[i, 0] + covpix_offset_values[i, 0]
                stop = start + delta_pix[i]

                _do_operation_on_sparse_map_range(operation, self._sparse_map, start, stop, value)

    def set_bits_pix(self, pixels, bits, nest=True):
        """
        Set bits of a wide_mask map.

        Parameters
        ----------
        pixels : `np.ndarray`
           Integer array of sparse_map pixel values
        bits : `list`
           List of bits to set
        """
        if not self._is_wide_mask:
            raise NotImplementedError("Can only use set_bits_pix on wide_mask map")

        if np.max(bits) >= self._wide_mask_maxbits:
            raise ValueError("Bit position %d too large (>= %d)" % (np.max(bits),
                                                                    self._wide_mask_maxbits))

        value = _bitvals_to_packed_array(bits, self._wide_mask_maxbits)

        self.update_values_pix(pixels, value, nest=nest, operation='or')

    def clear_bits_pix(self, pixels, bits, nest=True):
        """
        Clear bits of a wide_mask map.

        Parameters
        ----------
        pixels : `np.ndarray`
           Integer array of sparse_map pixel values
        bits : `list`
           List of bits to clear
        """
        if not self._is_wide_mask:
            raise NotImplementedError("Can only use set_bits_pix on wide_mask map")

        if np.max(bits) >= self._wide_mask_maxbits:
            raise ValueError("Bit position %d too large (>= %d)" % (np.max(bits),
                                                                    self._wide_mask_maxbits))

        value = _bitvals_to_packed_array(bits, self._wide_mask_maxbits)

        # A bit reset is performed with &= ~(bit1 | bit2)
        self.update_values_pix(pixels, ~value, nest=nest, operation='and')

    def get_values_pos(self, ra_or_theta, dec_or_phi, lonlat=True, valid_mask=False):
        """
        Get the map value for the position.  Positions may be theta/phi
        co-latitude and longitude in radians, or longitude and latitude in
        degrees.

        Parameters
        ----------
        ra_or_theta : `float`, array-like
           Angular coordinates of points on a sphere.
        dec_or_phi : `float`, array-like
           Angular coordinates of points on a sphere.
        lonlat : `bool`, optional
           If True, input angles are longitude and latitude in degrees.
           Otherwise, they are co-latitude and longitude in radians.
        valid_mask : `bool`, optional
           Return mask of True/False instead of values

        Returns
        -------
        values : `np.ndarray`
           Array of values/validity from the map.
        """
        return self.get_values_pix(hpg.angle_to_pixel(self._nside_sparse,
                                                      ra_or_theta,
                                                      dec_or_phi,
                                                      lonlat=lonlat),
                                   valid_mask=valid_mask)

    def get_values_pix(self, pixels, nest=True, valid_mask=False, nside=None):
        """
        Get the map value for a set of pixels.

        This routine will optionally convert from a higher resolution nside
        to the nside of the sparse map.

        Parameters
        ----------
        pixel : `np.ndarray`
            Integer array of healpix pixels.
        nest : `bool`, optional
            Are the pixels in nest scheme?  Default is True.
        valid_mask : `bool`, optional
            Return mask of True/False instead of values
        nside : `int`, optional
            nside of pixels, if different from native.
            Must be greater than the native nside.

        Returns
        -------
        values : `np.ndarray`
            Array of values/validity from the map.
        """
        if hasattr(pixels, "__len__") and len(pixels) == 0:
            if self._is_wide_mask:
                return np.zeros((0, self._wide_mask_width), dtype=self.dtype)
            else:
                return np.array([], dtype=self.dtype)

        if not nest:
            _pix = hpg.ring_to_nest(self._nside_sparse, pixels)
        else:
            _pix = pixels

        if nside is not None:
            if nside < self._nside_sparse:
                raise ValueError("nside must be higher resolution than the sparse map.")
            # Convert pixels to sparse map resolution
            bit_shift = _compute_bitshift(self._nside_sparse, nside)
            _pix = np.right_shift(_pix, np.abs(bit_shift))

        ipnest_cov = self._cov_map.cov_pixels(_pix)

        if self._is_wide_mask:
            values = self._sparse_map[_pix + self._cov_map[ipnest_cov], :]
        else:
            values = self._sparse_map[_pix + self._cov_map[ipnest_cov]]

        if valid_mask:
            if self._is_rec_array:
                return (values[self._primary] != self._sentinel)
            elif self._is_wide_mask:
                return (values > 0).sum(axis=1, dtype=np.bool_)
            else:
                return (values != self._sentinel)
        else:
            # Just return the values
            return values

    def check_bits_pos(self, ra_or_theta, dec_or_phi, bits, lonlat=True):
        """
        Check the bits at the map for an array of positions.  Positions may be
        theta/phi co-latitude and longitude in radians, or longitude and
        latitude in degrees.

        Parameters
        ----------
        ra_or_theta : `float`, array-like
           Angular coordinates of points on a sphere.
        dec_or_phi : `float`, array-like
           Angular coordinates of points on a sphere.
        lonlat : `bool`, optional
           If True, input angles are longitude and latitude in degrees.
           Otherwise, they are co-latitude and longitude in radians.
        bits : `list`
           List of bits to check

        Returns
        -------
        bit_flags : `np.ndarray`
           Array of `np.bool_` flags on whether any of the input bits were
           set
        """
        return self.check_bits_pix(hpg.angle_to_pixel(self._nside_sparse,
                                                      ra_or_theta,
                                                      dec_or_phi,
                                                      lonlat=lonlat),
                                   bits)

    def check_bits_pix(self, pixels, bits, nest=True):
        """
        Check the bits at the map for a set of pixels.

        Parameters
        ----------
        pixels : `np.ndarray`
           Integer array of healpix pixels.
        nest : `bool`, optional
           Are the pixels in nest scheme?  Default is True.
        bits : `list`
           List of bits to check

        Returns
        -------
        bit_flags : `np.ndarray`
           Array of `np.bool_` flags on whether any of the input bits were
           set
        """
        values = self.get_values_pix(np.atleast_1d(pixels), nest=nest)

        bit_value = _bitvals_to_packed_array(bits, self._wide_mask_maxbits)
        return np.any((values & bit_value) > 0, axis=1)

    @property
    def sentinel(self):
        """
        Get the sentinel of the map.
        """
        return self._sentinel

    @property
    def dtype(self):
        """
        get the dtype of the map
        """
        return self._sparse_map.dtype

    @property
    def coverage_map(self):
        """
        Get the fractional area covered by the sparse map
        in the resolution of the coverage map

        Returns
        -------
        cov_map : `np.ndarray`
           Float array of fractional coverage of each pixel
        """

        cov_map = np.zeros_like(self.coverage_mask, dtype=np.float64)
        cov_mask = self.coverage_mask
        npop_pix = np.count_nonzero(cov_mask)
        if self._is_wide_mask:
            shape_new = (npop_pix + 1,
                         self._cov_map.nfine_per_cov,
                         self._wide_mask_width)
            sp_map_t = self._sparse_map.reshape(shape_new)
            # This trickery first checks all the bits, and then sums into the
            # coverage pixel
            counts = np.sum(np.any(sp_map_t != self._sentinel, axis=2), axis=1)
        else:
            shape_new = (npop_pix + 1,
                         self._cov_map.nfine_per_cov)
            if self._is_rec_array:
                sp_map_t = self._sparse_map[self._primary].reshape(shape_new)
            else:
                sp_map_t = self._sparse_map.reshape(shape_new)
            counts = np.sum((sp_map_t != self._sentinel), axis=1).astype(np.float64)

        cov_map[cov_mask] = counts[1:]/self._cov_map.nfine_per_cov
        return cov_map

    @property
    def coverage_mask(self):
        """
        Get the boolean mask of the coverage map.

        Returns
        -------
        cov_mask : `np.ndarray`
           Boolean array of coverage mask.
        """
        return self._cov_map.coverage_mask

    def fracdet_map(self, nside):
        """
        Get the fractional area covered by the sparse map at an arbitrary resolution.
        This output fracdet_map counts the fraction of "valid" sub-pixels (those that
        are not equal to the sentinel value) at the desired nside resolution.

        Note: You should not compute the fracdet_map of an existing fracdet_map.  To
        get a fracdet_map at a lower resolution, use the degrade method with the
        default "mean" reduction.

        Parameters
        ----------
        nside : `int`
           Healpix nside for fracdet map.  Must not be greater than sparse
           resolution or less than coverage resolution.

        Returns
        -------
        fracdet_map : `HealSparseMap`
           Fractional coverage map.
        """
        if nside > self.nside_sparse:
            raise ValueError("Cannot return fracdet_map at higher resolution than "
                             "the sparse map (nside=%d)." % (self.nside_sparse))
        if nside < self.nside_coverage:
            raise ValueError("Cannot return fractdet_map at lower resolution than "
                             "the coverage map (nside=%d)." % (self.nside_coverage))

        # This code is essentially a unification of coverage_map() and degrade()
        # to get the fracdet_coverage in a single step
        cov_mask = self.coverage_mask
        npop_pix = np.count_nonzero(cov_mask)

        bit_shift = _compute_bitshift(nside, self.nside_sparse)
        nfine_per_frac = 2**bit_shift
        nfrac_per_cov = self._cov_map.nfine_per_cov//nfine_per_frac

        if self._is_wide_mask:
            shape_new = ((npop_pix + 1)*nfrac_per_cov,
                         nfine_per_frac,
                         self._wide_mask_width)
            sp_map_t = self._sparse_map.reshape(shape_new)
            fracdet = np.sum(np.any(sp_map_t != self._sentinel, axis=2), axis=1).astype(np.float64)
        elif self._is_bit_packed:
            shape_new = ((npop_pix + 1)*nfrac_per_cov, nfine_per_frac)
            fracdet = self._sparse_map.sum(shape=shape_new, axis=1).astype(np.float64)
        else:
            shape_new = ((npop_pix + 1)*nfrac_per_cov,
                         nfine_per_frac)
            if self._is_rec_array:
                sp_map_t = self._sparse_map[self._primary].reshape(shape_new)
            else:
                sp_map_t = self._sparse_map.reshape(shape_new)
            fracdet = np.sum(sp_map_t != self._sentinel, axis=1).astype(np.float64)

        fracdet /= nfine_per_frac

        fracdet_cov_map = HealSparseCoverage.make_from_pixels(self.nside_coverage,
                                                              nside,
                                                              np.where(cov_mask)[0])

        # The sentinel for a fracdet_map is 0.0, no coverage.
        return HealSparseMap(cov_map=fracdet_cov_map, sparse_map=fracdet,
                             nside_sparse=nside, primary=self._primary,
                             sentinel=0.0)

    @property
    def nside_coverage(self):
        """
        Get the nside of the coverage map

        Returns
        -------
        nside_coverage : `int`
        """
        return self._cov_map.nside_coverage

    @property
    def nside_sparse(self):
        """
        Get the nside of the sparse map

        Returns
        -------
        nside_sparse : `int`
        """

        return self._nside_sparse

    @property
    def primary(self):
        """
        Get the primary field

        Returns
        -------
        primary : `str`
        """

        return self._primary

    @property
    def is_integer_map(self):
        """
        Check that the map is an integer map

        Returns
        -------
        is_integer_map : `bool`
        """

        if self._is_rec_array:
            return False

        return issubclass(self._sparse_map.dtype.type, (np.integer, np.bool_))

    @property
    def is_unsigned_map(self):
        """
        Check that the map is an unsigned integer map

        Returns
        -------
        is_unsigned_map : `bool`
        """

        if self._is_rec_array:
            return False

        return issubclass(self._sparse_map.dtype.type, np.unsignedinteger)

    @property
    def is_wide_mask_map(self):
        """
        Check that the map is a wide mask

        Returns
        -------
        is_wide_mask_map : `bool`
        """
        return self._is_wide_mask

    @property
    def is_bit_packed_map(self):
        """
        Check that the map is a bit-packed mask.

        Returns
        -------
        is_bit_packed_map : `bool`
        """
        return self._is_bit_packed

    @property
    def wide_mask_width(self):
        """
        Get the width of the wide mask

        Returns
        -------
        wide_mask_width : `int`
           Width of wide mask array.  0 if not wide mask.
        """
        return self._wide_mask_width

    @property
    def wide_mask_maxbits(self):
        """
        Get the maximum number of bits stored in the wide mask.

        Returns
        -------
        wide_mask_maxbits : `int`
           Maximum number of bits.  0 if not wide mask.
        """
        if self._is_wide_mask:
            return self._wide_mask_maxbits
        else:
            return 0

    @property
    def is_rec_array(self):
        """
        Check that the map is a recArray map.

        Returns
        -------
        is_rec_array : `bool`
        """

        return self._is_rec_array

    @property
    def metadata(self):
        """
        Return the metadata dict.

        Returns
        -------
        metadata : `dict`
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        """
        Set the metadata dict.

        This ensures that the keys conform to FITS standard (<=8 char string,
        all caps.)

        Parameters
        ----------
        metadata : `dict`
        """
        if metadata is None:
            self._metadata = metadata
        else:
            if not isinstance(metadata, dict):
                try:
                    metadata = dict(metadata)
                except ValueError:
                    raise ValueError("Could not convert metadata to dict")
            for key in metadata:
                if not isinstance(key, str):
                    raise ValueError("metadata key %s must be a string" % (str(key)))
                if not key.isupper():
                    raise ValueError("metadata key %s must be all upper case" % (key))

            self._metadata = metadata

    def generate_healpix_map(self, nside=None, reduction='mean', key=None, nest=True):
        """
        Generate the associated healpix map

        if nside is specified, then reduce to that nside

        Parameters
        ----------
        nside : `int`
            Output nside resolution parameter (should be a multiple of 2). If
            not specified the output resolution will be equal to the parent's
            sparsemap nside_sparse
        reduction : `str`
            If a change in resolution is requested, this controls the method to
            reduce the map computing the "mean", "median", "std", "max", "min",
            "sum" or "prod" (product)  of the neighboring pixels to compute the
            "degraded" map.
        key : `str`
            If the parent HealSparseMap contains recarrays, key selects the
            field that will be transformed into a HEALPix map.
        nest : `bool`, optional
            Output healpix map should be in nest format?

        Returns
        -------
        hp_map : `np.ndarray`
            Output HEALPix map with the requested resolution.
        """
        # If no nside is passed, we generate a map with the same resolution as the original
        if nside is None:
            nside = self._nside_sparse

        if self._is_rec_array:
            if key is None:
                raise ValueError('key should be specified for HealSparseMaps including `recarray`')
            else:
                # This is memory inefficient in that we are copying the memory
                # to ensure that we get a unique healpix map.  To not get a copy,
                # you can do map['column'][:]
                single_map = self.get_single(key, copy=True)
        elif self._is_wide_mask:
            raise NotImplementedError("Cannot make healpix map out of wide_mask")
        else:
            single_map = self

        # If we're degrading, let that code do the datatyping
        if nside < self._nside_sparse:
            # degrade to new resolution
            single_map = single_map.degrade(nside, reduction=reduction)
        elif nside > self._nside_sparse:
            raise ValueError("Cannot generate HEALPix map with higher resolution than the original.")

        # Check to see if we have an integer map.
        if issubclass(single_map._sparse_map.dtype.type, np.integer):
            dtypeOut = np.float64
        else:
            dtypeOut = single_map._sparse_map.dtype

        # Create an empty HEALPix map, filled with UNSEEN values
        hp_map = np.full(hpg.nside_to_npixel(nside), hpg.UNSEEN, dtype=dtypeOut)

        valid_pixels = single_map.valid_pixels
        if not nest:
            valid_pixels = hpg.nest_to_ring(nside, valid_pixels)
        hp_map[valid_pixels] = single_map.get_values_pix(valid_pixels, nest=nest)

        return hp_map

    @property
    def valid_pixels(self):
        """
        Get an array of valid pixels in the sparse map.

        Returns
        -------
        valid_pixels : `np.ndarray`
        """
        if self._is_rec_array:
            valid_pixel_inds, = np.where(self._sparse_map[self._primary] != self._sentinel)
        elif self._is_wide_mask:
            valid_pixel_inds, = np.where(np.any(self._sparse_map != self._sentinel, axis=1))
        elif self._is_bit_packed:
            # This is dangerous because it expands into a full array first; this
            # can blow up memory.
            valid_pixel_inds, = np.where(np.array(self._sparse_map) != self._sentinel)
        else:
            valid_pixel_inds, = np.where(self._sparse_map != self._sentinel)

        return valid_pixel_inds - self._cov_map[self._cov_map.cov_pixels_from_index(valid_pixel_inds)]

    def valid_pixels_pos(self, lonlat=True, return_pixels=False):
        """
        Get an array with the position of valid pixels in the sparse map.

        Parameters
        ----------
        lonlat: `bool`, optional
            If True, input angles are longitude and latitude in degrees.
            Otherwise, they are co-latitude and longitude in radians.
        return_pixels: `bool`, optional
            If true, return valid_pixels / co-lat / co-lon or
            valid_pixels / lat / lon instead of lat / lon

        Returns
        -------
        positions : `tuple`
            By default it will return a tuple of the form (`theta`, `phi`) in radians
            unless `lonlat = True`, for which it will return (`ra`, `dec`) in degrees.
            If `return_pixels = True`, valid_pixels will be returned as first element
            in tuple.
        """
        if return_pixels:
            valid_pixels = self.valid_pixels
            lon, lat = hpg.pixel_to_angle(self.nside_sparse, valid_pixels, lonlat=lonlat)
            return (valid_pixels, lon, lat)
        else:
            return hpg.pixel_to_angle(self.nside_sparse, self.valid_pixels, lonlat=lonlat)

    @property
    def n_valid(self):
        """
        Get the number of valid pixels in the map.

        Returns
        -------
        n_valid : `int`
        """
        if self._n_valid is not None:
            # This has been cached.
            return self._n_valid

        # This is more memory efficient to work with bits rather than
        # integer indices.
        if self._is_rec_array:
            n_valid = np.sum(self._sparse_map[self._primary] != self._sentinel)
        elif self._is_wide_mask:
            n_valid = np.sum(np.any(self._sparse_map != self._sentinel, axis=1))
        elif self._is_bit_packed:
            # TODO: This will need to be updated if we allow True sentinel.
            n_valid = self._sparse_map.sum()
        else:
            n_valid = np.sum(self._sparse_map != self._sentinel)

        self._n_valid = n_valid
        return n_valid

    def iter_valid_pixels_by_covpix(self):
        """
        Generator to get valid_pixels associated with each coverage pixel,
        yielded one coverage pixel at a time.

        Yields
        ------
        single_pixel_valid_pixels : `np.ndarray`

        Examples
        --------
        >>> for valid_pixels in m.iter_valid_pixels_by_covpix():
        ...     print(valid_pixels)
        """
        cov_pixels, = np.where(self._cov_map.coverage_mask)

        for cov_pix in cov_pixels:
            yield self.valid_pixels_single_covpix(cov_pix)

    def valid_pixels_single_covpix(self, cov_pix):
        """Get an array with the valid pixels in a single coverage pixel.

        This uses much less memory than a full valid_pixels list for large
        maps.

        Parameters
        ----------
        cov_pix : `int`
            Coverage pixel to get valid pixels.

        Returns
        -------
        valid_pixels : `np.ndarray`
            Array of valid pixels in the given coverage pixel.
        """
        # Check if this is in the coverage mask.
        if not self.coverage_mask[cov_pix]:
            return np.array([], dtype=np.int64)

        # This is the start of the coverage pixel slice.
        start = (self._cov_map[cov_pix] +
                 self._cov_map.nfine_per_cov*cov_pix)
        s = slice(start, start + self._cov_map.nfine_per_cov)

        if self._is_rec_array:
            valid_pixel_inds, = np.where(self._sparse_map[self._primary][s] != self._sentinel)
        elif self._is_wide_mask:
            valid_pixel_inds, = np.where(np.any(self._sparse_map[s, :] != self._sentinel, axis=1))
        elif self._is_bit_packed:
            valid_pixel_inds, = np.where(np.array(self._sparse_map[s]) != self._sentinel)
        else:
            valid_pixel_inds, = np.where(self._sparse_map[s] != self._sentinel)

        # We need to get the correct offsets for our valid pixel subset.
        return (valid_pixel_inds -
                self._cov_map[self._cov_map.cov_pixels_from_index(start)] +
                start)

    def get_valid_area(self, degrees=True):
        """
        Get the area covered by valid pixels

        Parameters
        ----------
        degrees : `bool` If True (default) returns the area in square degrees,
        if False it returns the area in steradians

        Returns
        -------
        valid_area : `float`
        """
        return self.n_valid*hpg.nside_to_pixel_area(self._nside_sparse, degrees=degrees)

    def _degrade(self, nside_out, reduction='mean', weights=None):
        """
        Auxiliary method to reduce the resolution, i.e., increase the pixel size
        of a given sparse map (which is called by `degrade`).

        Parameters
        ----------
        nside_out : `int`
           Output Nside resolution parameter.
        reduction : `str`
           Reduction method (mean, median, std, max, min, and, or, sum, prod, wmean).
        weights : `healSparseMap`
           If the reduction is `wmean` this is the map with the weights to use.
           It should have the same characteristics as the original map.

        Returns
        -------
        healSparseMap : `HealSparseMap`
           New map, at the desired resolution.
        """
        if self._nside_sparse < nside_out:
            raise ValueError('nside_out should be smaller than nside for the sparse_map.')
        # Count the number of filled pixels in the coverage mask
        npop_pix = np.count_nonzero(self.coverage_mask)
        # We need the new bit_shifts and we have to build a new CovIndexMap
        bit_shift = _compute_bitshift(self.nside_coverage, nside_out)
        nfine_per_cov = 2**bit_shift

        # Check weights and add guards
        weight_values = None
        if weights is not None:
            if reduction != 'wmean':
                warnings.warn('Weights only used with wmean reduction.  Ignoring weights.',
                              UserWarning)
            else:
                # Check format/size of weight-map here.
                if not isinstance(weights, HealSparseMap):
                    raise ValueError("weights must be a HealSparseMap.")
                if weights.is_rec_array or weights.is_wide_mask_map or weights.is_integer_map:
                    raise ValueError("weights must be a floating-point map.")
                bad_map = ((weights.nside_sparse != self.nside_sparse) or
                           (weights.nside_coverage != self.nside_coverage) or
                           (not np.array_equal(weights.valid_pixels, self.valid_pixels)))
                if bad_map:
                    raise ValueError('weights dimensions must be the same as this map.')

                weight_values = weights._sparse_map
                # Set to zero weight those pixels that are not observed
                # This is valid for all types of maps because they share the same valid_pixels.
                weight_values[weight_values == weights._sentinel] = 0.0
                weight_values = weight_values.reshape((npop_pix + 1,
                                                       (nside_out//self.nside_coverage)**2, -1))
        elif reduction == 'wmean':
            raise ValueError('Must specify weights when using wmean reduction.')
        # At this point, the weight map has been checked and will only be used if
        # the reduction is set to wmean.

        # Work with wide masks
        if self._is_wide_mask:
            if reduction not in ['and', 'or']:
                raise NotImplementedError('Cannot degrade a wide_mask map with this \
                reduction operation, try and/or.')
            else:
                nbits = self._sparse_map.shape[1]
                aux = self._sparse_map.reshape((npop_pix+1, (nside_out//self.nside_coverage)**2, -1, nbits))
                sparse_map_out = reduce_array(aux, reduction=reduction, axis=2).reshape((-1, nbits))
                sentinel_out = self._sentinel

        # Work with RecArray (we have to change the resolution to all maps...)
        elif self._is_rec_array:
            dtype = []
            sentinel_out = hpg.UNSEEN
            # We should avoid integers
            for key, value in self._sparse_map.dtype.fields.items():
                if issubclass(self._sparse_map[key].dtype.type, np.integer):
                    dtype.append((key, np.float64))
                else:
                    dtype.append((key, value[0]))
            # Allocate new map
            sparse_map_out = np.zeros((npop_pix + 1)*nfine_per_cov, dtype=dtype)
            for key, value in sparse_map_out.dtype.fields.items():
                aux = self._sparse_map[key].astype(np.float64)
                aux[self._sparse_map[self._primary] == self._sentinel] = np.nan
                aux = aux.reshape((npop_pix + 1, (nside_out//self.nside_coverage)**2, -1))
                # Perform the reduction operation (check utils.reduce_array)
                aux = reduce_array(aux, reduction=reduction, weights=weight_values)
                # Transform back to sentinel value
                aux[np.isnan(aux)] = sentinel_out
                sparse_map_out[key] = aux

        # Work with int array and ndarray
        elif (issubclass(self._sparse_map.dtype.type, np.integer)) and (reduction in ['and', 'or']):
            aux = self._sparse_map.reshape((npop_pix+1, (nside_out//self.nside_coverage)**2, -1))
            sparse_map_out = reduce_array(aux, reduction=reduction)
            sentinel_out = self._sentinel
        else:
            if issubclass(self._sparse_map.dtype.type, (np.integer, np.bool_)):
                aux_dtype = np.float64
            else:
                aux_dtype = self._sparse_map.dtype
            sentinel_out = hpg.UNSEEN
            aux = self._sparse_map.astype(aux_dtype)
            aux[self._sparse_map == self._sentinel] = np.nan
            aux = aux.reshape((npop_pix + 1, (nside_out//self.nside_coverage)**2, -1))
            aux = reduce_array(aux, reduction=reduction, weights=weight_values)
            # NaN are converted to UNSEEN
            aux[np.isnan(aux)] = sentinel_out
            sparse_map_out = aux

        # The coverage index map is now offset, we have to build a new one
        # Note that we need to keep the same order of the coverage map
        new_cov_map = HealSparseCoverage.make_from_pixels(self.nside_coverage,
                                                          nside_out,
                                                          self._cov_map._block_to_cov_index)
        return HealSparseMap(cov_map=new_cov_map, sparse_map=sparse_map_out,
                             nside_sparse=nside_out, primary=self._primary, sentinel=sentinel_out)

    def degrade(self, nside_out, reduction='mean', weights=None):
        """
        Decrease the resolution of the map, i.e., increase the pixel size.

        Parameters
        ----------
        nside_out : `int`
            Output nside resolution parameter.
        reduction : `str`, optional
            Reduction method (mean, median, std, max, min, and, or, sum, prod, wmean).
        weights : `HealSparseMap`, optional
            If the reduction is `wmean` this is the map with the weights to use.
            It should have the same characteristics as the original map.

        Returns
        -------
        healSparseMap : `HealSparseMap`
           New map, at the desired resolution.
        """
        if nside_out > self._nside_sparse:
            raise ValueError("To increase the resolution of the map, use ``upgrade``.")

        if self._is_bit_packed:
            raise NotImplementedError("Map degrading is not yet supported for bit_packed maps.")

        if nside_out < self.nside_coverage:
            # The way we do the reduction requires nside_out to be >= nside_coverage
            # we allocate a new map with the required nside_out
            # CAUTION: This may require a lot of memory!!
            warnings.warn("`nside_out` < `nside_coverage`. \
                            Allocating new map with nside_coverage=nside_out",
                          ResourceWarning)
            sparse_map_out = HealSparseMap.make_empty_like(self,
                                                           nside_coverage=nside_out)
            if weights is not None:
                wgt_valid = weights.valid_pixels
                _weights = HealSparseMap.make_empty_like(weights, nside_coverage=nside_out)
                _weights[wgt_valid] = weights[wgt_valid]
                weights = _weights
            valid_pixels = self.valid_pixels
            sparse_map_out[valid_pixels] = self[valid_pixels]
            sparse_map_out = sparse_map_out._degrade(nside_out, reduction=reduction, weights=weights)
        else:
            if self._nside_sparse == nside_out:
                sparse_map_out = self
            else:
                # Regular degrade
                sparse_map_out = self._degrade(nside_out,
                                               reduction=reduction,
                                               weights=weights)

        return sparse_map_out

    def upgrade(self, nside_out):
        """
        Increase the resolution of the map, i.e., decrease the pixel size.

        All covering pixels will be duplicated at the higher resolution.

        Parameters
        ----------
        nside_out : `int`
            Output nside resolution parameter.

        Returns
        -------
        healSparseMap : `HealSparseMap`
            New map, at the desired resolution.
        """
        if self._nside_sparse >= nside_out:
            raise ValueError("To decrease the resolution of the map, use ``degrade``.")

        if self._is_wide_mask:
            raise NotImplementedError("Upgrading wide masks is not supported.")
        elif self._is_bit_packed:
            raise NotImplementedError("Upgrading bit_packed maps is not yet supported.")

        # Make an order preserving coverage map.
        new_cov_map = HealSparseCoverage.make_from_pixels(self.nside_coverage,
                                                          nside_out,
                                                          self._cov_map._block_to_cov_index)
        # And a new sparse map
        bit_shift = _compute_bitshift(self._nside_sparse, nside_out)
        nout_per_self = 2**bit_shift
        # Nest maps at higher resolution are just repeats of the same values
        new_sparse_map = np.repeat(self._sparse_map, nout_per_self)

        return HealSparseMap(cov_map=new_cov_map, sparse_map=new_sparse_map,
                             nside_sparse=nside_out, primary=self._primary,
                             sentinel=self._sentinel)

    def apply_mask(self, mask_map, mask_bits=None, mask_bit_arr=None, in_place=True):
        """
        Apply an integer mask to the map.  All pixels in the integer
        mask that have any bits in mask_bits set will be zeroed in the
        output map.  The default is that this operation will be done
        in place, but it may be set to return a copy with a masked map.

        Parameters
        ----------
        mask_map : `HealSparseMap`
           Integer mask to apply to the map.
        mask_bits : `int`, optional
           Bits to be treated as bad in the mask_map.
           Default is None (all non-zero pixels are masked)
        mask_bit_arr : `list` or `np.ndarray`, optional
           Array of bit values, used if mask_map is a wide_mask_map.
        in_place : `bool`, optional
           Apply operation in place.  Default is True

        Returns
        -------
        masked_map : `HealSparseMap`
           self if in_place is True, a new copy otherwise
        """

        # Check that the mask_map is an integer map (and not a recArray)
        if not mask_map.is_integer_map:
            raise RuntimeError("Can only apply a mask_map that is an integer map.")
        if mask_bits is not None and mask_map.is_wide_mask_map:
            raise RuntimeError("Cannot use mask_bits with wide_mask_map.")

        # operate on this map valid_pixels
        valid_pixels = self.valid_pixels

        if mask_bits is None:
            if mask_map.is_wide_mask_map:
                if mask_bit_arr is None:
                    bad_pixels, = np.where(mask_map.get_values_pix(valid_pixels).sum(axis=1) > 0)
                else:
                    mask_values = mask_map.get_values_pix(valid_pixels)

                    bit_value = _bitvals_to_packed_array(mask_bit_arr, mask_map._wide_mask_maxbits)
                    bad_pixels, = np.where(np.any((mask_values & bit_value) > 0, axis=1))
            else:
                bad_pixels, = np.where(mask_map.get_values_pix(valid_pixels) > 0)
        else:
            bad_pixels, = np.where((mask_map.get_values_pix(valid_pixels) & mask_bits) > 0)

        if in_place:
            new_map = self
        else:
            new_map = HealSparseMap(cov_map=self._cov_map.copy(),
                                    sparse_map=self._sparse_map.copy(),
                                    nside_sparse=self._nside_sparse,
                                    primary=self._primary,
                                    sentinel=self._sentinel)

        new_value = new_map._sparse_map[0]

        ipnest_cov = self._cov_map.cov_pixels(valid_pixels[bad_pixels])
        new_map._sparse_map[valid_pixels[bad_pixels] + new_map._cov_map[ipnest_cov]] = new_value

        return new_map

    def interpolate_pos(self, ra_or_theta, dec_or_phi, lonlat=True, allow_partial=False):
        """
        Return the bilinear interpolation of the map using 4 nearest neighbors.

        Parameters
        ----------
        ra_or_theta : `float`, array-like
            Angular coordinates of points on a sphere.
        dec_or_phi : `float`, array-like
            Angular coordinates of points on a sphere.
        lonlat : `bool`, optional
            If True, input angles are longitude and latitude in degrees.
            Otherwise, they are co-latitude and longitude in radians.
        allow_partial : `bool`, optional
            If this is True, then unseen (not validvalid) neighbors will be
            ignored and the output value will be the weighted average of the
            valid neighbors. Otherwise, if any neighbor is not valid then
            the interpolated value will be set to UNSEEN.

        Returns
        -------
        values : `np.ndarray`
            Array of interpolated values corresponding to input positions.
            The return array will always be 64-bit floats.

        Notes
        -----
        The interpolation routing works only on numeric data, and not on wide
        mask maps, recarray maps, or boolean maps.
        """
        if self._is_wide_mask:
            raise NotImplementedError("Interpolation does not run on a wide mask map.")
        elif self._is_rec_array:
            raise NotImplementedError("Interpolation does not run on a recarray map.")
        elif isinstance(self._sentinel, bool):
            raise NotImplementedError("Interpolation does not run on a boolean map.")

        interp_pix, interp_wgt = hpg.get_interpolation_weights(
            self.nside_sparse,
            np.atleast_1d(ra_or_theta),
            np.atleast_1d(dec_or_phi),
            lonlat=lonlat,
        )
        aux = self.get_values_pix(interp_pix)
        out_of_bounds = (aux == self._sentinel)
        aux = aux.astype(np.float64)
        aux[out_of_bounds] = np.nan

        if not allow_partial:
            # Any pixel that has an out-of-bounds neighbor will be set to UNSEEN.
            values = np.nansum(aux * interp_wgt, axis=1) / np.sum(interp_wgt, axis=1)
            values[~np.all(~out_of_bounds, axis=1)] = hpg.UNSEEN
        else:
            # Use only the neighbor pixels that are valid.
            interp_wgt[out_of_bounds] = np.nan
            wgt_sum = np.nansum(interp_wgt, axis=1)
            values = np.nansum(aux * interp_wgt, axis=1)
            all_bad = (wgt_sum == 0.0)
            values[~all_bad] /= wgt_sum[~all_bad]
            # Any pixel that has all bad neighbors will be UNSEEN.
            values[all_bad] = hpg.UNSEEN

        return values

    def __getitem__(self, key):
        """
        Get part of a healpix map.
        """
        if isinstance(key, str):
            if not self._is_rec_array:
                raise IndexError("HealSparseMap is not a recarray map, cannot use string index.")
            return self.get_single(key, sentinel=None)
        elif isinstance(key, numbers.Integral):
            # Get a single pixel
            # Return a single (non-array) value
            return self.get_values_pix(np.array([key]))[0]
        elif isinstance(key, slice):
            # Get a slice of pixels
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else hpg.nside_to_npixel(self._nside_sparse)
            step = key.step if key.step is not None else 1
            return self.get_values_pix(np.arange(start, stop, step))
        elif isinstance(key, np.ndarray):
            # Make sure that it's integers
            test_value = np.zeros(1, key.dtype)[0]
            if not is_integer_value(test_value):
                raise IndexError("Numpy array indices must be integers for __getitem__")
            return self.get_values_pix(key)
        elif isinstance(key, list):
            # Make sure that it's integers
            arr = np.atleast_1d(key)
            if len(arr) > 0:
                if not is_integer_value(arr[0]):
                    raise IndexError("List array indices must be integers for __getitem__")
            return self.get_values_pix(arr)
        else:
            raise IndexError("Illegal index type (%s) for __getitem__ in HealSparseMap." %
                             (key.__class__))

    def __setitem__(self, key, value):
        """
        Set part of a healpix map
        """
        if isinstance(key, numbers.Integral):
            # Set a single pixel
            return self.update_values_pix(np.array([key]), value)
        elif isinstance(key, slice):
            # Set a slice of pixels
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else hpg.nside_to_npixel(self._nside_sparse)
            step = key.step if key.step is not None else 1
            return self.update_values_pix(np.arange(start, stop, step),
                                          value)
        elif isinstance(key, np.ndarray):
            test_value = np.zeros(1, key.dtype)[0]
            if not is_integer_value(test_value):
                raise IndexError("Numpy array indices must be integers for __setitem__")
            return self.update_values_pix(key, value)
        elif isinstance(key, list):
            arr = np.atleast_1d(key)
            if len(arr) > 0 and not is_integer_value(arr[0]):
                raise IndexError("List/Tuple array indices must be integers for __setitem__")
            return self.update_values_pix(arr, value)
        else:
            raise IndexError("Illegal index type (%s) for __setitem__ in HealSparseMap." %
                             (key.__class__))

    def get_single(self, key, sentinel=None, copy=False):
        """
        Get a single healsparse map out of a recarray map, with the ability to
        override a sentinel value.

        Parameters
        ----------
        key : `str`
           Field for the recarray
        sentinel : `int` or `float` or None, optional
           Override the default sentinel value.  Default is None (use default)

        Returns
        -------
        single_map : `HealSparseMap`
        """
        if not self._is_rec_array:
            raise TypeError("HealSparseMap is not a recarray map")

        # If we are the primary key, use the sentinel as set.  Otherwise,
        # use the default sentinel unless otherwise overridden.
        if key == self._primary:
            _sentinel = check_sentinel(self._sparse_map[key].dtype.type, self._sentinel)
        else:
            _sentinel = check_sentinel(self._sparse_map[key].dtype.type, sentinel)

        if not copy:
            # This will not copy memory which allows in-recarray assignment.
            # Problems can potentially happen with mixed type recarrays depending
            # on how they were constructed (though using make_empty should be safe).
            # However, these linked maps cannot be used to add new pixels which
            # is why there is the _is_view flag.
            return HealSparseMap(cov_map=self._cov_map,
                                 sparse_map=self._sparse_map[key],
                                 nside_sparse=self._nside_sparse, sentinel=_sentinel,
                                 _is_view=True)

        new_sparse_map = np.full_like(self._sparse_map[key], _sentinel)

        valid_indices = (self._sparse_map[self._primary] != self._sentinel)
        new_sparse_map[valid_indices] = self._sparse_map[key][valid_indices]

        return HealSparseMap(cov_map=self._cov_map, sparse_map=new_sparse_map,
                             nside_sparse=self._nside_sparse, sentinel=_sentinel)

    def get_single_covpix_map(self, covpix):
        """
        Get a healsparse map for a single coverage pixel.

        Note that this makes a copy of the data.

        Parameters
        ----------
        covpix : `int`
            Coverage pixel to copy

        Returns
        -------
        single_pixel_map : `HealSparseMap`
            Copy of map with a single coverage pixel.
        """
        nfine_per_cov = self._cov_map._nfine_per_cov

        if self._cov_map[covpix] + covpix*nfine_per_cov < nfine_per_cov:
            # Pixel is not in the coverage map; return an empty map
            return HealSparseMap.make_empty_like(self)

        new_cov_map = HealSparseCoverage.make_from_pixels(self.nside_coverage,
                                                          self._nside_sparse,
                                                          [covpix])
        if self._is_wide_mask:
            new_sparse_map = np.zeros((2*nfine_per_cov, self._wide_mask_width), dtype=self.dtype)
            # Copy overflow bin
            new_sparse_map[0: nfine_per_cov, :] = self._sparse_map[0: nfine_per_cov, :]
            # Copy the pixel
            new_sparse_map[nfine_per_cov: 2*nfine_per_cov, :] = self._sparse_map[
                self._cov_map[covpix] + covpix*nfine_per_cov:
                self._cov_map[covpix] + covpix*nfine_per_cov + nfine_per_cov, :]
        else:
            new_sparse_map = np.zeros(2*nfine_per_cov, dtype=self.dtype)
            # Copy overflow bin
            new_sparse_map[0: nfine_per_cov] = self._sparse_map[0: nfine_per_cov]
            # Copy the pixel
            new_sparse_map[nfine_per_cov: 2*nfine_per_cov] = self._sparse_map[
                self._cov_map[covpix] + covpix*nfine_per_cov:
                self._cov_map[covpix] + covpix*nfine_per_cov + nfine_per_cov]

        return HealSparseMap(cov_map=new_cov_map, sparse_map=new_sparse_map,
                             nside_sparse=self._nside_sparse, primary=self._primary,
                             sentinel=self._sentinel)

    def get_covpix_maps(self):
        """
        Generator to get individual maps associated with each coverage pixel,
        yielded one coverage pixel at a time.

        This routine makes a copy of the data for each individual coverage
        pixel map.

        Yields
        ------
        single_pixel_map : `HealSparseMap`

        Examples
        --------
        >>> for covpix_map in m.get_covpix_maps():
        ...     print(covpix_map.valid_pixels)
        """
        cov_pixels, = np.where(self._cov_map.coverage_mask)

        for cov_pix in cov_pixels:
            yield self.get_single_covpix_map(cov_pix)

    def astype(self, dtype, sentinel=None):
        """
        Convert sparse map to a different numpy datatype, including sentinel
        values.  If sentinel is not specified the default for the converted
        datatype is used (`UNSEEN` for float, and -MAXINT for ints).

        Parameters
        ----------
        dtype : `numpy.dtype`
            Valid numpy dtype for a single array.
        sentinel : `int` or `float`, optional
            Converted map sentinel value.

        Returns
        -------
        sparse_map : `HealSparseMap`
            New map with new data type.
        """
        if self._is_rec_array:
            raise RuntimeError("Cannot convert datatype of a recarray map.")
        elif self._is_wide_mask:
            raise RuntimeError("Cannot convert datatype of a wide mask.")

        new_sparse_map = np.zeros(self._sparse_map.shape, dtype=dtype)
        valid_pix = (self._sparse_map != self._sentinel)
        new_sparse_map[valid_pix] = self._sparse_map[valid_pix].astype(dtype)

        _sentinel = check_sentinel(new_sparse_map.dtype.type, sentinel)
        new_sparse_map[~valid_pix] = _sentinel

        return HealSparseMap(cov_map=self._cov_map, sparse_map=new_sparse_map,
                             nside_sparse=self.nside_sparse, sentinel=_sentinel)

    def as_bit_packed_map(self):
        """
        Convert map to a bit-packed mask map.

        This only maintains information on valid pixels, which will be True in
        the bit array mask.

        Returns
        -------
        bit_packed_map : `HealSparseMap`
        """
        if self._is_bit_packed:
            return self.copy()

        if (self._cov_map.nfine_per_cov % 8) != 0:
            raise ValueError("Can only create a bit_packed mask map at least two "
                             "healpix levels between coverage and mask.")

        # Need to go through coverage pixels, and copy the data into the new thing.
        # There is some fancy indexing that has to happen here.
        # The size will be the number of coverage pixels + 1 times nside
        coverage_pixels, = np.where(self.coverage_mask)
        n_cov = len(coverage_pixels)

        bitmask_map = _PackedBoolArray(size=(n_cov + 1)*self._cov_map.nfine_per_cov)

        # This is the map without the offset.
        cov_index_map_temp = self._cov_map[:] + np.arange(hpg.nside_to_npixel(self._cov_map.nside_coverage),
                                                          dtype=np.int64)*self._cov_map.nfine_per_cov

        for cov_pix in coverage_pixels:
            s = slice(cov_index_map_temp[cov_pix], cov_index_map_temp[cov_pix] + self._cov_map.nfine_per_cov)
            if self._is_rec_array:
                bool_data = (self._sparse_map[self._primary][s] != self._sentinel)
            elif self._is_wide_mask:
                bool_data = np.any(self._sparse_map[s] != self._sentinel, axis=1)
            else:
                bool_data = (self._sparse_map[s] != self._sentinel)

            # This is a bulk setter for aligned data.
            bitmask_map[s] = bool_data

        return HealSparseMap(
            cov_map=self._cov_map,
            sparse_map=bitmask_map,
            nside_sparse=self.nside_sparse,
            sentinel=False,
            metadata=self.metadata,
        )

    def __add__(self, other):
        """
        Add a constant.

        Cannot be used with recarray maps.
        """
        if issubclass(type(other), GeomBase):
            new_map = self.copy()
            new_map.update_values_pix(
                other.get_pixel_ranges(nside=self.nside_sparse),
                other.value,
                operation="add",
            )
            return new_map
        else:
            return self._apply_operation(other, np.add)

    def __iadd__(self, other):
        """
        Add a constant, in place.

        Cannot be used with recarray maps.
        """
        if issubclass(type(other), GeomBase):
            self.update_values_pix(
                other.get_pixel_ranges(nside=self.nside_sparse),
                other.value,
                operation="add",
            )
            return self
        else:
            return self._apply_operation(other, np.add, in_place=True)

    def __sub__(self, other):
        """
        Subtract a constant.

        Cannot be used with recarray maps.
        """

        return self._apply_operation(other, np.subtract)

    def __isub__(self, other):
        """
        Subtract a constant, in place.

        Cannot be used with recarray maps.
        """

        return self._apply_operation(other, np.subtract, in_place=True)

    def __mul__(self, other):
        """
        Multiply a constant.

        Cannot be used with recarray maps.
        """

        return self._apply_operation(other, np.multiply)

    def __imul__(self, other):
        """
        Multiply a constant, in place.

        Cannot be used with recarray maps.
        """

        return self._apply_operation(other, np.multiply, in_place=True)

    def __truediv__(self, other):
        """
        Divide a constant.

        Cannot be used with recarray maps.
        """

        return self._apply_operation(other, np.divide)

    def __itruediv__(self, other):
        """
        Divide a constant, in place.

        Cannot be used with recarray maps.
        """

        return self._apply_operation(other, np.divide, in_place=True)

    def __pow__(self, other):
        """
        Raise the map to a power.

        Cannot be used with recarray maps.
        """

        return self._apply_operation(other, np.power)

    def __ipow__(self, other):
        """
        Divide a constant, in place.

        Cannot be used with recarray maps.
        """

        return self._apply_operation(other, np.power, in_place=True)

    def __and__(self, other):
        """
        Perform a bitwise and with a constant.

        Cannot be used with recarray maps.
        """
        if issubclass(type(other), GeomBase):
            new_map = self.copy()
            if self._is_wide_mask:
                value = _bitvals_to_packed_array(other.value, self._wide_mask_maxbits)
            else:
                value = other.value
            new_map.update_values_pix(
                other.get_pixel_ranges(nside=self.nside_sparse),
                value,
                operation="and",
            )
            return new_map
        elif self.dtype == np.bool_:
            return self._apply_boolean_map_operation(other, "and")
        else:
            return self._apply_operation(other, np.bitwise_and, int_only=True)

    def __iand__(self, other):
        """
        Perform a bitwise and with a constant, in place.

        Cannot be used with recarray maps.
        """
        if issubclass(type(other), GeomBase):
            if self._is_wide_mask:
                value = _bitvals_to_packed_array(other.value, self._wide_mask_maxbits)
            else:
                value = other.value
            self.update_values_pix(
                other.get_pixel_ranges(nside=self.nside_sparse),
                value,
                operation="and",
            )
            return self
        elif self.dtype == np.bool_:
            return self._apply_boolean_map_operation(other, "and", in_place=True)
        else:
            return self._apply_operation(other, np.bitwise_and, int_only=True, in_place=True)

    def __xor__(self, other):
        """
        Perform a bitwise xor with a constant.

        Cannot be used with recarray maps.
        """
        if self.dtype == np.bool_:
            return self._apply_boolean_map_operation(other, "xor")
        else:
            return self._apply_operation(other, np.bitwise_xor, int_only=True)

    def __ixor__(self, other):
        """
        Perform a bitwise xor with a constant, in place.

        Cannot be used with recarray maps.
        """
        if self.dtype == np.bool_:
            return self._apply_boolean_map_operation(other, "xor", in_place=True)
        else:
            return self._apply_operation(other, np.bitwise_xor, int_only=True, in_place=True)

    def __or__(self, other):
        """
        Perform a bitwise or with a constant.

        Cannot be used with recarray maps.
        """
        if issubclass(type(other), GeomBase):
            new_map = self.copy()
            if self._is_wide_mask:
                value = _bitvals_to_packed_array(other.value, self._wide_mask_maxbits)
            else:
                value = other.value
            new_map.update_values_pix(
                other.get_pixel_ranges(nside=self.nside_sparse),
                value,
                operation="or",
            )
            return new_map
        elif self.dtype == np.bool_:
            return self._apply_boolean_map_operation(other, "or")
        else:
            return self._apply_operation(other, np.bitwise_or, int_only=True)

    def __ior__(self, other):
        """
        Perform a bitwise or with a constant, in place.

        Cannot be used with recarray maps.
        """
        if issubclass(type(other), GeomBase):
            if self._is_wide_mask:
                value = _bitvals_to_packed_array(other.value, self._wide_mask_maxbits)
            else:
                value = other.value
            self.update_values_pix(
                other.get_pixel_ranges(nside=self.nside_sparse),
                value,
                operation="or",
            )
            return self
        elif self.dtype == np.bool_:
            return self._apply_boolean_map_operation(other, "or", in_place=True)
        else:
            return self._apply_operation(other, np.bitwise_or, int_only=True, in_place=True)

    def invert(self):
        """Perform a bitwise inversion, over the coverage pixels, in place.
        """
        if self.dtype != np.bool_:
            raise NotImplementedError("Can only use invert(~) on boolean maps.")

        # We invalidate the n_valid cache here.
        self._n_valid = None

        self._sparse_map[self._cov_map.nfine_per_cov:] = ~self._sparse_map[self._cov_map.nfine_per_cov:]
        return self

    def __invert__(self):
        """
        Perform a bit inversion, over the coverage pixels.

        Only available on boolean maps.
        """
        if self.dtype != np.bool_:
            raise NotImplementedError("Can only use invert(~) on boolean maps.")

        sparse_map_temp = self._sparse_map.copy()
        sparse_map_temp[self._cov_map.nfine_per_cov:] = ~sparse_map_temp[self._cov_map.nfine_per_cov:]
        return HealSparseMap(
            cov_map=self._cov_map.copy(),
            sparse_map=sparse_map_temp,
            nside_sparse=self._nside_sparse,
            sentinel=self._sentinel,
        )

    def _apply_operation(self, other, func, int_only=False, in_place=False):
        """
        Apply a generic arithmetic function.

        Cannot be used with recarray maps.

        Parameters
        ----------
        other : `int` or `float` (or numpy equivalents)
            The other item to perform the operator on.
        func : `np.ufunc`
            The numpy universal function to apply.
        int_only : `bool`, optional
            Only accept integer types.
        in_place : `bool`, optional
            Perform operation in-place.

        Returns
        -------
        result : `HealSparseMap`
            Resulting map
        """
        name = func.__str__()

        if self._is_rec_array:
            raise NotImplementedError("Cannot use %s with recarray maps" % (name))
        if self.dtype == np.bool_:
            raise NotImplementedError("Cannot ue %s with boolean maps" % (name))
        if int_only:
            if not self.is_integer_map:
                raise NotImplementedError("Can only apply %s to integer maps" % (name))
        else:
            # If not int_only then it can't be used with a wide mask.
            if self._is_wide_mask:
                raise NotImplementedError("Cannot use %s with wide mask maps" % (name))

        if in_place:
            # We invalidate the n_valid cache here.
            self._n_valid = None

        other_int = False
        other_float = False
        other_bits = False

        if isinstance(other, numbers.Integral):
            other_int = True
        elif isinstance(other, numbers.Real):
            other_float = True
        elif isinstance(other, (tuple, list)):
            if not self._is_wide_mask:
                raise NotImplementedError("Must use a wide mask to operate with a bit list")
            other_bits = True
            for elt in other:
                if not isinstance(elt, numbers.Integral):
                    raise NotImplementedError("Can only use an integer list of bits "
                                              "with %s operation" % (name))
            if np.max(other) >= self._wide_mask_maxbits:
                raise ValueError("Bit position %d too large (>= %d)" % (np.max(other),
                                                                        self._wide_mask_maxbits))

        if self._is_wide_mask:
            if not other_bits:
                raise NotImplementedError("Must use a bit list with the %s operation with "
                                          "a wide mask" % (name))
        else:
            if not other_int and not other_float:
                raise NotImplementedError("Can only use a constant with the %s operation" % (name))
            if not other_int and int_only:
                raise NotImplementedError("Can only use an integer constant with the %s operation" % (name))

        if self._is_wide_mask:
            valid_sparse_pixels = (self._sparse_map != self._sentinel).sum(axis=1, dtype=np.bool_)

            other_value = _bitvals_to_packed_array(other, self._wide_mask_maxbits)
        else:
            valid_sparse_pixels = (self._sparse_map != self._sentinel)

        if in_place:
            if self._is_wide_mask:
                for i in range(self._wide_mask_width):
                    col = self._sparse_map[:, i]
                    func(col, other_value[i], out=col, where=valid_sparse_pixels)
            else:
                func(self._sparse_map, other, out=self._sparse_map, where=valid_sparse_pixels)
            return self
        else:
            combinedSparseMap = self._sparse_map.copy()
            if self._is_wide_mask:
                for i in range(self._wide_mask_width):
                    col = combinedSparseMap[:, i]
                    func(col, other_value[i], out=col, where=valid_sparse_pixels)
            else:
                func(combinedSparseMap, other, out=combinedSparseMap, where=valid_sparse_pixels)
            return HealSparseMap(cov_map=self._cov_map, sparse_map=combinedSparseMap,
                                 nside_sparse=self._nside_sparse, sentinel=self._sentinel)

    def _apply_boolean_map_operation(self, other, name, in_place=False):
        """Apply an operation to a boolean mask map.

        Parameters
        ----------
        other : `int` or `float` (or numpy equivalents)
            The other item to perform the operator on.
        name : `str`
            The name of the operation: ``and``, ``or``, or ``xor``.
        in_place : `bool`, optional
            Perform operation in-place.

        Returns
        -------
        result : `HealSparseMap`
            Resulting map
        """
        if name not in ("and", "or", "xor"):
            raise NotImplementedError("_apply_boolean_map_operation does not support %s" % (name))

        if in_place:
            # We invalidate the n_valid cache here.
            self._n_valid = None

        start = self._cov_map.nfine_per_cov
        if in_place:
            sparse_map_temp = self._sparse_map
        else:
            sparse_map_temp = self._sparse_map.copy()

        cov_map_temp = self._cov_map

        if isinstance(other, (bool, np.bool_)):
            # Do a straight bit operation on all pixels outside the overflow
            # pixel.
            start = self._cov_map.nfine_per_cov
            if name == "and":
                sparse_map_temp[start:] &= other
            elif name == "or":
                sparse_map_temp[start:] |= other
            elif name == "xor":
                sparse_map_temp[start:] ^= other
        elif isinstance(other, HealSparseMap):
            # Do an operation if these are allowed
            if not other.dtype == np.bool_:
                raise NotImplementedError("Can only combine a boolean map with another boolean map.")
            if self.nside_sparse != other.nside_sparse:
                raise NotImplementedError("Boolean map operations only supported between maps with the "
                                          "same nside_sparse.")
            if self.nside_coverage != other.nside_coverage:
                raise NotImplementedError("Boolean map operations only supported between maps with the "
                                          "same nside_coverage.")
            if self.sentinel or other.sentinel:
                raise NotImplementedError("Boolean map operations only supported for maps with "
                                          "False sentinel.")

            # This routine will combine the coverage maps of the two masks.
            # We then loop over coverage pixels in the other map to do the
            # operation.

            coverage_mask = self.coverage_mask | other.coverage_mask
            cov_pixels_combined, = coverage_mask.nonzero()
            cov_pixels_run, = other.coverage_mask.nonzero()

            new_cov_pix, = (coverage_mask & ~self.coverage_mask).nonzero()
            if in_place:
                new_cov_pix, = (coverage_mask & ~self.coverage_mask).nonzero()
                self._reserve_cov_pix(new_cov_pix)
                cov_map_temp = self._cov_map
                sparse_map_temp = self._sparse_map
            else:
                # Extend the coverage pixel map and copy data into new buffer.
                cov_map_temp = self._cov_map.append_pixels(len(self._sparse_map), new_cov_pix, check=False)
                nsparse = (cov_pixels_combined.size + 1)*cov_map_temp.nfine_per_cov
                if self._is_bit_packed:
                    sparse_map_temp = _PackedBoolArray(size=nsparse)
                else:
                    sparse_map_temp = np.zeros(nsparse, dtype=np.bool_)

                sparse_map_temp[0: len(self._sparse_map)] = self._sparse_map[0: len(self._sparse_map)]

            for cov_pixel in cov_pixels_run:
                start_self = self._cov_map[cov_pixel] + cov_pixel*cov_map_temp.nfine_per_cov
                end_self = start_self + cov_map_temp.nfine_per_cov
                start_other = other._cov_map[cov_pixel] + cov_pixel*cov_map_temp.nfine_per_cov
                end_other = start_other + cov_map_temp.nfine_per_cov
                start_temp = cov_map_temp[cov_pixel] + cov_pixel*cov_map_temp.nfine_per_cov
                end_temp = start_temp + cov_map_temp.nfine_per_cov

                # The LHS will be guaranteed to have coverage (from above),
                # and the RHS may point to the overflow bin or data.

                if not in_place:
                    # Need to copy data.
                    sparse_map_temp[start_temp: end_temp] = self._sparse_map[start_self: end_self]

                lhs = sparse_map_temp[start_temp: end_temp]
                if self._is_bit_packed == other._is_bit_packed:
                    # These match, no conversions necessary.
                    rhs = other._sparse_map[start_other: end_other]
                elif self._is_bit_packed and not other._is_bit_packed:
                    # Convert the RHS to a _PackedBoolArray.
                    rhs = _PackedBoolArray.from_boolean_array(other._sparse_map[start_other: end_other])
                elif not self._is_bit_packed and other._is_bit_packed:
                    # Expand the RHS to a regular boolean array.
                    rhs = np.array(other._sparse_map[start_other: end_other])

                if name == "and":
                    lhs &= rhs
                elif name == "or":
                    lhs |= rhs
                elif name == "xor":
                    lhs ^= rhs
        else:
            raise NotImplementedError("Can only use a boolean or a boolean map with operation %s" % (name))

        if in_place:
            return self
        else:
            return HealSparseMap(cov_map=cov_map_temp, sparse_map=sparse_map_temp,
                                 nside_sparse=self._nside_sparse, sentinel=self._sentinel)

    def __copy__(self):
        return HealSparseMap(cov_map=self._cov_map.copy(),
                             sparse_map=self._sparse_map.copy(), nside_sparse=self._nside_sparse,
                             sentinel=self._sentinel, primary=self._primary)

    def copy(self):
        return self.__copy__()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        descr = 'HealSparseMap: nside_coverage = %d, nside_sparse = %d' % (self.nside_coverage,
                                                                           self._nside_sparse)
        if self._is_rec_array:
            descr += ', record array type.\n'
            descr += self._sparse_map.dtype.descr.__str__()
        elif self._is_wide_mask:
            descr += ', %d bit wide mask' % (self._wide_mask_maxbits)
        elif self._is_bit_packed:
            descr += ', boolean bit-packed mask'
        else:
            descr += ', ' + self._sparse_map.dtype.name

        add_n_valid = True
        if self._is_bit_packed and self._n_valid is None:
            # Only plot the number of valid pixels if we have
            # previously computed it. This is to keep the memory
            # from blowing up until we optimize this operation.
            add_n_valid = False

        if add_n_valid:
            descr += ', %d valid pixels' % (self.n_valid)

        return descr
