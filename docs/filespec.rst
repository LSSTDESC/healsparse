.. role:: python(code)
   :language: python

HealSparseMap File Specification v1.8.0
=======================================

A :code:`HealSparseMap` file can be either a standard FITS file or a Parquet dataset.
Each provides a way to record the data from a :code:`HealSparseMap` object for efficient retrieval of individual coverage pixels.
This document describes the memory layout of the key components of the :code:`HealSparseMap` objects, as well as the specific file formats in FITS (:ref:`fits_format`) and Parquet (:ref:`parquet_format`).

HealSparseMap Memory Layout
===========================

A :code:`HealSparseMap` object has two primary components, the coverage map and the sparse map.

.. _terminology:

Terminology
-----------

This is a list of terminology used in a :code:`HealSparseMap` object:

* :code:`nside_sparse`: The `HEALPix` nside for the fine-grained (sparse) map
* :code:`nside_coverage`: The `HEALPix` nside for the coverage map
* :code:`bit_shift`: The number of bits to shift to convert from :code:`nside_sparse` to :code:`nside_coverage` in the `HEALPix` NEST scheme.  :code:`bit_shift = 2*log_2(nside_sparse/nside_coverage)`.
* :code:`valid_pixels`: The list of pixels with defined values (:code:`> sentinel`) in the sparse map.
* :code:`sentinel`: The sentinel value that notes if a pixel is not a valid pixel.  Default is :code:`hpgeom.UNSEEN` for floating-point maps, :code:`-MAXINT` for integer maps, and :code:`0` for wide mask maps.
* :code:`nfine_per_cov`: The number of fine (sparse) pixels per coverage pixel.  :code:`nfine_per_cov = 2**bit_shift`.
* :code:`wide_mask_width`: The width of a wide mask, in bytes.

.. _pixel_lookups:

Pixel Lookups
-------------

The :code:`HealSparseMap` file format is derived from the method of encoding fast look-ups of arbitary pixel values using the `HEALPix` nest pixel scheme.

Given a nest-encoded sparse (high resolution) pixel value, :code:`pix_nest`, we can compute the coverage (low resolution) pixel value with a simple bit shift operation: :code:`ipnest_cov = right_shift(pix_nest, bit_shift)`, where :code:`bit_shift` is defined in :ref:`terminology`.

The sparse map itself is stored in blocks of data, each of which includes :code:`nfine_per_cov` contiguous pixels for each coverage pixel that contains valid data.
To find the location *within* a given a coverage block, we need to subtract the first fine (sparse) nest pixel value for the given coverage pixel.
Therefore, :code:`first_pixel = nfine_per_cov*ipnest_cov`.

Next, if we have a map of offsets which points to the location of the proper sparse map block, :code:`cov_map_raw_offset`, we find the look-up index is :code:`index = pix_nest - nfine_per_cov*ipnest_cov + cov_map_raw_offset[ipnest_cov]`.
In practice, we can combine the final terms here.
The look-up index is then :code:`index = pix_nest + cov_map[ipnest_cov]` where :code:`cov_map = cov_map_raw_offset[ipnest_cov] - nfine_per_cov*ipnest_cov`.

As described in :ref:`sparse_map`, the first block in the sparse map is special, and is always filled with :code:`sentinel` values.
All :code:`cov_map` indices for coverage pixels outside the coverage map point to this sentinel block.

.. _coverage_map:

Coverage Map
------------

The coverage map encodes the mapping from raw pixel number to location within the sparse map.
It is an integer (:code:`numpy.int64`) map with :code:`12*nside_coverage*nside_coverage` values, all of which are filled.

As described in :ref:`pixel_lookups`, the coverage map contains indices that are offset pointers to the block in the sparse map with associated values for that coverage pixel.
An empty :code:`HealSparseMap` is intialized with the following coverage pixel values, which all point to the first :code:`sentinel` block in the sparse map.

.. code-block :: python

    import numpy as np
    import hpgeom as hpg

    cov_map[:] = -1*np.arange(hpg.nside_to_npixel(nside_coverage), dtype=np.int64)*nfine_per_cov

.. _sparse_map:

Sparse Map
----------

The sparse map contains the map data, split into blocks, each of which is :code:`nfine_per_cov` elements long.  The first block is special, and is always filled with :code:`sentinel` values.

The following datatypes are allowed:

* 1-byte unsigned integer (:code:`numpy.uint8`)
* 1-byte signed integer (:code:`numpy.int8`)
* 2-byte unsigned integer (:code:`numpy.uint16`)
* 2-byte signed integer (:code:`numpy.int16`)
* 4-byte unsigned integer (:code:`numpy.uint32`)
* 4-byte signed integer (:code:`numpy.uint32`)
* 8-byte signed integer (:code:`numpy.int64`)
* 4-byte floating point (:code:`numpy.float32`)
* 8-byte floating point (:code:`numpy.float64`)
* Numpy record array of numeric types that can be serialized with FITS
* The :code:`WIDE_MASK` special encoding

**Sparse Map Image**

If the sparse map is a single datatype, it is held in memory as a one-dimensional image array.
The first block of :code:`nfine_per_cov` values are set to :code:`sentinel`.
Each additional block of :code:`nfine_per_cov` is associated with a single element in the coverage map.
These blocks may be in any arbitrary order, allowing for easy appending of new coverage pixels.
All invalid pixels must be set to :code:`sentinel`.

**Sparse Map Wide Mask**

If the sparse map is a wide mask map, the sparse map is held in memory as a :code:`wide_mask_width * npix` array.
The sentinel value for wide masks must be :code:`0`, and all invalid pixels must be set to :code:`0`.

**Sparse Map Table**

If the sparse map is a numpy record array type, it is held in memory as a one dimensional table array.
The first block of :code:`nfine_per_cov` values are set such that the :code:`primary` field must be set to :code:`sentinel`.
As with the sparse map image, each additional block of :code:`nfine_per_cov` is associated with a single element in the coverage map.
These blocks may be in any arbitrary order, allowing for easy appending of new coverage pixels.
All invalid pixels must have the :code:`primary` field set to :code:`sentinel`.

**Sparse Map Bit-Packed Mask**

If the sparse map is a bit-packed mask, the sparse map is held in memory as an array of :code:`numpy.uint8`, bit-packed with lowest significant bit (LSB) ordering.
It is this array of :code:`numpy.uint8` that is serialized, with an additional flag in the header.
The sentinel value for sparse bit-packed masks must be :code:`False`.

.. _fits_format:

HealSparseMap FITS Serialization
================================

A :code:`HealSparseMap` FITS file is a standard FITS file with two extensions.
The primary (zeroth) extension is an integer image that describes the coverage map, and the first extension is an image or binary table that describes the sparse map.
This section describes the file format specification of these two extensions in the FITS file.

Coverage Map
------------

**Coverage Map Header**

The coverage map header must contain the following keywords:

* **EXTNAME** must be :code:`"COV"`
* **PIXTYPE** must be :code:`"HEALSPARSE"`
* **NSIDE** is equal to :code:`nside_coverage`

**Coverage Map Image**

The FITS coverage map is a direct serialization of the coverage map image in-memory layout described in :ref:`coverage_map`.

**Sparse Map Header**

The sparse map header must contain:

* **EXTNAME** must be :code:`"SPARSE"`
* **PIXTYPE** must be :code:`"HEALSPARSE"`
* **SENTINEL** is equal to :code:`sentinel`
* **NSIDE** is equal to :code:`nside_sparse`

If the sparse map is a numpy record array, it must contain:

* **PRIMARY** is equal to the name of the "primary" field which defines the valid pixels.

If the sparse map is a wide mask, it must contain:

* **WIDEMASK** must be :code:`True`
* **WWIDTH** must be the width (in bytes) of the wide mask.

If the sparse map is a bit-packed mask it must contain:

* **BITPACK** must be :code:`True`

**Sparse Map Image**

If the sparse map is not of a numpy record array type, it is stored as a one dimensional image array.
If the image is an integer type with 32 bits or fewer, it may be stored with FITS tile compression, with the tile size set to the block size (:code:`nfine_per_cov`).
If the image is a floating-point image, it may be stored with FITS tile compression, with :code:`quantization_level=0` and :code:`GZIP_2` (lossless gzip compression), with the tile size set to the block size (:code:`nfine_per_cov`).

**Sparse Map Wide Mask**

If the sparse map is a wide mask map, the sparse map is stored as a flattened version of the in-memory :code:`wide_mask_width * npix` array.
This should be flattened on storage, and reshaped on read, using the default numpy memory ordering.
The wide mask image may be stored with FITS tile compression, with the tile size set to the block size times with width (:code:`wide_mask_width * nfine_per_cov`).

**Sparse Map Table**

If the sparse map is a numpy record array type, it is stored as a one dimensional table array.

**Sparse Map Bit-Packed Mask**

If the sparse map is a bit-packed mask, the sparse map is stored as an array of unsigned 8-bit integers.
This will be used directly as the data buffer backing the bit-packing array structure.

.. _parquet_format:

HealSparseMap Parquet Serialization
===================================

A :code:`HealSparseMap` serialized as Parquet is sharded as a Parquet dataset for efficient access to sub-regions of very large area, high resolution maps.
The :code:`HealSparseMap` Parquet dataset consists of a directory with two metadata files; the coverage map; and a list of low resolution "i/o pixel" directories (default :code:`nside_io=4`).
In each i/o pixel directory is a Parquet file with all of the sparse map data from that i/o pixel, divided into Parquet row groups for each coverage pixel.
The :code:`HealSparseMap` Parquet format uses the default :code:`snappy` per-column compression.

Parquet File Structure
----------------------

Parquet Metadata
----------------

The :code:`metadata` is stored separately in the `_metadata` and `_common_metadata` files in the dataset directory, as per the Parquet dataset specification.
The :code:`metadata` is stored as a set of key-value pairs, each of which is a binary string.
For simplicity we describe the strings here as regular strings, but beware that behind the scenes they are stored in the python :code:`b'string'` format.

The following metadata strings are required:
* :code:`'healsparse::version'`: :code:`'1'`
* :code:`'healsparse::nside_sparse'`: :code:`str(nside_sparse)`
* :code:`'healsparse::nside_coverage'`: :code:`str(nside_coverage)`
* :code:`'healsparse::nside_io'`: :code:`str(nside_io)`
* :code:`'healsparse::filetype'`: :code:`'healsparse'`
* :code:`'healsparse::primary'`: :code:`'primary'` or :code:`''`
* :code:`'healsparse::sentinel'`: :code:`str(sentinel)` or :code:`'UNSEEN'`
* :code:`'healsparse::widemask'`: :code:`'True'` or :code:`'False'`
* :code:`'healsparse::wwidth'`: :code:`str(wide_mask_width)` or :code:`'1'`
* :code:`'healsparse::bitpacked'`: :code:`str(is_bit_mask_map)`

Note that the string :code:`'UNSEEN'` will use the special value :code:`hpgeom.UNSEEN` to fill empty/overflow pixels.

Additional metadata from the map is stored as a FITS header string (for compatibility with the FITS serialization) such that:
* :code:`'healsparse::header'`: :code:`header_string`

Parquet Coverage Map
--------------------

The coverage map is a Parquet file with the name :code:`_coverage.parquet`, stored in the dataset directory.
The coverage map has two columns:
* :code:`cov_pix`: Valid coverage pixels (:code:`nside = nside_coverage`) for the sparse map.
* :code:`row_group`: The row group index within the appropriate i/o pixel file to find the sparse data for the given coverage map.

Parquet Map Files
-----------------

Each sparse map Parquet file covers one i/o pixel.
The name of each file is :code:`iopix=###/###.parquet`, where :code:`###` is a zero-padded three digit number for the given i/o pixel.
By putting each pixel in its own directory with this naming scheme we allow pyarrow to use the hive partitioning and only touch the files as necessary.

Each file is written as a series of Parquet row groups.
Each row group contains all the data for a single coverage pixel, with :code:`nfine_per_cov` rows per row group.
The row group number within the given i/o pixel is recorded in the :code:`_coverage.parquet` coverage map file for quick access to individual and groups of coverage pixels.

The exact format of the data depends on whether the map is a simple image, a wide mask, or a record array.

**Sparse Map Image**

If the sparse map is not of a numpy record array type, it is stored in a two-column Parquet table.
The schema is given by:
* :code:`cov_pix`: :code:`int32`
* :code:`sparse`: Datatype of the sparse image data.

The :code:`cov_pix` gives the coverage pixel of the data, and is redundant with the data in :code:`_coverage.parquet`.
It compresses very efficiently, and can be used to reconstruct the :code:`_coverage.parquet` from the data files if necessary.

The :code:`sparse` column has the sparse map data (with sparse map image datatype).

Unlike the FITS serialization, the initial "overflow" coverage pixel is not serialized.
Instead, on read this is filled in with the :code:`sentinel` value from the Parquet metadata.

**Sparse Map Wide Mask**

If the sparse map is a wide mask map, the schema is the same as for a regular sparse map image.
In this case, as with the FITS serialization, the sparse map is stored as a flattened version of the in-memory :code:`wide_mask_width * npix` array.
This means that there will be :code:`wide_mask_width * nfine_per_cov` rows per row group in each wide mask Parquet file.

**Sparse Map Table**

If the sparse map is a numpy record array type, it is stored as a multi-column Parquet table with the following schema:
* :code:`cov_pix`: :code:`int32`
* :code:`column_1`: Datatype of column 1.
* :code:`column_2`: Datatype of column 2.
* Etc.

Unlike the FITS serialization, the initial "overflow" coverage pixel is not serialized.
Instead, on read this is filled in with the :code:`sentinel` value from the Parquet metadata for the :code:`primary` column.
The other columns in the overflow coverage pixel are filled with the default sentinel for that datatype (e.g., :code:`hpgeom.UNSEEN` for floating-point columns and :code:`-MAXINT` for integer columns).

**Sparse Map Bit-Packed Mask**

If the sparse map is a bit-packed mask, the schema is the same as for a regular sparse map image.
In this case, as with the FITS serialization, the sparse map is stored as an array of unsigned 8-bit integers which is the in-memory backing of the bit-packed array.
