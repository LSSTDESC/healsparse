.. role:: python(code)
   :language: python

HealSparseMap File Specification v1.1.2
=======================================

A :code:`HealSparseMap` file is a standard FITS file with two extensions.  The primary (zeroth) extension is an integer image that describes the coverage map, and the first extension is an image or binary table that describes the sparse map.  This document describes the file format specification of these two extensions in the FITS file.

.. _terminology:

Terminology
-----------

This is a list of terminology used in a :code:`HealSparseMap` file:

* :code:`nside_sparse`: The `HEALPix` nside for the fine-grained (sparse) map
* :code:`nside_coverage`: The `HEALPix` nside for the coverage map
* :code:`bit_shift`: The number of bits to shift to convert from :code:`nside_sparse` to :code:`nside_coverage` in the `HEALPix` NEST scheme.  :code:`bit_shift = 2*log_2(nside_sparse/nside_coverage)`.
* :code:`valid_pixels`: The list of pixels with defined values (:code:`> sentinel`) in the sparse map.
* :code:`sentinel`: The sentinel value that notes if a pixel is not a valid pixel.  Default is :code:`healpy.UNSEEN` for floating-point maps, :code:`-MAXINT` for integer maps, and :code:`0` for wide mask maps.
* :code:`nfine_per_cov`: The number of fine (sparse) pixels per coverage pixel.  :code:`nfine_per_cov = 2**bit_shift`.
* :code:`wide_mask_width`: The width of a wide mask, in bytes.

.. _pixel_lookups:

Pixel Lookups
-------------

The :code:`HealSparseMap` file format is derived from the method of encoding fast look-ups of arbitary pixel values using the `HEALPix` nest pixel scheme.

Given a nest-encoded sparse (high resolution) pixel value, :code:`pix_nest`, we can compute the coverage (low resolution) pixel value with a simple bit shift operation: :code:`ipnest_cov = right_shift(pix_nest, bit_shift)`, where :code:`bit_shift` is defined in :ref:`terminology`.

The sparse map itself is stored in blocks of data, each of which includes :code:`nfine_per_cov` contiguous pixels for each coverage pixel that contains valid data.  To find the location *within* a given a coverage block, we need to subtract the first fine (sparse) nest pixel value for the given coverage pixel.  Therefore, :code:`first_pixel = nfine_per_cov*ipnest_cov`.

Next, if we have a map of offsets which points to the location of the proper sparse map block, :code:`cov_map_raw_offset`, we find the look-up index is :code:`index = pix_nest - nfine_per_cov*ipnest_cov + cov_map_raw_offset[ipnest_cov]`.  In practice, we can combine the final terms here.  The look-up index is then :code:`index = pix_nest + cov_map[ipnest_cov]` where :code:`cov_map = cov_map_raw_offset[ipnest_cov] - nfine_per_cov*ipnest_cov`.

As described in :ref:`sparse_map`, the first block in the sparse map is special, and is always filled with :code:`sentinel` values.  All :code:`cov_map` indices for coverage pixels outside the coverage map point to this sentinel block.

.. _coverage_map:

Coverage Map
------------

The coverage map encodes the mapping from raw pixel number to location within the sparse map.  It is an integer (:code:`numpy.int64`) map with :code:`12*nside_coverage*nside_coverage` values, all of which are filled. The structure of the coverage map is as follows.

**Coverage Map Header**

The coverage map header must contain the following keywords:

* **EXTNAME** must be :code:`"COV"`
* **PIXTYPE** must be :code:`"HEALSPARSE"`
* **NSIDE** is equal to :code:`nside_coverage`

**Coverage Map Image**

As described in :ref:`pixel_lookups`, the coverage map image contains indices that are offset pointers to the block in the sparse map with associated values for that coverage pixel.  An empty :code:`HealSparseMap` is intialized with the following coverage pixel values, which all point to the first :code:`sentinel` block in the sparse map.

.. code-block :: python

    import numpy as np
    import healpy as hp

    cov_map[:] = -1*np.arange(hp.nside2npix(nside_coverage), dtype=np.int64)*nfine_per_cov

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

**Sparse Map Header**

The sparse map header must contain:

* **EXTNAME** must be :code:`"SPARSE"`
* **PIXTYPE** must be :code:`"HEALSPARSE"`
* **SENTINEL** is equal to :code:`sentinel`

If the sparse map is a numpy record array, it must contain:

* **PRIMARY** is equal to the name of the "primary" field which defines the valid pixels.

If the sparse map is a wide mask, it must contain:

* **WIDEMASK** must be :code:`True`
* **WWIDTH** must be the width (in bytes) of the wide mask.

**Sparse Map Image**

If the sparse map is not of a numpy record array type, it is stored as a one dimensional image array.
The first block of :code:`nfine_per_cov` values are set to :code:`sentinel`.
Each additional block of :code:`nfine_per_cov` is associated with a single element in the coverage map.
These blocks may be in any arbitrary order, allowing for easy appending of new coverage pixels.
All invalid pixels must be set to :code:`sentinel`.
If the image is an integer type with 32 bits or fewer, it may be stored with FITS tile compression, with the tile size set to the block size (:code:`nfine_per_cov`).
If the image is a floating-point image, it may be stored with FITS tile compression, with :code:`quantization_level=0` and :code:`GZIP_2` (lossless gzip compression), with the tile size set to the block size (:code:`nfine_per_cov`).

**Sparse Map Wide Mask**

If the sparse map is a wide mask map, the sparse map is stored as a flattened version of the in-memory :code:`wide_mask_width * npix` array.
This should be flattened on storage, and reshaped on read, using the default numpy memory ordering.
The sentinel value for wide masks must be :code:`0`, and all invalid pixels must be set to :code:`0`.
The wide mask image may be stored with FITS tile compression, with the tile size set to the block size times with width (:code:`wide_mask_width * nfine_per_cov`).

**Sparse Map Table**

If the sparse map is a numpy record array type, it is stored as a one dimensional table array.  The first block of :code:`nfine_per_cov` values are set such that the :code:`primary` field must be set to :code:`sentinel`.  As with the sparse map image, each additional block of :code:`nfine_per_cov` is associated with a single element in the coverage map.  These blocks may be in any arbitrary order, allowing for easy appending of new coverage pixels.  All invalid pixels must have the :code:`primary` field set to :code:`sentinel`.
