# `healsparse`
Python implementation of sparse HEALPix maps.

`HealSparse` is a sparse implementation of
[HEALPix](https://healpix.jpl.nasa.gov/) in Python, written for the Rubin
Observatory Legacy Survey of Space and Time Dark Energy Science Collaboration
([DESC](https://lsst-desc.org/)).  `HealSparse` is a pure python library that
sits on top of [numpy](https://github.com/numpy/numpy) and
[healpy](https://github.com/healpy/healpy/) and is designed to avoid storing
full sky maps in case of partial coverage, including easy reading of sub-maps.
This reduces the overall memory footprint allowing maps to be rendered at
arcsecond resolution while keeping the familiarity and power of
[healpy](https://github.com/healpy/healpy/).

`HealSparse` expands on [healpy](https://github.com/healpy/healpy/) and
straight [HEALPix](https://healpix.jpl.nasa.gov/) maps by allowing maps of
different data types, including 32- and 64-bit floats; 8-, 16-, 32-, and 64-bit
integers; "wide bit masks" of arbitrary width (allowing hundreds of bits to be
efficiently and conveniently stored); and
[numpy](https://github.com/numpy/numpy) record arrays.  Arithmetic operations
between maps are supported, including sum, product, min/max, and and/or/xor
bitwise operations for integer maps.  In addition, there is general support for
any [numpy](https://github.com/numpy/numpy) universal function.

`HealSparse` also includes a simple geometric primitive library, to render
circles and convex polygons.

## Requirements:

`healsparse` requires to have pre-installed the following packages:

- [numpy](https://github.com/numpy/numpy)
- [healpy](https://github.com/healpy/healpy)
- [astropy](https://astropy.org)

The following package is optional but recommended for all features:
- [fitsio](https://github.com/esheldon/fitsio)

## Documentation:

Read the full documentation at https://healsparse.readthedocs.io/en/latest/.
