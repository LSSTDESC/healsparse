# `healsparse`
Python implementation of sparse HEALPix maps.

`HealSparse` is a sparse implementation of [HEALPix](https://healpix.jpl.nasa.gov/) in Python, written for the Rubin Observatory Legacy Survey of Space and Time Dark Energy Science Collaboration ([DESC](https://lsst-desc.org/)).
`HealSparse` is a pure python library that sits on top of [numpy](https://github.com/numpy/numpy) and [hpgeom](https://github.com/LSSTDESC/hpgeom/) and is designed to avoid storing full sky maps in case of partial coverage, including easy reading of sub-maps.
This reduces the overall memory footprint allowing maps to be rendered at arcsecond resolution while keeping the familiarity and power of [HEALPix](https://healpix.jpl.nasa.gov/).

`HealSparse` expands on functionality available in [healpy](https://github.com/healpy/healpy/) and straight [HEALPix](https://healpix.jpl.nasa.gov/) maps by allowing maps of different data types, including 32- and 64-bit floats; 8-, 16-, 32-, and 64-bit integers; "wide bit masks" of arbitrary width (allowing hundreds of bits to be efficiently and conveniently stored); and [numpy](https://github.com/numpy/numpy) record arrays.
Arithmetic operations between maps are supported, including sum, product, min/max, and and/or/xor bitwise operations for integer maps.
In addition, there is general support for any [numpy](https://github.com/numpy/numpy) universal function.

`HealSparse` also includes a simple geometric primitive library, to render circles and convex polygons.

## Requirements:

`healsparse` requires to have pre-installed the following packages:

- [numpy](https://github.com/numpy/numpy)
- [hpgeom](https://github.com/LSSTDESC/hpgeom)
- [astropy](https://astropy.org)

The following package is optional but recommended for all features including reading full maps in HEALPix format:
- [fitsio](https://github.com/esheldon/fitsio)
- [healpy](https://github.com/healpy/healpy/)

## Install:

To install the package from source go to the parent directory of the package
and do `python setup.py install [options]` or use `pip install . [options]`

## Quickstart:

A jupyter notebook is available for tutorial purposes
[here](./tutorial/quickstart.ipynb).

## Documentation:

Read the full documentation at https://healsparse.readthedocs.io/en/latest/.

## Notes:

The list of released versions of this package can be found
[here](https://github.com/LSSTDESC/healsparse/releases), with the main branch
including the most recent (non-released) development.

## Acknowledgements:

The `HealSparse` code was written by Eli Rykoff and Javier Sanchez based on an idea from Anže Slosar.

This software was developed under the Rubin Observatory Legacy Survey of Space and Time (LSST) Dark Energy Science Collaboration (DESC) using LSST DESC resources.
The DESC acknowledges ongoing support from the Institut National de Physique Nucléaire et de Physique des Particules in France; the Science & Technology Facilities Council in the United Kingdom; and the Department of Energy, the National Science Foundation, and the LSST Corporation in the United States.
DESC uses resources of the IN2P3 Computing Center (CC-IN2P3--Lyon/Villeurbanne - France) funded by the Centre National de la Recherche Scientifique; the National Energy Research Scientific Computing Center, a DOE Office of Science User Facility supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231; STFC DiRAC HPC Facilities, funded by UK BEIS National E-infrastructure capital grants; and the UK particle physics grid, supported by the GridPP Collaboration.
This work was performed in part under DOE Contract DE-AC02-76SF00515.
