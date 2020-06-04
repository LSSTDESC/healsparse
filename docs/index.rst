.. healsparse documentation master file, created by
   sphinx-quickstart on Fri Jun  5 07:57:24 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

`HealSparse`: A sparse implementation of HEALPix_
=================================================

`HealSparse` is a sparse implementation of HEALPix_ in Python, written for the Rubin Observatory Legacy Survey of Space and Time Dark Energy Science Collaboration (DESC_).  `HealSparse` is a pure Python library that sits on top of numpy_ and healpy_ and is designed to avoid storing full sky maps in case of partial coverage, including easy reading of sub-maps.  This reduces the overall memory footprint allowing maps to be rendered at arcsecond resolution while keeping the familiarity and power of healpy_.

`HealSparse` expands on healpy_ and straight HEALPix_ maps by allowing maps of different data types, including 32- and 64-bit floats; 8-, 16-, 32-, and 64-bit integers; "wide bit masks" of arbitrary width (allowing hundreds of bits to be efficiently and conveniently stored); and numpy_ record arrays.  Arithmetic operations between maps are supported, including sum, product, min/max, and and/or/xor bitwise operations for integer maps.  In addition, there is general support for any numpy_ universal function.

`HealSparse` also includes a simple geometric primitive library, to render circles and convex polygons.

The code is hosted in GitHub_.  Please use the `issue tracker <https://github.com/LSSTDESC/healsparse/issues>`_ to let us know about any problems or questions with the code.

.. _HEALPix: https://healpix.jpl.nasa.gov/
.. _DESC: https://lsst-desc.org/
.. _healpy: https://github.com/healpy/healpy/
.. _GitHub: https://github.com/LSSTDESC/healsparse
.. _numpy: https://github.com/numpy/numpy

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   quickstart
   basic_interface
   operations
   geometry
   concatenation
   filespec

Modules API Reference
=====================

.. toctree::
   :maxdepth: 3

   modules

Search
======

* :ref:`search`


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
