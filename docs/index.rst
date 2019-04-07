HealSparse: A sparse implementation of HEALPix_
===============================================

HealSparse is a sparse implementation of HEALPix_ in Python for the LSST Dark Energy Science Collaboration (DESC_).
HealSparse is a pure Python library that uses healpy_ internally and is designed to conveniently avoid storing full
sky maps in case of partial coverage, reducing the overall memory footprint while keeping the familiarity and power
of healpy_. 

The code is hosted in GitHub_. Please use the `issue tracker <https://github.com/LSSTDESC/healsparse/issues>`_ to
let us know about any problems or questions with the code.

.. _HEALPix: https://healpix.jpl.nasa.gov/
.. _DESC: https://lsst-desc.org/
.. _healpy: https://github.com/healpy/healpy/
.. _GitHub: https://github.com/LSSTDESC/healsparse


Contents
========

.. toctree::
   :maxdepth: 2
   
   quickstart
   install
   
Modules API Reference
=====================

.. toctree::
   :maxdepth: 3

   src/healsparse

Search
======

* :ref:`search`

