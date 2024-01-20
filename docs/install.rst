Install
=======

`HealSparse` requires `hpgeom <https://github.com/LSSTDESC/hpgeom/>`_, `numpy <https://github.com/numpy/numpy>`_, and `astropy <https://astropy.org>`_.
If you have `fitsio <https://github.com/esheldon/fitsio>`_ installed then additional features including memory-efficient concatenation of `HealSparse` maps are made available.
Installation of `healpy <https://github.com/healpy/healpy>`_ is required for reading full maps in HEALPix format.

`HealSparse` is available at `pypi <https://pypi.org/project/healsparse>`_ and `conda-forge <https://anaconda.org/conda-forge/healsparse>`_, and the most convenient way of installing the latest released version is simply:

.. code-block:: bash

  conda install -c conda-forge healsparse

or

.. code-block:: bash

  pip install healsparse

To install from source, you can run from the root directory:

.. code-block:: bash

  pip install .

