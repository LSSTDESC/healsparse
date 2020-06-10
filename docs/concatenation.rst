.. role:: python(code)
   :language: python

Concatenation of HealSparse Files
=================================

:code:`HealSparse` contains a routine for concatenating (combining) multiple :code:`HealSparseMap` files.  If :code:`fitsio` is available, this will be done in a memory-efficient way.  In this way, multiple non-overlapping maps can be combined.  This makes possible a simple parallelized scatter-gather approach to creating complex survey maps, where individual tiles are run independently, and then all combined at the end.

Using :code:`cat_healsparse_files()`
------------------------------------

The :code:`cat_healsparse_files()` routine takes in a list of filename, and an output filename.  The individual files must have the same :code:`nside_sparse`, but may have different :code:`nside_coverage`.  The output file will have the same :code:`nside_coverage` as the first input file unless otherwise specified.

By default, for speed, the code will not check that the input :code:`HealSparseMap` files are non-overlapping (that is, that they do not share :code:`valid_pixels`; they may share coverage in the coverage map).  This can be checked.

If :code:`fitsio` is available (recommended), the combination is not done in-memory.  This behavior can be modified by the user by setting :code:`in_memory` to :code:`True`.  However, if only :code:`astropy.io.fits` is available for FITS interfacing, the concatenation can only be done in-memory (and the :code:`in_memory` value should be overridden.

.. code-block :: python

    import healsparse

    healsparse.cat_healsparse_files(file_list, outfile, check_overlap=False, clobber=False,
                                    in_memory=False, nside_coverage_out=None)
