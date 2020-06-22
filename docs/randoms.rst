.. role:: python(code)
   :language: python

HealSparse Randoms
==================

:code:`HealSparse` can generate uniform randoms based on the valid pixels in a :code:`HealSparseMap` map.  It is up to the user of these random points to weight them appropriately.

There are two methods to generate randoms.  The "fast" method requires more memory, and produces randoms that are quantized (at a very high level).  The regular method is not quantized and does not require any extra memory, but is significantly slower.


Fast Random Generation
----------------------

The fast random generation is run with :code:`healsparse.make_uniform_randoms_fast()`.  This code path requires more memory and may not be suitable for very large, high resolution masks.  The output randoms are quantized by the :code:`nside_randoms` quantity.  The default is :code:`2**23`, which is approximately 7e-6 arcsecond quantization, which should be adequate for most purposes.

.. code-block :: python

    import healsparse

    circ = healsparse.Circle(ra=200.0, dec=0.0, radius=1.0, value=1)
    smap = circ.get_map(nside_coverage=32, nside_sparse=32768, dtype=np.int16)

    # Make 100000 randoms
    ra_rand, dec_rand = healsparse.make_uniform_randoms_fast(smap, 100000)


Regular Random Generation
-------------------------

The regular random generation is run with :code:`healsparse.make_uniform_randoms()`.  This code requirest less memory than the fast generation, and there is no quanitization in the random points.  However, it is slower than the fast generation.  The API is very similar to :code:`healsparse.make_uniform_randoms_fast()`.

.. code-block :: python

    import healsparse

    circ = healsparse.Circle(ra=200.0, dec=0.0, radius=1.0, value=1)
    smap = circ.get_map(nside_coverage=32, nside_sparse=32768, dtype=np.int16)

    # Make 100000 randoms
    ra_rand, dec_rand = healsparse.make_uniform_randoms(smap, 100000)
