.. role:: python(code)
   :language: python

Basic HealSparse Interface
==========================

Getting Started
---------------

To conserve memory, `HealSparse` uses a dual-map approach, where a low-resolution full-sky "coverage map" is combined with a high resolution map containing the pixel data where it is available.
The resolution of the coverage map is controlled by the :code:`nside_coverage` parameter, and the resolution of the high-resolution map is controlled by the :code:`nside_sparse` parameter.
Behind the scenes, `HealSparse` uses clever indexing to allow the user to treat these as contiguous maps with minimal overhead.
**All HealSparse maps use HEALPix nest indexing behind the scenes, should be treated as nest-indexed maps.**

There are 3 basic ways to make a :code:`HealSparseMap`.
First, one can read in an existing `HEALPix` map; second, one can read in an existing :code:`HealSparseMap`; and third, one can create a new map.

.. code-block :: python

    import numpy as np
    import healsparse

    # To read a HEALPix map, the nside_coverage must be specified
    map1 = healsparse.HealSparseMap.read('healpix_map.fits', nside_coverage=32)

    # To read a healsparse map, no additional keywords are necessary
    map2 = healsparse.HealSparseMap.read('healsparse_map.hs')

    # To read part of a healsparse map, you can specify the coverage pixels to read
    map2_partial = healsparse.HealSparseMap.read('healsparse_map.hs', pixels=[100, 101])

    # To create a new map, the resolutions and datatype must be specified
    nside_coverage = 32
    nside_sparse = 4096
    map3 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_sparse, np.float64)


To set values in the map, you can use simple indexing or the explicit API:

.. code-block :: python

    map3[0: 1000] = np.arange(1000, dtype=np.float64)
    map3.update_values_pix(np.arange(1000, 2000), np.arange(1000, dtype=np.float64), nest=True)


To retrieve values from the map, you can use simple indexing or the explicit API via pixels or positions:

.. code-block :: python

    print(map3[0: 1000])
    >>> [0. ... 999.]
    print(map3.get_values_pix(np.arange(1000, 2000), nest=True))
    >>> [0. ... 999.]
    print(map3.get_values_pos(45.0, 0.1, lonlat=True))
    >>> 51.0


A :code:`HealSparseMap` has the concept of "valid pixels", the pixels over which the map is defined (as opposed to :code:`hpgeom.UNSEEN` in the case of floating point maps).
You can retrieve the array of valid pixels or the associated positions of the valid pixels easily:

.. code-block :: python

    print(map3.valid_pixels)
    >>> [   0    1    2 ... 1997 1998 1999]

    ra, dec = map3.valid_pixels_pos(lonlat=True)
    print(ra)
    >>> [45.        ... 45.3515625 ]
    print(dec)
    >>> [0.00932548 ... 0.81134431]


You can convert a :code:`HealSparseMap` to a :code:`healpy` map (:code:`numpy` array) either by using a full slice (:code:`[:]`) or with the :code:`generate_healpix_map()` method.
Do watch out, at high resolution this can blow away your memory!
In these cases, :code:`generate_healpix_map()` can degrade the map before conversion, using a reduction function (over valid pixels) of your choosing, including :code:`mean`, :code:`median`, :code:`std`, :code:`max`, :code:`min`, :code:`and`, :code:`or`, :code:`sum`, :code:`prod` (product), and :code:`wmean` (weighted mean).

.. code-block :: python

    hpmap4096 = map3[:]
    hpmap128 = map3.generate_healpix_map(nside=128, reduction='mean')


Integer Maps
------------

In addition to floating-point maps, which are natively supported by :code:`healpy`, :code:`HealSparseMap` supports integer maps.
The "sentinel" value of these maps (equivalent to :code:`hpgeom.UNSEEN`) is either :code:`-MAXINT` or :code:`0`, depending on the desired use of the map (e.g., integer values or positive bitmasks).
Note that these maps cannot be trivially converted to :code:`healpy` maps because `HEALPix` has no concept of sentinel values that are not :code:`hpgeom.UNSEEN`, which is a very large negative floating-point value.

.. code-block :: python

    import numpy as np
    import healsparse

    map_int = healsparse.HealSparseMap.make_empty(32, 4096, np.int32)
    print(map_int)
    >>> HealSparseMap: nside_coverage = 32, nside_sparse = 4096, int32

    map_int[0: 1000] = np.arange(1000, dtype=np.int32)

    print(map_int[500])
    >>> 500


Recarray Maps
-------------

:code:`HealSparseMap` also supports maps made up of :code:`numpy` record arrays.
These recarray maps will have one field that is the "primary" field which is used to test if a pixel has a valid value or not.
Therefore, these recarray maps should be used to describe associated values that share the exact same valid footprint.
Each field in the recarray can be treated as its own :code:`HealSparseMap`.
For example,

.. code-block :: python

    import numpy as np
    import healsparse

    dtype = [('a', np.float32), ('b', np.int32)]

    map_rec = healsparse.HealSparseMap.make_empty(32, 4096, dtype, primary='a')

    map_rec[0: 10000] = np.zeros(10000, dtype=dtype)
    print(map_rec.valid_pixels)
    >>> [   0    1    2 ... 9997 9998 9999]

    map_rec['a'][0: 5000] = np.arange(5000, dtype=np.float32)
    map_rec['b'][5000: 10000] = np.arange(5000, dtype=np.int32)

    print(map_rec[map_rec.valid_pixels])
    >>> [(0.,    0) (1.,    0) (2.,    0) ... (0., 4997) (0., 4998) (0., 4999)]


Note that the call :code:`map_rec['a'][0: 5000] = values` will work, but
:code:`map_rec[0: 5000]['a'] = values` will not.  Also note that using the
fields of the recarray *cannot* be used to set new pixels, this construction
can only be used to change pixel values.


Wide Masks
----------

`HealSparse` has support for "wide" bit masks with an arbitrary number of bits that are referred to by bit position rather than value.
This is useful, for example, when constructing a coadd coverage map where every pixel can uniquely identify the set of input exposures that contributed at the location of that pixel.
In the case of >64 input exposures you can no longer use a simple 64-bit integer bit mask.
Wide mask bits are always specified by giving a list of integer positions rather than values (e.g., use :code:`10` to set the 10th bit instead of :code:`1024 = 2**10`).

.. code-block :: python

    import numpy as np
    import healsparse

    map_wide = healsparse.HealSparseMap.make_empty(32, 4096, healsparse.WIDE_MASK, wide_mask_maxbits=128)

    pixels = np.arange(10000)
    map_wide.set_bits_pix(pixels, [4, 100])

    print(map_wide.check_bits_pix(pixels, [2]))
    >>> [False False False ... False False False]
    print(map_wide.check_bits_pix(pixels, [4]))
    >>> [ True  True  True ...  True  True  True]
    print(map_wide.check_bits_pix(pixels, [100]))
    >>> [ True  True  True ...  True  True  True]
    print(map_wide.check_bits_pix(pixels, [101]))
    >>> [False False False ... False False False]

    # Check if any of the bits are set
    print(map_wide.check_bits_pos([45.2], [0.2], [100, 101], lonlat=True))
    >>> [ True]


Bit-Packed Boolean Maps
-----------------------

:code:`HealSparseMap` also supports bit-packed boolean maps.
For boolean coverage masks (True/False) this fits 8 one-bit pixels per byte (rather than the default numpy boolean array which uses one byte per boolean).
In this way the full 5000 deg2 Dark Energy Survey coverage mask can be stored with nside 131072 (1.6 arcsecond resolution) in less than 4Gb of memory and very fast lookup performance.
On disk this is stored with better than 4x compression.
Note that the in-memory performance is superior to that of the Multi-Order Coverage (MOC) because of the need for MOC to store two 64-bit integers for each (hierarchical) pixel, vs. `HealSparse` using 1 bit per pixel.
Currently, the sentinel value for bit-packed boolean maps must be :code:`False`.

Note that for very large bit-packed maps the lookup performance is very good, but using :code:`valid_pixels` can be very inefficient, as you have to store all the 64-bit valid pixel indices in memory at once.
In this case, looping over coverage pixels for sub-maps (using :code:`get_covpix_maps()`) is recommended.


.. code-block :: python

    import numpy as np
    import healsparse

    map_packed = healsparse.HealSparseMap.make_empty(32, 131072, bool, bit_packed=True)
    print(map_packed)
    >>> HealSparseMap: nside_coverage = 32, nside_sparse = 131072, boolean bit-packed mask, 0 valid pixels

    map_packed[1_000_000: 2_000_000] = True
    print(map_packed.n_valid)
    >>> 1000000
    print(map_packed[1_500_000])
    >>> True


Writing Maps
------------

Writing a :code:`HealSparseMap` is easy.  To write a map in the default FITS format:

.. code-block :: python

    map3.write('output_file.hs', clobber=False)

And to write a map in the Parquet format with ``pyarrow``:

.. code-block :: python

    map3.write('output_file.hsparquet', clobber=False, format='parquet')


Metadata
--------

You can also set key/value metadata to a map that will be stored in the fits header of the file and read back in.
The keys must confirm to FITS header key standards (strings, upper case).
The metadata will be stored as a Python dictionary, and can be accessed with the :code:`metadata` property.

.. code-block :: python

    metadata = {'KEY1': 5, 'KEY2': 10.0}
    map3.metadata = metadata
    print(map3.metadata['KEY2'])
    >>> 10.0


Coverage Masks
--------------

A :code:`HealSparseMap` contains a coverage map that defines the coarse coverage over the sky.
You can retrieve a boolean array describing which pixels are covered in the map with the :code:`coverage_mask` property:

.. code-block :: python

    import hpgeom as hpg
    import matplotlib.pyplot as plt

    cov_mask = map3.coverage_mask
    cov_pixels, = np.where(cov_mask)
    ra, dec = hpg.pixel_to_angle(map3.nside_coverage, cov_pixels)
    plt.plot(ra, dec, 'r.')
    plt.show()


It is also possible to read the coverage map of a :code:`HealSparseMap` on its own:

.. code-block :: python

    cov_map = healsparse.HealSparseCoverage.read('output_file.hs')
    cov_mask = cov_map.coverage_mask


In some cases, you may be building a map and you already know the coverage when it will be finished.
In this case, it can be faster to initialize the memory at the beginning.
In this case, you can add :code:`cov_pixels` to the :code:`make_empty` call.
Be aware this may make the map larger than your actual coverage.

.. code-block :: python

    import healsparse

    nside_coverage = 32
    nside_sparse = 4096
    map4 = healsparse.HealSparseMap.make_empty(nside_coverage, nside_sparse, np.float32,
                                               cov_pixels=[5, 10, 20, 21])


Fractional Detection Maps
-------------------------

One can compute the fractional detection map of a :code:`HealSparseMap` with the :code:`fracdet_map()` method.
This method will compute the fractional area covered by the sparse map at an arbitrary resolution (not higher than the native resolution, and not lower than the coverage map :code:`nside_coverage`).
This is a count of the fraction of "valid" sub-pixels (those that are not equal to the sentinel value) in the original map.
These maps can be useful in conjunction with a degraded map to easily determine the coverage fraction of each degraded pixel.

In order to translate a :code:`fracdet_map` to lower resolution, the :code:`degrade()` method should be used with the default "mean" reduction operation.
If one tries to compute the :code:`fracdet_map` of an existing :code:`fracdet_map` then you will not get the expected output, because this is the fractional coverage of the :code:`fracdet_map` itself, not of the original sparse map.


Basic Visualization
-------------------

:code:`healsparse` does not provide any built-in visualization tools.
However, it is possible to perform quick visualizations of a :code:`HealSparseMap` using the :code:`matplotlib` package.
For example, we can take render our map as a collection of hexagonal cells using :code:`matplotlib.pyplot.hexbin`:

.. code-block :: python

    import healsparse
    import matplotlib.pyplot as plt

    nside_coverage = 32
    nside_sparse = 4096

    # Generation of the map
    hsp_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_sparse, np.float32)
    idx = np.arange(2000, 6000)
    hsp_map[idx] = np.random.uniform(size=idx.size).astype(np.float32)

    # Visualization of the map
    vpix, ra, dec = hsp_map.valid_pixels_pos(return_pixels=True)
    plt.hexbin(ra, dec, C=hsp_map[vpix])
    plt.colorbar()
    plt.show()

For more sophisticated visualizations and projections of HealSparse maps, please see `SkyProj <https://skyproj.readthedocs.io/en/latest/>`_.
