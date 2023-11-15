.. role:: python(code)
   :language: python

HealSparse Map Operations
=========================

Introduction
------------

`HealSparse` has support for basic arithmetic operations between maps, including sum, product, min/max, and and/or/xor bitwise operations for integer maps.  In addition, there is general support for any :code:`numpy` universal function.  It is important to note that map operations are complicated by the fact that any given maps may not have the same pixel coverage.  Therefore, map operations can be done either with "union" (where the final map has a valid pixel list that is the union of the input maps) or "intersection" (where the final map has a valid pixel list that is the intersection of the input maps).  In the case of "union" operations, the maps that do not have coverage at a given point will be filled with an appropriate value (for example, 0 for summation and 1 for products).

Operations With a Constant
--------------------------

Arithmetic operations with a constant are very simple, and are handled with the default Python operations, supporting copy or in-place operations.

.. code-block :: python

    import numpy as np
    import healsparse

    map_float = healsparse.HealSparseMap.make_empty(32, 4096, np.float64)
    map_float[0: 10000] = np.zeros(1, dtype=np.float64)

    print(map_float[0: 10000])
    >>> [0. 0. 0. ... 0. 0. 0.]

    map_float += 10.0
    print(map_float[0: 10000])
    >>> [10. 10. 10. ... 10. 10. 10.]

    map_float /= 10.0
    print(map_float[0: 10000])
    >>> [1. 1. 1. ... 1. 1. 1.]

    map_float2 = map_float*100.0
    print(map_float[0: 10000])
    >>> [1. 1. 1. ... 1. 1. 1.]
    print(map_float2[0: 10000])
    >>> [100. 100. 100. ... 100. 100. 100.]


Operations With Multiple Maps
-----------------------------

Operations between maps can be done with either "union" or "intersection" mode.
The basic operations supported are :code:`sum`, :code:`product`, :code:`divide`, :code:`floor_divide`, :code:`min`, :code:`max`, :code:`or`, :code:`and`, :code:`xor`, and :code:`ufunc`.  Note that :code:`or`, :code:`and`:, and :code:`xor` operations are only supported for integer maps.  Note that operations between maps are only supported if they have the same :code:`nside_sparse` resolution and data type.
In all cases except division, the function name is :code:`operation_union()` or :code:`operation_intersection()`.
The division routines only support intersection operations.
For example,

.. code-block :: python

    import numpy as np
    import healsparse

    map1 = healsparse.HealSparseMap.make_empty(32, 4096, np.float64)
    map1[0: 10000] = np.ones(10000)
    map2 = healsparse.HealSparseMap.make_empty_like(map1)
    map2[5000: 15000] = np.ones(10000)*5.0

    # The union sum will have coverage that was from map1 OR map2
    sum_union = healsparse.operations.sum_union([map1, map2])
    print(sum_union[sum_union.valid_pixels].min())
    >>> 1.0
    print(sum_union[sum_union.valid_pixels].max())
    >>> 6.0
    print(sum_union.valid_pixels.min(), sum_union.valid_pixels.max())
    >>> 0 14999

    # The intersection sum will have coverage that was from map1 AND map2
    sum_intersection = healsparse.operations.sum_intersection([map1, map2])
    print(sum_intersection[sum_intersection.valid_pixels].min())
    >>> 6.0
    print(sum_intersection[sum_intersection.valid_pixels].max())
    >>> 6.0
    print(sum_intersection.valid_pixels.min(), sum_intersection.valid_pixels.max())
    >>> 5000 9999


Boolean Mask Operations
-----------------------

When using boolean masks (either regular :code:`numpy.bool_` maps or :code:`bit_packed` maps), additional bitwise operations are available.
These include ``and`` (:code:`&` and :code:`&=`); ``or`` (:code:`|` and :code:`|=`); ``xor`` (:code:`^` and :code:`^=`); and ``invert`` (:code:`~`).

When a constant value is used as the right high side (RHS) of an operation, these operations are only applied over the coverage mask of the map.
Thus, when inverting a high resolution map that only covers 3.36 deg2 (one :code:`nside=32` coverage pixel), only these pixels will be inverted and not the full 40000 deg2.

When another boolean map is used as the RHS of an operation, these operations are only applied over the coverage mask of the RHS map.
Thus, when using something like :code:`map3 = map1 | map2`, if :code:`map1` is a large map and :code:`map2` covers only a subregion, then only the subregion will be used.
Note that the coverage of the combined map will be expanded to encompass the coverage of each of the maps to be combined.
This makes it straightforward to build up a large mask out of many sub-maps.
For example,

.. code-block :: python

    import numpy as np
    import healsparse
    import hpgeom as hpg

    nside = 2**15

    # Create a "footprint" map with some pixels.
    footprint = healsparse.HealSparseMap.make_empty(32, nside, np.bool_, bit_packed=True)
    footprint[hpg.query_circle(nside, 10.0, 10.0, 1.0)] = True

    # Create a "bad region" map that is True for bad pixels.
    bad_region = healsparse.HealSparseMap.make_empty(32, nside, np.bool_, bit_packed=True)
    bad_region[hpg.query_circle(nside, 10.0, 10.5, 0.2)] = True

    # Create a "good region" map that is True over a large region
    # and false in some spots that are holes, etc.
    good_region = healsparse.HealSparseMap.make_empty(32, nside, np.bool_, bit_packed=True)
    good_region[hpg.query_box(nside, 9.0, 11.0, 9.0, 11.0)] = True
    good_region[hpg.query_circle(nside, 9.5, 9.5, 0.1)] = False

    # Now create a combined map.
    combined_map = healsparse.HealSparseMap.make_empty(32, nside, np.bool_, bit_packed=True)

    # The combined map should start with the footprint.
    combined_map |= footprint

    # The bad regions should be masked out.
    combined_map &= ~bad_region

    # Then apply the good region mask.
    combined_map &= good_region
