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

