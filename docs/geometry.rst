.. role:: python(code)
   :language: python

HealSparse Geometry
===================

:code:`HealSparse` has a basic geometry library that allows you to generate maps from circles, convex polygons, ellipses, and boxes as supported by :code:`hpgeom`.
Each geometric object is associated with a single value.
On construction, geometry objects only contain information about the shape, and they are only rendered onto a `HEALPix` grid when requested.

In addition to information about the shape itself, a geometric object may optionally contain a value of :code:`nside_render`.
This indicates that the shape should always be rendered at this given resolution, no matter the resolution of the map that it is being combined with.
(Note that you can only render at a resolution that is less than or equal to the map resolution, or else a :code:`ValueError` is raised.)
This functionality may be useful if one is building a map that may be used with multiple resolutions, and one wants to ensure that a higher and lower resolution maps have exactly the same outline for these shapes.
If no :code:`nside_render` is set with the object it will always be rendered at the same resolution as the corresponding map or via the :code:`nside` parameter of :code:`get_pixels()` and :code:`get_pixel_ranges()`.

There are a few methods to realize geometry objects.
The easiest is to combine a geometric object with a :code:`HealSparseMap` map, with the ``or``, ``and``, or ``add`` operation.
One can generate a :code:`HealSparseMap` from the geometric object.
Finally, for integer-value objects one can use the :code:`realize_geom()` method to combine multiple objects by :cod:`or`-ing the integer values together.


HealSparse Geometry Shapes
--------------------------

The four shapes supported are :code:`Circle`, :code:`Ellipse`, :code:`Polygon`, and :code:`Box`.
They share a base class, and while the instantiation is different, the operations are the same.

**Circle**

.. code-block :: python

    import healsparse

    # All units are decimal degrees
    circ = healsparse.Circle(ra=200.0, dec=0.0, radius=1.0, value=1)


**Ellipse**

.. code-block :: python

    import healsparse

    # All units are decimal degrees
    # The inclination angle alpha is defined counterclockwise with respect to North.
    # See https://hpgeom.readthedocs.io/en/latest .
    ellipse = healsparse.Ellipse(ra=200.0, dec=0.0, semi_major=1.0, semi_minor=0.5, alpha=45.0, value=1)


**Convex Polygon**

.. code-block :: python

    # All units are decimal degrees
    poly = healsparse.Polygon(ra=[200.0, 200.2, 200.3, 200.2, 200.1],
                              dec=[0.0, 0.1, 0.2, 0.25, 0.13],
                              value=8)


**Box**

.. code-block :: python

    # All units are decimal degrees
    # A box differs from a polygon in that the sides will be constant ra/dec rather
    # than great circles.
    # See https://hpgeom.readthedocs.io/en/latest .
    box = healsparse.Box(ra1=20.0, ra2=30.0, dec1=10.0, dec2=5.0, value=True)


Combining Geometric Objects with Maps
-------------------------------------

Given a map, it is very simple to combine geometric objects to build up complex shapes/masks/etc.
Behind the scenes, large geometric objects are rendered with :code:`hpgeom` pixel ranges which leads to greater memory efficiency.
Note that these operations can be applied to integer or boolean maps.

.. code-block :: python

    import healsparse
    import numpy as np

    # Create an empty map.
    m = healsparse.HealSparseMap.make_empty(32, 4096, np.uint16)

    # Set a large circle to a value using the ``or`` operation
    m |= healsparse.Circle(ra=200.0, dec=20.0, radius=5.0, value=1)

    # Remove a small circle from the center using the ``and`` operation
    m &= healsparse.Circle(ra=200.0, dec=20.0, radius=1.0, value=0)

    # And add in another circle.
    m += healsparse.Circle(ra=202.0, dec=21.0, radius=0.5, value=10)


Making a Map
------------

To make a map from a geometry object, use the :code:`get_map()` method as such.  The higher resolution you choose, the better the aliasing at the edges (given that these are pixelized approximations of the true shapes).  You can also combine two maps using the general operations.  Note that if the polygon is an integer value, the default sentinel when using :code:`get_map()` is :code:`0`.

.. code-block :: python

    smap_poly = poly.get_map(nside_coverage=32, nside_sparse=32768, dtype=np.int16)
    smap_circ = circ.get_map(nside_coverage=32, nside_sparse=32768, dtype=np.int16)

    combo = healsparse.or_union([smap_poly, smap_circ])


Using :code:`realize_geom()`
----------------------------

You can only use :code:`realize_geom()` to create maps from combinations of polygons if you are using integer maps, and want to :code:`or` them together.
This method is more memory efficient than generating each individual individual map and combining them, as above.

.. code-block :: python

    realized_combo = healsparse.HealSparseMap.make_empty(32, 32768, np.int16, sentinel=0)
    healsparse.realize_geom([poly, circ], realized_combo)
