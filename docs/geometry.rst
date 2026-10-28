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
In addition, there are two combination shapes, :code:`GeomUnion` and :code:`GeomIntersection`, which can be much faster when operating on (e.g.) a large set of mask shapes at once.
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


**GeomUnion**

.. code-block :: python

    # This provides the union of pixels from multiple shapes. They must
    # all have the same associated value.
    union = healsparse.GeomUnion()
    union |= healsparse.Circle(ra=200.0, dec=0.0, radius=1.0, value=True)
    union |= healsparse.Box(ra1=201.0, ra2=202.0, dec1=0.0, dec2=2.0, value=True)


**GeomIntersection**

.. code-block :: python

    # This provides the intersection of pixels from multiple shapes. They must
    # all have the same associated value.
    intersection = healsparse.GeomIntersection()
    intersection &= healsparse.Circle(ra=200.0, dec=0.0, radius=1.0, value=True)
    intersection &= healsparse.Box(ra1=200.5, ra2=202.0, dec1=0.0, dec2=2.0, value=True)


HealSparse Geometry Object Frames
---------------------------------

All of the :code:`HealSparse` shapes support different reference frames.
The default is equatorial/celestial coordinates, but can be set to Galactic or Ecliptic coordinates following :code:`hpgeom`:

The :code:`frame` options are:
* Equatorial frame (default): Specified with :code:`"equatorial"` or :code:`"icrs"` or :code:`"c"`.
* Galactic frame: Specified with :code:`"galactic"` or :code:`"g"`.
* Ecliptic frame: Specified with :code:`"ecliptic"` or :code:`"barycentricmeanecliptic"` or :code:`"e"`.

The basic coordinate system for :code:`HealSparseMap` objects are equatorial.
If, for example, a shape uses :code:`frame="galactic"` then the shape will be described in Galactic coordinates and the output pixels will be in equatorial coordinates.


Combining Geometric Objects with Maps
-------------------------------------

Given a map, it is very simple to combine geometric objects to build up complex shapes/masks/etc.
Behind the scenes, large geometric objects are rendered with :code:`hpgeom` pixel ranges which leads to greater memory efficiency.
If you are adding a large number of geometric objects at once, even greater efficiency can be achieved by using the :code:`GeomUnion` or :code:`GeomIntersection` combination classes.
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


Using Geometric Shapes To Set And Clear Pixels
----------------------------------------------

Geometric shapes can also be used to set and clear pixels.
Clearing pixels means that the pixels will set to the map's sentinel value.
This is typically :code:`UNSEEN` for floating-point maps; :code:`False` for boolean maps; and :code:`-MAXINT` or :code:`0` for integer maps.
This will remove the pixels from the map's :code:`valid_pixels`.
Clearing pixels will not expand the coverage map.

The following example renders a large circular footprint and cuts out all the low Galactic latitude pixels for a boolean mask.
Note that there are multiple equivalent ways to clear pixels.

.. code-block :: python

    import healsparse
    import numpy as np

    m = healsparse.HealSparseMap.make_empty(32, 32768, bool, bit_packed=True)

    celestial_circle = healsparse.Circle(ra=270.0, dec=-20.0, radius=40.0, value=True)

    m[celestial_circle] = True
    # Equivalently, one can do `m |= celestial_circle`

    print(m.n_valid)
    # 1507247285

    galactic_box = healsparse.Box(ra1=0.0, ra2=360.0, dec1=-20.0, dec2=20.0, value=False, frame="g")

    m.clear_pixels(galactic_box)
    # Equivalently, one can do `m &= galactic_box` (because the value is False)
    # or `m[galactic_box] = None`

    print(m.n_valid)
    # 564284571

    # Note that because this map was very large and cut down, it has extra coverage
    # and is "fragmented":
    print(m.is_fragmented)
    # True

    m.defragment()

    print(m.is_fragmented)
    # False


Using Geometric Shapes To Crop Pixels
-------------------------------------

Geometric shapes can be used to crop maps.
By default the cropped map will be "defragmented" with the coverage map shrunk to cover the new cropped region.
The cropping method can either modify a map in place (:code:`in_place=True`) or return a copy leaving the original unaltered (:code:`in_place=False`).
Note that the value of a geometric shape is not used when used in the cropping tool.

The following example demonstrates how to crop all high Galactic latitude pixels, equivalent to the sample above.
This additionally demonstrates how to create a union of multiple shapes which can be combined as one.

.. code-block :: python

    import healsparse
    import numpy as np

    m = healsparse.HealSparseMap.make_empty(32, 32768, bool, bit_packed=True)

    celestial_circle = healsparse.Circle(ra=270.0, dec=-20.0, radius=40.0, value=True)

    m[celestial_circle] = True

    print(m.n_valid)
    # 1507247285

    galactic_box1 = healsparse.Box(ra1=0.0, ra2=360.0, dec1=20.0, dec2=90.0, value=False, frame="g")
    galactic_box2 = healsparse.Box(ra1=0.0, ra2=360.0, dec1=-90.0, dec2=-20.0, value=False, frame="g")
    union = healsparse.GeomUnion(geom_list=[galactic_box1, galactic_box2])

    m.crop(union, in_place=True)

    print(m.n_valid)
    # 564284571

    # Note that by default this map will be defragmented when cropped:
    print(m.is_fragmented)
    # False


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
