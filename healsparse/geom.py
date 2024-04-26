import numpy as np
import hpgeom as hpg
from .utils import is_integer_value
import numbers


def realize_geom(geom, smap, type='or'):
    """
    Realize geometry objects in a map.

    Parameters
    ----------
    geom : Geometric primitive or list thereof
        List of Geom objects, e.g. Circle, Polygon
    smap : `HealSparseMap`
        The map in which to realize the objects.
    type : `str`
        Way to combine the list of geometric objects.
        Currently only supports ``or``.
    """
    if type not in ['or']:
        raise ValueError('Type of composition must be ``or``')

    if not smap.is_integer_map:
        raise ValueError(f'Can only {type} geometry objects into an integer map')

    if not isinstance(geom, (list, tuple)):
        geom = [geom]

    # Check all the values before starting.
    for g in geom:
        value = g.value
        if isinstance(value, (tuple, list, np.ndarray)):
            value = tuple(value)
            # This is a wide mask
            if not smap.is_wide_mask_map:
                raise ValueError("Can only use wide bit geometry values in a wide mask map")
            for v in value:
                _check_int(v)
        else:
            _check_int(value)
            _check_int_size(value, smap.dtype)

    # Now generate the map.
    for g in geom:
        smap |= g


def _check_int(x):
    check = isinstance(x, numbers.Integral)
    if not check:
        raise ValueError('value must be integer type, '
                         'got %s' % x)


def _check_int_size(value, dtype):
    ii = np.iinfo(dtype)
    if value < ii.min or value > ii.max:
        raise ValueError('value %d outside range [%d, %d]' %
                         (value, ii.min, ii.max))


class GeomBase(object):
    """
    Base class for goemetric objects that can convert
    themselves to maps.
    """

    @property
    def is_integer_value(self):
        """
        Check if the value is an integer type
        """
        return is_integer_value(self._value)

    @property
    def value(self):
        """
        Get the value to be used for all pixels in the map.
        """
        return self._value

    def get_pixels(self, *, nside):
        """
        Get pixels for this geometric shape.

        Parameters
        ----------
        nside : `int`
            HEALPix nside for the pixels.

        Raises
        ------
        ValueError : If shape has nside_render set, this is raised if
            nside < nside_render.
        """
        if self._nside_render is not None:
            if nside < self._nside_render:
                raise ValueError(f"Cannot render a Circle with {self._nside_render} into nside={nside}")
            _nside = self._nside_render
        else:
            _nside = nside

        pixels = self._render(nside_render=_nside, return_pixel_ranges=False)

        if self._nside_render is not None:
            return hpg.upgrade_pixels(_nside, pixels, nside)
        else:
            return pixels

    def get_pixel_ranges(self, *, nside):
        """
        Get pixel ranges for this geometric shape.

        Parameters
        ----------
        nside : `int`
            HEALPix nside for the pixels.

        Raises
        ------
        ValueError : If shape has nside_render set, this is raised if
            nside < nside_render.
        """
        if self._nside_render is not None:
            if nside < self._nside_render:
                raise ValueError(f"Cannot render a Circle with {self._nside_render} into nside={nside}")
            _nside = self._nside_render
        else:
            _nside = nside

        pixel_ranges = self._render(nside_render=_nside, return_pixel_ranges=True)

        if self._nside_render is not None:
            return hpg.upgrade_pixel_ranges(_nside, pixel_ranges, nside)
        else:
            return pixel_ranges

    def _render(self, *, nside_render, return_pixel_ranges):
        """Internal method to render to pixels/ranges for this shape.

        Parameters
        ----------
        nside_render : `int`
            Rendering resolution.
        return_pixel_ranges : `bool`
            Return pixel ranges instead of pixels?

        Returns
        -------
        pixels or pixel_ranges : `np.ndarray`
        """
        raise NotImplementedError("The _render method must be overridden.")

    def get_map(self, *, nside_coverage, nside_sparse, dtype, wide_mask_maxbits=None):
        """
        Get a healsparse map corresponding to this geometric primitive.

        Parameters
        ----------
        nside_coverage : `int`
            nside of coverage map
        nside_sparse : `int`
            nside of sparse map
        dtype : `np.dtype`
            dtype of the output array
        wide_mask_maxbits : `int`, optional
            Create a "wide bit mask" map, with this many bits.

        Returns
        -------
        hsmap : `healsparse.HealSparseMap`
        """
        from .healSparseMap import HealSparseMap

        x = np.zeros(1, dtype=dtype)
        if is_integer_value(x[0]):
            sentinel = 0
        elif dtype == np.bool_ or dtype == bool:
            sentinel = False
        else:
            sentinel = hpg.UNSEEN

        if isinstance(self._value, (tuple, list, np.ndarray)):
            # This is a wide mask
            if wide_mask_maxbits is None:
                wide_mask_maxbits = np.max(self._value)
            else:
                if wide_mask_maxbits < np.max(self._value):
                    raise ValueError("wide_mask_maxbits (%d) is less than maximum bit value (%d)" %
                                     (wide_mask_maxbits, np.max(self._value)))

        smap = HealSparseMap.make_empty(
            nside_coverage=nside_coverage,
            nside_sparse=nside_sparse,
            dtype=dtype,
            sentinel=sentinel,
            wide_mask_maxbits=wide_mask_maxbits
        )
        pixels = self.get_pixels(nside=nside_sparse)

        if wide_mask_maxbits is None:
            # This is a regular set
            smap.update_values_pix(pixels, np.array([self._value], dtype=dtype))
        else:
            # This is a wide mask
            smap.set_bits_pix(pixels, self._value)

        return smap

    def get_map_like(self, sparseMap):
        """
        Get a healsparse map corresponding to this geometric primitive,
        with the same parameters as an input sparseMap.

        Parameters
        ----------
        sparseMap : `healsparse.HealSparseMap`
            Input map to match parameters

        Returns
        -------
        hsmap : `healsparse.HealSparseMap`
        """
        from .healSparseMap import HealSparseMap

        if not isinstance(sparseMap, HealSparseMap):
            raise RuntimeError("Input sparseMap must be a HealSparseMap")
        if sparseMap.is_rec_array:
            raise RuntimeError("Input SparseMap cannot be a rec array")

        if sparseMap.is_wide_mask_map:
            wide_mask_maxbits = sparseMap.wide_mask_maxbits
        else:
            wide_mask_maxbits = None

        return self.get_map(nside_coverage=sparseMap.nside_coverage,
                            nside_sparse=sparseMap.nside_sparse,
                            dtype=sparseMap.dtype, wide_mask_maxbits=wide_mask_maxbits)


class Circle(GeomBase):
    """
    Parameters
    ----------
    ra : `float`
        RA in degrees (scalar-only).
    dec : `float`
        Declination in degrees (scalar-only).
    radius : `float`
        Radius in degrees (scalar-only).
    value : number
        Value for pixels in the map (scalar or list of bits for `wide_mask`)
    nside_render : `int`, optional
        If this is set, the shape will always be rendered at this
        nside and then these pixels will be 'upgraded' to the resolution
        of the map.
    """
    def __init__(self, *, ra, dec, radius, value, nside_render=None):
        self._ra = ra
        self._dec = dec
        self._radius = radius
        self._value = value
        self._nside_render = nside_render
        sc_ra = np.isscalar(self._ra)
        sc_dec = np.isscalar(self._dec)
        sc_radius = np.isscalar(self._radius)
        if (not sc_ra) or (not sc_dec) or (not sc_radius):
            raise ValueError('Circle only accepts scalar inputs for ra, dec, and radius')

    @property
    def ra(self):
        """
        Get the RA value.
        """
        return self._ra

    @property
    def dec(self):
        """
        Get the dec value.
        """
        return self._dec

    @property
    def radius(self):
        """
        Get the radius value.
        """
        return self._radius

    def _render(self, *, nside_render, return_pixel_ranges):
        return hpg.query_circle(
            nside_render,
            self._ra,
            self._dec,
            self._radius,
            nest=True,
            inclusive=False,
            return_pixel_ranges=return_pixel_ranges,
        )

    def __repr__(self):
        s = 'Circle(ra=%.16g, dec=%.16g, radius=%.16g, value=%s, nside_render=%s)'
        return s % (self._ra, self._dec, self._radius, repr(self._value), repr(self._nside_render))


class Polygon(GeomBase):
    """
    Represent a polygon.

    Both counter clockwise and clockwise order for polygon vertices works

    Parameters
    ----------
    ra : `np.ndarray` (nvert,)
        RA of vertices in degrees.
    dec : `np.ndarray` (nvert,)
        Declination of vertices in degrees.
    value : number
        Value for pixels in the map
    """
    def __init__(self, *, ra, dec, value, nside_render=None):
        ra = np.array(ra, ndmin=1)
        dec = np.array(dec, ndmin=1)

        if ra.size != dec.size:
            raise ValueError('ra/dec are different sizes')
        if ra.size < 3:
            raise ValueError('A polygon must have at least 3 vertices')
        self._ra = ra
        self._dec = dec
        self._vertices = hpg.angle_to_vector(ra, dec, lonlat=True)
        self._value = value
        self._nside_render = nside_render

        self._is_integer = is_integer_value(value)

    @property
    def ra(self):
        """
        Get the RA values of the vertices.
        """
        return self._ra

    @property
    def dec(self):
        """
        Get the dec values of the vertices.
        """
        return self._dec

    @property
    def vertices(self):
        """
        Get the vertices in unit vector form.
        """
        return self._vertices

    def _render(self, *, nside_render, return_pixel_ranges):
        return hpg.query_polygon(
            nside_render,
            self._ra,
            self._dec,
            nest=True,
            inclusive=False,
            return_pixel_ranges=return_pixel_ranges,
        )

    def __repr__(self):
        ras = repr(self._ra)
        decs = repr(self._dec)

        s = 'Polygon(ra=%s, dec=%s, value=%s, nside_render=%s)'
        return s % (ras, decs, repr(self._value), repr(self._nside_render))


class Ellipse(GeomBase):
    """
    Create an ellipse.

    Parameters
    ----------
    ra : `float`
        ra in degrees (scalar only)
    dec : `float`
        dec in degrees (scalar only)
    semi_major : `float`
        The semi-major axis of the ellipse in degrees.
    semi_minor : `float`
        The semi-minor axis of the ellipse in degrees.
    alpha : `float`
        Inclination angle, counterclockwise with respect to North (degrees).
    value : number
        Value for pixels in the map (scalar or list of bits for `wide_mask`).
    """
    def __init__(self, *, ra, dec, semi_major, semi_minor, alpha, value, nside_render=None):
        self._ra = ra
        self._dec = dec
        self._semi_major = semi_major
        self._semi_minor = semi_minor
        self._alpha = alpha
        self._value = value
        self._nside_render = nside_render
        sc_ra = np.isscalar(self._ra)
        sc_dec = np.isscalar(self._dec)
        sc_semi_major = np.isscalar(self._semi_major)
        sc_semi_minor = np.isscalar(self._semi_minor)
        sc_alpha = np.isscalar(self._alpha)
        if not sc_ra or not sc_dec or not sc_semi_major or not sc_semi_minor or not sc_alpha:
            raise ValueError(
                'Ellipse only accepts scalar inputs for ra, dec, semi_major, semi_minor, and alpha.'
            )

    @property
    def ra(self):
        """
        Get the RA value.
        """
        return self._ra

    @property
    def dec(self):
        """
        Get the dec value.
        """
        return self._dec

    @property
    def semi_major(self):
        """
        Get the semi_major value.
        """
        return self._semi_major

    @property
    def semi_minor(self):
        """
        Get the semi_minor value.
        """
        return self._semi_minor

    @property
    def alpha(self):
        """
        Get the alpha value.
        """
        return self._alpha

    def _render(self, *, nside_render, return_pixel_ranges):
        return hpg.query_ellipse(
            nside_render,
            self._ra,
            self._dec,
            self._semi_major,
            self._semi_minor,
            self._alpha,
            nest=True,
            inclusive=False,
            return_pixel_ranges=return_pixel_ranges,
        )

    def __repr__(self):
        s = ("Ellipse(ra=%.16g, dec=%16g, semi_major=%16g, semi_minor=%16g, alpha=%16g, value=%s, "
             "nside_render=%s)")
        return s % (self._ra, self._dec, self._semi_major, self._semi_minor, self._alpha, repr(self._value),
                    repr(self._nside_render))


class Box(GeomBase):
    """
    A geometric shape that has sides of constant lon/lat.

    This shape is in contrast with a Polygon which will have great
    circle boundaries. See hpgeom.query_box() for details.

    Parameters
    ----------
    ra1, ra2 : `float`
        RA in degrees. All points within [ra1, ra2] will be selected.
        If ra1 > ra2 then the box will wrap around 360 degrees. If
        ra1 == 0.0 and ra2 == 360.0 then the box will contain points at
        all right ascensions.
    dec1, dec2 : `float`
        Declination in degrees. All points within [dec1, dec2] will be
        selected.  If dec1 or dec2 is 90.0 or -90.0 then the box will
        be an arc of a circle with the center at the north/south pole.
    value : number
        Value for pixels in the map (scalar or list of bits for `wide_mask`)
    nside_render : `int`, optional
        If this is set, the shape will always be rendered at this
        nside and then these pixels will be 'upgraded' to the resolution
        of the map.
    """
    def __init__(self, *, ra1, ra2, dec1, dec2, value, nside_render=None):
        self._ra1 = ra1
        self._ra2 = ra2
        self._dec1 = dec1
        self._dec2 = dec2
        self._value = value
        self._nside_render = nside_render

        sc_ra1 = np.isscalar(self._ra1)
        sc_ra2 = np.isscalar(self._ra2)
        sc_dec1 = np.isscalar(self._dec1)
        sc_dec2 = np.isscalar(self._dec2)

        if not sc_ra1 or not sc_ra2 or not sc_dec1 or not sc_dec2:
            raise ValueError("Box only accepts scalar inputs for ra1, ra2, dec1, and dec2")

    @property
    def ra1(self):
        return self._ra1

    @property
    def ra2(self):
        return self._ra2

    @property
    def dec1(self):
        return self._dec1

    @property
    def dec2(self):
        return self._dec2

    def _render(self, *, nside_render, return_pixel_ranges):
        return hpg.query_box(
            nside_render,
            self._ra1,
            self._ra2,
            self._dec1,
            self._dec2,
            nest=True,
            inclusive=False,
            return_pixel_ranges=return_pixel_ranges,
        )

    def __repr__(self):
        s = "Box(ra1=%.16g, ra2=%.16g, dec1=%.16g, dec2=%.16g, value=%s, nside_render=%s"
        return s % (self._ra1, self._ra2, self._dec1, self._dec2, repr(self._value), repr(self._nside_render))
