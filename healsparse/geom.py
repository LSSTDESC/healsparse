import numpy as np
import hpgeom as hpg
from .healSparseMap import HealSparseMap
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
        Way to combine the list of geometric objects.  Default
        is to "or" them.
    """

    if type != 'or':
        raise ValueError('type of composition must be or')

    if not smap.is_integer_map:
        raise ValueError('can only or geometry objects into an integer map')

    if not isinstance(geom, (list, tuple)):
        geom = [geom]

    # split the geom objects up by value
    gdict = {}
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

        if value not in gdict:
            gdict[value] = [g]
        else:
            gdict[value].append(g)

    # deal with each value separately and add to
    # the map
    for value, glist in gdict.items():
        for i, g in enumerate(glist):
            tpixels = g.get_pixels(nside=smap.nside_sparse)
            if i == 0:
                pixels = tpixels
            else:
                oldsize = pixels.size
                newsize = oldsize + tpixels.size
                # need refcheck=False because it will fail when running
                # the python profiler; I infer that the profiler holds
                # a reference to the objects
                pixels.resize(newsize, refcheck=False)
                pixels[oldsize:] = tpixels

        pixels = np.unique(pixels)

        if smap.is_wide_mask_map:
            smap.set_bits_pix(pixels, value)
        else:
            values = smap.get_values_pix(pixels)
            values |= value
            smap.update_values_pix(pixels, values)


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
        """
        raise NotImplementedError('Implement get_pixels')

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

        x = np.zeros(1, dtype=dtype)
        if is_integer_value(x[0]):
            sentinel = 0
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
    """
    def __init__(self, *, ra, dec, radius, value):
        self._ra = ra
        self._dec = dec
        self._radius = radius
        self._value = value
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

    def get_pixels(self, *, nside):
        return hpg.query_circle(
            nside,
            self._ra,
            self._dec,
            self._radius,
            nest=True,
            inclusive=False,
        )

    def __repr__(self):
        s = 'Circle(ra=%.16g, dec=%.16g, radius=%.16g, value=%s)'
        return s % (self._ra, self._dec, self._radius, repr(self._value))


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
    def __init__(self, *, ra, dec, value):
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

    def get_pixels(self, *, nside):
        pixels = hpg.query_polygon(
            nside,
            self._ra,
            self._dec,
            nest=True,
            inclusive=False,
        )

        return pixels

    def __repr__(self):
        ras = repr(self._ra)
        decs = repr(self._dec)

        s = 'Polygon(ra=%s, dec=%s, value=%s)'
        return s % (ras, decs, repr(self._value))


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
    def __init__(self, *, ra, dec, semi_major, semi_minor, alpha, value):
        self._ra = ra
        self._dec = dec
        self._semi_major = semi_major
        self._semi_minor = semi_minor
        self._alpha = alpha
        self._value = value
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

    def get_pixels(self, *, nside):
        return hpg.query_ellipse(
            nside,
            self._ra,
            self._dec,
            self._semi_major,
            self._semi_minor,
            self._alpha,
            nest=True,
            inclusive=False
        )

    def __repr__(self):
        s = 'Ellipse(ra=%.16g, dec=%16g, semi_major=%16g, semi_minor=%16g, alpha=%16g, value=%s)'
        return s % (self._ra, self._dec, self._semi_major, self._semi_minor, self._alpha, repr(self._value))
