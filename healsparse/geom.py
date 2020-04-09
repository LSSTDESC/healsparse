import numpy as np
import healpy as hp
from .healSparseMap import HealSparseMap
from .utils import is_integer_value
import numbers


def realize_geom(geom, smap, type='or'):
    """
    Realize geometry objects in a map

    Parameters
    ----------
    geom: geometric primitive or list thereof
        List of Geom objects, e.g. Circle, Polygon
    smap: HealSparseMaps
        The map in which to realize the objects
    type: string
        Way to combine the list of geometric objects.  Default
        is to "or" them
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
    base class for goemetric objects that can convert
    themselves to maps
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
        get the value to be used for all pixels in the map
        """
        return self._value

    def get_pixels(self, *, nside):
        """
        get pixels for this map

        Parameters
        ----------
        nside: int
            Nside for the pixels
        """
        raise NotImplementedError('implment get_pixels')

    def get_map(self, *, nside_coverage, nside_sparse, dtype, wide_mask_maxbits=None):
        """
        get a healsparse map corresponding to this geometric primitive

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
        HealSparseMap
        """

        x = np.zeros(1, dtype=dtype)
        if is_integer_value(x[0]):
            sentinel = 0
        else:
            sentinel = hp.UNSEEN

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
        HealSparseMap
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
    def __init__(self, *, ra, dec, radius, value):
        """
        Parameters
        ----------
        ra: float
            ra in degrees
        dec: float
            dec in degrees
        radius: float
            radius in degrees
        value: number
            Value for pixels in the map
        """

        self._ra = ra
        self._dec = dec
        self._radius = radius
        self._radius_rad = np.deg2rad(radius)
        self._vec = hp.ang2vec(ra, dec, lonlat=True)
        self._value = value

    @property
    def ra(self):
        """
        get the ra value
        """
        return self._ra

    @property
    def dec(self):
        """
        get the dec value
        """
        return self._dec

    @property
    def radius(self):
        """
        get the radius value
        """
        return self._radius

    def get_pixels(self, *, nside):
        """
        get the pixels associated with this circle

        Parameters
        ----------
        nside: int
            Nside for the pixels
        """
        return hp.query_disc(
            nside,
            self._vec,
            self._radius_rad,
            nest=True,
            inclusive=False,
        )

    def __repr__(self):
        s = 'Circle(ra=%.16g, dec=%.16g, radius=%.16g, value=%s)'
        return s % (self._ra, self._dec, self._radius, repr(self._value))


class Polygon(GeomBase):
    def __init__(self, *, ra, dec, value):
        """
        represent a polygon

        both counter clockwise and clockwise order for polygon vertices works

        Parameters
        ----------
        ra: array
            ra of vertices in degrees, size [nvert]
        dec: array
            dec of vertices in degrees, size [nvert]
        value: number
            Value for pixels in the map
        """

        ra = np.array(ra, ndmin=1)
        dec = np.array(dec, ndmin=1)

        if ra.size != dec.size:
            raise ValueError('ra/dec different sizes')
        if ra.size < 3:
            raise ValueError('a polygon must have at least 3 vertices')

        self._ra = ra
        self._dec = dec
        self._vertices = hp.ang2vec(ra, dec, lonlat=True)
        self._value = value

        self._is_integer = is_integer_value(value)

    @property
    def ra(self):
        """
        get the ra value
        """
        return self._ra

    @property
    def dec(self):
        """
        get the dec value
        """
        return self._dec

    @property
    def vertices(self):
        """
        get the dec value
        """
        return self._vertices

    def get_pixels(self, *, nside):
        """
        get the pixels associated with this polygon

        Parameters
        ----------
        nside: int
            Nside for the pixels
        """
        try:

            pixels = hp.query_polygon(
                nside,
                self._vertices,
                nest=True,
                inclusive=False,
            )

        except RuntimeError:
            # healpy raises a RuntimeError with no information attached in the
            # string, but this seems to always be a non-convex polygon
            raise ValueError('polygon is not convex: %s' % repr(self))

        return pixels

    def __repr__(self):
        ras = repr(self._ra)
        decs = repr(self._dec)

        s = 'Polygon(ra=%s, dec=%s, value=%s)'
        return s % (ras, decs, repr(self._value))
