import numpy as np
import healpy as hp
from .healSparseMap import HealSparseMap


def make_circles(*, ra, dec, radius, value):
    """
    make multiple Circles

    Parameters
    ----------
    ra: array
        RA in degrees
    dec: array
        DEC in degrees
    radius: array
        Radius in degrees
    value: scalar or array
        A scalar of array of values

    Returns
    -------
    List of Circle objects
    """
    ra = np.array(ra, ndmin=1)
    dec = np.array(dec, ndmin=1)
    radius = np.array(radius, ndmin=1)
    value = np.array(value, ndmin=1)

    if ra.size != dec.size:
        raise ValueError('ra/dec different sizes')
    if radius.size != dec.size:
        raise ValueError('ra/radius different sizes')
    if value.size != dec.size and value.size != 1:
        raise ValueError('value should be scalar or '
                         'same size as ra')

    circles = []
    for i in range(ra.size):
        if value.size == 1:
            v = value[0]
        else:
            v = value[i]

        circle = Circle(ra=ra[i], dec=dec[i], radius=radius[i], value=v)
        circles.append(circle)

    return circles


def or_geom(geom, smap):
    """
    Realize geometry objects in a map

    Parameters
    ----------
    geom: geometric primitive or list thereof
        List of Geom objects, e.g. Circle, Polygon
    smap: HealSparseMaps
        The map in which to realize the objects
    """

    if not smap.isIntegerMap:
        raise ValueError('can only or geometry objects into an integer map')

    if not isinstance(geom, (list, tuple)):
        geom = [geom]

    # split the geom objects up by value
    gdict = {}
    for g in geom:
        value = g.value
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
            tpixels = g.get_pixels(nside=smap.nsideSparse)
            if i == 0:
                pixels = tpixels
            else:
                oldsize = pixels.size
                newsize = oldsize + tpixels.size
                pixels.resize(newsize)
                pixels[oldsize:] = tpixels

        pixels = np.unique(pixels)

        values = smap.getValuePixel(pixels)
        values |= value
        print(pixels, values)
        smap.updateValues(pixels, values)


def _check_int(x):
    check = (
        issubclass(x.__class__, np.integer) or
        issubclass(x.__class__, int)
    )
    if not check:
        raise ValueError('value must be integer type, '
                         'got %s' % x)


def _check_int_size(value, dtype):
    ii = np.iinfo(dtype)
    if value < ii.min or value > ii.max:
        raise ValueError('value %d outside range [%d, %d]' %
                         value, ii.min, ii.max)


class GeomBase(object):
    """
    base class for goemetric objects that can convert
    themselves to maps
    """

    @property
    def value(self):
        """
        get the value to be used for all pixels in the map
        """
        return self._value

    def get_pixels(self, *args, **kw):
        raise NotImplementedError('implment get_pixels')

    def get_map(self, *, nside, dtype):
        """
        get a healsparse map corresponding to this Circle
        """

        smap = HealSparseMap.makeEmpty(
            nsideCoverage=32,
            nsideSparse=nside,
            dtype=dtype,
            sentinel=0,
        )
        pixels = self.get_pixels(nside=nside)
        values = self.get_values(size=pixels.size, dtype=dtype)
        smap.updateValues(pixels, values)

        return smap

    def get_values(self, *, size, dtype):
        """
        get the values associated with this circle
        """
        values = np.zeros(size, dtype=dtype)
        values[:] = self._value

        return values


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
        """
        return hp.query_disc(
            nside,
            self._vec,
            self._radius_rad,
            nest=True,
            inclusive=False,
        )


class Polygon(GeomBase):
    def __init__(self, *, ra, dec, value):
        """
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
        get the pixels associated with this circle
        """
        return hp.query_polygon(
            nside,
            self._vertices,
            nest=True,
            inclusive=False,
        )


def test_circle(show=False):
    ra, dec = 200.0, 0.0
    radius = 30.0/3600.0
    nside = 2**17
    circle = Circle(
        ra=ra,
        dec=dec,
        radius=radius,
        value=2**4,
    )
    pixels = circle.get_pixels(nside=nside)
    print('pixels:', pixels)

    smap = circle.get_map(nside=nside, dtype=np.int16)

    if show:
        import biggles
        pra, pdec = hp.pix2ang(nside, pixels, nest=True, lonlat=True)
        plt = biggles.plot(
            pra,
            pdec,
            type='filled circle',
            xlabel='RA',
            ylabel='DEC',
            aspect_ratio=1,
            visible=False,
        )
        plt.add(
            biggles.Circle(ra, dec, radius, color='red'),
        )
        plt.show()

        """
        from .visu_func import hsp_view_map

        extent = [
            ra-radius*1.1,
            ra+radius*1.1,
            dec-radius*1.1,
            dec+radius*1.1,
        ]
        hsp_view_map(
            smap,
            savename='/tmp/test.png',
            show_coverage=False,
            extent=extent,
        )
        """
        return plt
    else:
        return None


def test_circles(show=False):
    nside = 2**17
    dtype = np.int16

    rng = np.random.RandomState(31415)
    ncircle = 5
    ra_range = 199.0, 200.1
    dec_range = -0.1, 0.1
    ra = rng.uniform(low=ra_range[0], high=ra_range[1], size=ncircle)
    dec = rng.uniform(low=dec_range[0], high=dec_range[1], size=ncircle)

    radius = rng.uniform(low=30.0/3600.0, high=120.0/3600.0, size=ncircle)

    values = np.array([2, 4, 8, 16, 32], dtype=dtype)

    circles = make_circles(
        ra=ra,
        dec=dec,
        radius=radius,
        value=values,
    )

    smap = HealSparseMap.makeEmpty(
        nsideCoverage=32,
        nsideSparse=nside,
        dtype=dtype,
        sentinel=0,
    )
    or_geom(circles, smap)

    test, = np.where(smap._sparseMap > 0)
    print(test.size)

    w, = np.where(smap._sparseMap != smap._sentinel)
    print(w.size)

    data = smap.getValuePixel(smap.validPixels)
    print('_sparseMap:', smap._sparseMap)
    print('data:', data)
    w, = np.where(data != smap._sentinel)
    print(w.size)

    if show:
        from .visu_func import hsp_view_map

        extent = [
            ra_range[0]*0.9,
            ra_range[1]*1.1,
            dec_range[0]*1.1,
            dec_range[1]*1.1,
        ]
        hsp_view_map(
            smap,
            savename='/tmp/test.png',
            show_coverage=False,
            extent=extent,
        )


def test_box(show=False):
    ra = [200.0, 200.2, 200.2, 200.0]
    dec = [0.0, 0.0, 0.1, 0.1]
    nside = 2**15
    poly = Polygon(
        ra=ra,
        dec=dec,
        value=2**4,
    )

    smap = poly.get_map(nside=nside, dtype=np.int16)

    if show:
        import biggles
        pixels = poly.get_pixels(nside=nside)
        pra, pdec = hp.pix2ang(nside, pixels, nest=True, lonlat=True)
        plt = biggles.plot(
            pra,
            pdec,
            type='filled circle',
            xlabel='RA',
            ylabel='DEC',
            aspect_ratio=0.5,
            visible=False,
        )
        plt.add(
            biggles.Box((ra[0], dec[0]), (ra[2], dec[2]), color='red'),
        )
        plt.show()


def test_polygon(show=False):
    # counter clockwise
    ra  = [200.0, 200.2, 200.3, 200.2, 200.1]
    dec = [0.0,     0.1,   0.2,   0.25, 0.13]
    nside = 2**15
    poly = Polygon(
        ra=ra,
        dec=dec,
        value=2**4,
    )

    smap = poly.get_map(nside=nside, dtype=np.int16)

    if show:
        import biggles
        pixels = poly.get_pixels(nside=nside)
        pra, pdec = hp.pix2ang(nside, pixels, nest=True, lonlat=True)
        plt = biggles.plot(
            pra,
            pdec,
            type='filled circle',
            xlabel='RA',
            ylabel='DEC',
            aspect_ratio=0.5,
            visible=False,
        )

        rac = np.array(list(ra) + [ra[0]])
        decc = np.array(list(dec) + [dec[0]])
        plt.add(
            biggles.Curve(rac, decc, color='red'),
        )
        plt.show()

        return plt

    else:
        return None
