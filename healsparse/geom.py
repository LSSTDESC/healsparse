import numpy as np
import healpy as hp
from .healSparseMap import HealSparseMap
from . import utils

def make_circles(ra, dec, radius):
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

    Returns
    -------
    List of Circle objects
    """
    ra = np.array(ra, ndmin=1)
    dec = np.array(dec, ndmin=1)
    radius = np.array(radius, ndmin=1)

    if ra.size != dec.size:
        raise ValueError()

def realize_geom(geom, smap):
    """
    Realize geometry objects in a map

    Parameters
    ----------
    geom: geommetric primitive or list thereof
        List of Geom objects, e.g. Circle, Polygon
    smap: HealSparseMaps
        Map in which to realize the objects
    """

    if not isinstance(geom, (list, tuple)):
        geom = [geom]

    # split the geom objects up by value
    gdict = {}
    for g in geom:
        value = geom.value
        if value not in gdict:
            gdict[value] = [g]
        else:
            gdict[value].append(g)

    # deal with each value separately and add to
    # the map
    for value, glist in gdict.items():
        for i, g in enumerate(glist):
            tpixels = g.get_pixels(nside=smap.nsideSparse)
            if i==0:
                pixels = tpixels
            else:
                oldsize = pixels.size
                newsize = oldsize + tpixels.size
                pixels.resize(newsize)
                pixels[oldsize:] = tpixels

        pixels = np.unique(pixels)

        values = sparseMap.getValuePixel(pixels)
        values |= value
        smap.updateValues(pixels, values)


class GeomBase(object):
    """
    base class for goemetric objects that can convert
    themselves to maps
    """

    @property
    def value(self):
        return self._value

    def get_pixels(self, *args, **kw):
        raise NotImplementedError('implment get_pixels')

    def get_values(self, *args, **kw):
        raise NotImplementedError('implment get_values')

    def get_map(self, *args, **kw):
        raise NotImplementedError('implment get_map')


class Circle(GeomBase):
    def __init__(self, *, ra, dec, radius, value, dtype=np.int16):
        """
        Parameters
        ----------
        ra: float
            ra in degrees
        dec: float
            dec in degrees
        radius: float
            radius in degrees
        value: numpy dtype
            Default int16
        """

        self._ra = ra
        self._dec = dec
        self._radius = radius
        self._radius_rad = np.deg2rad(radius)
        self._vec = hp.ang2vec(ra, dec, lonlat=True)
        self._value = value
        self._dtype = dtype

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

    def get_map(self, *, nside):
        """
        get a healsparse map corresponding to this Circle
        """

        smap = HealSparseMap.makeEmpty(
            nsideCoverage=32,
            nsideSparse=nside,
            dtype=self._dtype,
            sentinel=0,
        )
        pixels = self.get_pixels(nside=nside)
        values = self.get_values(size=pixels.size)
        smap.updateValues(pixels, values)

        return smap

    def get_pixels(self, *, nside):
        """
        get the pixels associated with this circle
        """
        pixels = hp.query_disc(
            nside,
            self._vec,
            self._radius_rad,
            nest=True,
            inclusive=False,
        )

        return pixels

    def get_values(self, *, size):
        """
        get the values associated with this circle
        """
        values = np.zeros(size, dtype=self._dtype)
        values[:] = self._value

        return values


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
    values = circle.get_values(size=pixels.size)
    print('pixels:', pixels)

    smap = circle.get_map(nside=nside)
    if show:
        import biggles
        pra, pdec = hp.pix2ang(nside, pixels, nest=True, lonlat=True)
        plt=biggles.plot(
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
        hsp_view_map(smap, savename='test.png', show_coverage=False,
                     extent=extent)
        """
