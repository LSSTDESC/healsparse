import numpy as np
import healpy as hp
from .healSparseMap import HealSparseMap
from . import utils


class GeomBase(object):
    """
    base class for goemetric objects that can convert
    themselves to maps
    """

    @property
    def nside(self):
        return self._nside

    @property
    def pixels(self):
        raise NotImplementedError('implment pixels property')

    @property
    def sparsemap(self):
        raise NotImplementedError('implment sparsemap property')


class Circle(GeomBase):
    def __init__(self, *, ra, dec, radius, nside, value, dtype=np.int16):
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

        self._nside = nside

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

    @property
    def sparsemap(self):
        """
        get a healsparse mask
        """

        if not hasattr(self, '_sparsemap'):
            smap = HealSparseMap.makeEmpty(
                nsideCoverage=32,
                nsideSparse=self._nside,
                dtype=self._dtype,
                sentinel=0,
            )
            smap.updateValues(self.pixels, self.values)
            self._smap = smap

        return self._smap

    @property
    def pixels(self):
        """
        get the pixels associated with this circle
        """
        if not hasattr(self, '_pixels'):
            self._pixels = hp.query_disc(
                self._nside,
                self._vec,
                self._radius_rad,
                nest=True,
                inclusive=True,
            )

        return self._pixels

    @property
    def values(self):
        """
        get the values associated with this circle
        """
        if not hasattr(self, '_values'):
            self._values = np.zeros(self.pixels.size, dtype=self._dtype)
            self._values[:] = self._value

        return self._values



def test_circle(show=False):
    ra, dec = 200.0, 0.0
    radius = 30.0/3600.0
    nside = 2**17
    circle = Circle(
        ra=ra,
        dec=dec,
        radius=radius,
        nside=nside,
        value=2**4,
    )
    pixels = circle.pixels
    print('pixels:', pixels)

    smap = circle.sparsemap
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
            ra-radius*2,
            ra+radius*2,
            dec-radius*2,
            dec+radius*2,
        ]
        hsp_view_map(smap, savename='test.png', show_coverage=False,
                     extent=extent)
        """
