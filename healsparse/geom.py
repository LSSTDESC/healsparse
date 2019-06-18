import numpy as np
import healpy as hp
from .healSparseMap import HealSparseMap
from .utils import eq2vec


class GeomBase(object):
    """
    base class for goemetric objects that can convert
    themselves to maps
    """

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
        self._vec = eq2vec(self._ra, self._dec)
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
            smap.updateValues(self.pixels, self._value)
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


def test_circle():
    circle = Circle(
        ra=200,
        dec=0,
        radius=30.0/3600.0,
        nside=65536,
        value=2**4,
    )
    pixels = circle.pixels
    print('pixels:', pixels)
