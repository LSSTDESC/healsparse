import numpy as np
import healpy as hp
from .healSparseMap import HealSparseMap
from .utils import eq2vec


class GeomBase(object):
    """
    base class for goemetric objects that can convert
    themselves to maps
    """

    def get_pixels(self):
        raise NotImplementedError('implment get_pixels')


class Circle(GeomBase):
    def __init__(self, *, ra, dec, radius, nside):
        """
        Parameters
        ----------
        ra: float
            ra in degrees
        dec: float
            dec in degrees
        radius: float
            radius in degrees
        """

        self.ra = ra
        self.dec = dec
        self.radius = radius
        self.radius_rad = np.deg2rad(radius)
        self.vec = eq2vec(self.ra, self.dec)

        self.nside = nside

    @property
    def sparsemap(self):
        """
        get a healsparse mask
        """

        if not hasattr(self, '_sparsemap'):
            smap = HealSparseMap.makeEmpty(
                nsideCoverage=32,
                nsideSparse=self.nside,
                dtype=np.int16,
                sentinel=0,
            )
            smap.updateValues(self.pixels, 1)
            self._smap = smap

        return self._smap

    @property
    def pixels(self):
        """
        get the pixels associated with this circle
        """
        if not hasattr(self, '_pixels'):
            self._pixels = hp.query_disc(
                self.nside,
                self.vec,
                self.radius_rad,
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
    )
    pixels = circle.pixels
    print('pixels:', pixels)
