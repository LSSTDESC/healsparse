import unittest
import numpy.testing as testing
import numpy as np
import hpgeom as hpg
import tempfile
import shutil
import os
import warnings
import astropy.units as u
from astropy.coordinates import SkyCoord

import healsparse


class HealsparseMocTestCase(unittest.TestCase):
    def test_moc_writeread(self):
        """
        Test healsparse MOC write/read functionality.
        """
        nside_coverage = 32
        nside_map = 2048

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, bool)
        sparse_map[30000: 70000] = True

        fname = os.path.join(self.test_dir, 'healsparse_moc.fits')

        sparse_map.write_moc(fname)

        bool_map = healsparse.HealSparseMap.read(fname, nside_coverage=nside_coverage)

        # The MOC doesn't store the "native" resolution, so we need to upgrade.
        bool_map_ug = bool_map.upgrade(nside_map)

        self.assertEqual(len(bool_map_ug.valid_pixels), len(sparse_map.valid_pixels))
        testing.assert_array_equal(bool_map_ug.valid_pixels, sparse_map.valid_pixels)

    def test_moc_write_hsp_read_mocpy(self):
        """
        Test write healsparse MOC and read with mocpy.
        """
        try:
            import mocpy
        except ImportError:
            return

        nside_coverage = 32
        nside_map = 2048

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, bool)
        sparse_map[30000: 70000] = True
        sparse_map[80000: 90000] = True

        fname = os.path.join(self.test_dir, 'healsparse_moc.fits')

        sparse_map.write_moc(fname)

        # We use the old deprecated interface because the new one can't read
        # standard FITS files.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            moc = mocpy.MOC.from_fits(fname)

        # There is no mocpy pixel lookup, we must go through ra/dec for some reason.
        ra, dec = hpg.pixel_to_angle(nside_map, np.arange(hpg.nside_to_npixel(nside_map)))
        arr = moc.contains_lonlat(ra*u.degree, dec*u.degree)

        testing.assert_array_equal(arr.nonzero()[0], sparse_map.valid_pixels)

    def test_moc_write_mocpy_read_hsp(self):
        """
        Test write mocpy MOC and read with healsparse.
        """
        try:
            import mocpy
        except ImportError:
            return

        nside_coverage = 32
        nside_map = 2048

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, bool)
        sparse_map[30000: 70000] = True
        sparse_map[80000: 90000] = True

        fname = os.path.join(self.test_dir, 'mocpy_moc.fits')

        # There is no mocpy pixel setting, we must go through ra/dec for some reason.
        ra, dec = sparse_map.valid_pixels_pos()
        skycoords = SkyCoord(ra*u.degree, dec*u.degree)
        moc = mocpy.MOC.from_skycoords(skycoords, int(np.round(np.log2(nside_map))))

        # Use pre_v2 to force NUNIQ ordering.
        moc.save(fname, format="fits", pre_v2=True)

        bool_map = healsparse.HealSparseMap.read(fname, nside_coverage=nside_coverage)

        # The MOC doesn't store the "native" resolution, so we need to upgrade.
        bool_map_ug = bool_map.upgrade(nside_map)

        self.assertEqual(len(bool_map_ug.valid_pixels), len(sparse_map.valid_pixels))
        testing.assert_array_equal(bool_map_ug.valid_pixels, sparse_map.valid_pixels)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__ == '__main__':
    unittest.main()
