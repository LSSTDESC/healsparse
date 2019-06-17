from __future__ import division, absolute_import, print_function

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
from numpy import random
import os
import healsparse
import tempfile
# Set non-interactive backend for Travis
from healsparse.visu_func import *
try:
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    have_matplotlib = True
except ImportError:
    have_matplotlib = False
try:
   import cartopy
   have_cartopy = True
except ImportError:
   have_cartopy = False

class VisuFuncTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        cls.plot_file = os.path.join(cls.test_dir, 'test_view_map.png')
        cls.plot_file_2 = os.path.join(cls.test_dir, 'test_view_map2.png')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    @unittest.skipIf(not (have_matplotlib and have_cartopy),
                     'Skipping tests that require matplotlib and cartopy.')
    def test_hsp_view_map(self):
        """
        Test the generation of a healpix map from a sparse map for a single-value field
        """
        random.seed(seed=12345)

        nsideCoverage = 32
        nsideMap = 64

        nRand = 1000
        ra = np.random.random(nRand) * 360.0
        dec = np.random.random(nRand) * 180.0 - 90.0
        value = np.random.random(nRand)

        # Create a HEALPix map
        healpixMap = np.zeros(hp.nside2npix(nsideMap), dtype=np.float) + hp.UNSEEN
        idx = hp.ang2pix(nsideMap, np.pi/2-np.radians(dec), np.radians(ra), nest=True)
        healpixMap[idx]=value
        # Create a HealSparseMap
        sparseMap = healsparse.HealSparseMap(nsideCoverage=nsideCoverage, healpixMap=healpixMap)
        # View the map and the coverage mask and save it
        hsp_view_map(sparseMap, projection='PlateCarree', show_coverage=True, 
            extent=[-20,20,-40,40], nx=5, ny=5, title='test', colorlabel='test', savename = self.plot_file)
        # View the map and save it
        hsp_view_map(sparseMap, projection='Robinson', show_coverage=False, 
            title='test', colorlabel='test', savename = self.plot_file_2)


