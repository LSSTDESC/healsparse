from __future__ import division, absolute_import, print_function

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
from numpy import random
import tempfile
import shutil
import os
import fitsio

import healsparse

class RecArrayTestCase(unittest.TestCase):
    def test_readRecarray(self):
        """
        Test recarray reading.  (Because writing hasn't been written yet.)
        """

        random.seed(seed=12345)

        nsideCoverage = 32
        nsideMap = 64

        nRand = 1000
        ra = np.random.random(nRand) * 360.0
        dec = np.random.random(nRand) * 180.0 - 90.0

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        # Generate a random map...

        column1 = np.zeros(hp.nside2npix(nsideMap)) + hp.UNSEEN
        column1[0: 20000] = np.random.random(size=20000)

        column2 = np.zeros(hp.nside2npix(nsideMap)) + hp.UNSEEN
        column2[0: 20000] = np.random.random(size=20000)

        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        ipnest = hp.ang2pix(nsideMap, theta, phi, nest=True)

        testValues1 = column1[ipnest]
        testValues2 = column2[ipnest]

        # create a map by hand
        covIndexMap, sparseMap1 = healsparse.HealSparseMap.convertHealpixMap(column1, nsideCoverage)
        _, sparseMap2 = healsparse.HealSparseMap.convertHealpixMap(column2, nsideCoverage)

        cHdr = fitsio.FITSHDR()
        cHdr['PIXTYPE'] = 'HEALSPARSE'
        cHdr['NSIDE'] = nsideCoverage
        fitsio.write(os.path.join(self.test_dir, 'healsparse_map_recarray.fits'), covIndexMap, header=cHdr, extname='COV', clobber=True)

        sHdr = fitsio.FITSHDR()
        sHdr['PIXTYPE'] = 'HEALSPARSE'
        sHdr['NSIDE'] = nsideMap
        sHdr['PRIMARY'] = 'column1'

        recArray = np.zeros(sparseMap1.size, dtype=[('column1', 'f8'),
                                                    ('column2', 'f8')])
        recArray['column1'][:] = sparseMap1
        recArray['column2'][:] = sparseMap2
        fitsio.write(os.path.join(self.test_dir, 'healsparse_map_recarray.fits'), recArray, header=sHdr, extname='SPARSE')

        sparseMap = healsparse.HealSparseMap.read(os.path.join(self.test_dir, 'healsparse_map_recarray.fits'))

        testing.assert_almost_equal(sparseMap.getValuePixel(ipnest)['column1'], testValues1)
        testing.assert_almost_equal(sparseMap.getValuePixel(ipnest)['column2'], testValues2)



    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)

if __name__=='__main__':
    unittest.main()

