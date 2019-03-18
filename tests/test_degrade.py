from __future__ import division, absolute_import, print_function

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
from numpy import random
import healsparse

class DegradeMapTestCase(unittest.TestCase):
    def test_degradeMapFloat(self):
        """
        Test HealSparse.degrade functionality with float quantities
        """
        random.seed(12345)
        nsideCoverage = 32
        nsideMap = 1024
        nsideNew = 256
        fullMap = random.random(hp.nside2npix(nsideMap))

        # Generate sparse map

        sparseMap = healsparse.HealSparseMap(healpixMap=fullMap, nsideCoverage=nsideCoverage,
                        nsideSparse=nsideMap)
         
        # Degrade original HEALPix map

        deg_map = hp.ud_grade(fullMap, nside_out=nsideNew, order_in='NESTED', order_out='NESTED')

        # Degrade sparse map and compare to original

        newMap = sparseMap.degrade(nside_out=nsideNew)

        # Test the coverage map generation and lookup

        testing.assert_equal(deg_map, newMap._sparseMap[2**newMap._bitShift:])

    def test_degradeMapRecArray(self):
        """
        Test HealSparse.degrade functionality with recarray quantities
        """
        
        random.seed(seed=12345)

        nsideCoverage = 32
        nsideMap = 1024
        nsideNew = 256

        nRand = 1000
        ra = np.random.random(nRand) * 360.0
        dec = np.random.random(nRand) * 180.0 - 90.0

        dtype = [('col1', 'f8'), ('col2', 'f8')]
        sparseMap = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideMap, dtype, primary='col1')
        pixel = np.arange(20000)
        values = np.zeros_like(pixel, dtype=dtype)
        values['col1'] = np.random.random(size=pixel.size)
        values['col2'] = np.random.random(size=pixel.size)
        sparseMap.updateValues(pixel, values)
   
        # Make the test values
        hpmapCol1 = np.zeros(hp.nside2npix(nsideMap)) + hp.UNSEEN
        hpmapCol2 = np.zeros(hp.nside2npix(nsideMap)) + hp.UNSEEN
        hpmapCol1[pixel] = values['col1']
        hpmapCol2[pixel] = values['col2']
        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        # Degrade healpix maps
        hpmapCol1 = hp.ud_grade(hpmapCol1, nside_out=nsideNew, order_in='NESTED', order_out='NESTED')
        hpmapCol2 = hp.ud_grade(hpmapCol2, nside_out=nsideNew, order_in='NESTED', order_out='NESTED')
        ipnestTest = hp.ang2pix(nsideNew, theta, phi, nest=True) 

        # Degrade the old map
        newSparseMap = sparseMap.degrade(nside_out=nsideNew, primary='col1')
        testing.assert_almost_equal(newSparseMap.getValueRaDec(ra, dec)['col1'], hpmapCol1[ipnestTest])
        testing.assert_almost_equal(newSparseMap.getValueRaDec(ra, dec)['col2'], hpmapCol2[ipnestTest])

if __name__=='__main__':
    unittest.main()
