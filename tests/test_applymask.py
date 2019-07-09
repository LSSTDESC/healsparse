from __future__ import division, absolute_import, print_function

import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
from numpy import random

import healsparse

class ApplyMaskTestCase(unittest.TestCase):
    def test_apply_mask_int(self):
        """
        Test applyMask on an integer map
        """

        # Create an integer mask map, using a box...

        nsideCoverage = 32
        nsideSparse = 2**15

        box = healsparse.geom.Polygon(ra=[200.0, 200.2, 200.2, 200.0],
                                      dec=[10.0, 10.0, 10.2, 10.2],
                                      value=4)

        maskMap = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideSparse, np.int16, sentinel=0)
        healsparse.geom.realize_geom(box, maskMap)

        # Create an integer value map, using a bigger box...

        box2 = healsparse.geom.Polygon(ra=[199.8, 200.4, 200.4, 199.8],
                                       dec=[9.8, 9.8, 10.4, 10.4],
                                       value=1)
        intMap = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideSparse, np.int16, sentinel=0)
        healsparse.geom.realize_geom(box2, intMap)

        # First tests are with duplicates for reuse
        validPixels = intMap.validPixels

        # Default, mask all bits
        maskedMap = intMap.applyMask(maskMap, inPlace=False)
        maskedPixels = maskMap.validPixels

        # Masked pixels should be zero
        testing.assert_array_equal(maskedMap.getValuePixel(maskedPixels), 0)

        # Pixels that are in the original but are not in the masked pixels should be 1
        stillGood0, = np.where((intMap.getValuePixel(validPixels) > 0) &
                               (maskMap.getValuePixel(validPixels) == 0))
        testing.assert_array_equal(maskedMap.getValuePixel(validPixels[stillGood0]), 1)

        # Pixels in the original map should all be 1
        testing.assert_array_equal(intMap.getValuePixel(validPixels), 1)

        # Mask specific bits (in the mask)
        maskedMap = intMap.applyMask(maskMap, maskBits=4, inPlace=False)
        maskedPixels = maskMap.validPixels

        # Masked pixels should be zero
        testing.assert_array_equal(maskedMap.getValuePixel(maskedPixels), 0)

        # Pixels that are in the original but are not in the masked pixels should be 1
        stillGood, = np.where((intMap.getValuePixel(validPixels) > 0) &
                              (maskMap.getValuePixel(validPixels) == 0))
        testing.assert_array_equal(maskedMap.getValuePixel(validPixels[stillGood]), 1)

        # Mask specific bits (not in the mask)
        maskedMap = intMap.applyMask(maskMap, maskBits=16, inPlace=False)
        maskedPixels = (maskMap & 16).validPixels

        testing.assert_equal(maskedPixels.size, 0)

        stillGood, = np.where((intMap.getValuePixel(validPixels) > 0) &
                              ((maskMap.getValuePixel(validPixels) & 16) == 0))
        testing.assert_array_equal(maskedMap.getValuePixel(validPixels[stillGood]), 1)

        # Final test is in-place

        intMap.applyMask(maskMap, inPlace=True)
        maskedPixels = maskMap.validPixels

        # Masked pixels should be zero
        testing.assert_array_equal(intMap.getValuePixel(maskedPixels), 0)

        # Pixels that are in the original but are not in the masked pixels should be 1
        testing.assert_array_equal(intMap.getValuePixel(validPixels[stillGood0]), 1)

    def test_apply_mask_float(self):
        """
        Test applyMask on a float map
        """

        # Create an integer mask map, using a box...

        nsideCoverage = 32
        nsideSparse = 2**15

        box = healsparse.geom.Polygon(ra=[200.0, 200.2, 200.2, 200.0],
                                      dec=[10.0, 10.0, 10.2, 10.2],
                                      value=4)

        maskMap = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideSparse, np.int16, sentinel=0)
        healsparse.geom.realize_geom(box, maskMap)

        # Create a float value map, using a bigger box...

        box2 = healsparse.geom.Polygon(ra=[199.8, 200.4, 200.4, 199.8],
                                       dec=[9.8, 9.8, 10.4, 10.4],
                                       value=1)
        intMap = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideSparse, np.int16, sentinel=0)
        healsparse.geom.realize_geom(box2, intMap)

        floatMap = healsparse.HealSparseMap.makeEmpty(nsideCoverage, nsideSparse, np.float32)
        floatMap.updateValues(intMap.validPixels, np.ones(intMap.validPixels.size, dtype=np.float32))

        # First tests are with duplicates for reuse
        validPixels = floatMap.validPixels

        # Default, mask all bits
        maskedMap = floatMap.applyMask(maskMap, inPlace=False)
        maskedPixels = maskMap.validPixels

        # Masked pixels should be zero
        testing.assert_almost_equal(maskedMap.getValuePixel(maskedPixels), hp.UNSEEN)

        # Pixels that are in the original but are not in the masked pixels should be 1
        stillGood0, = np.where((floatMap.getValuePixel(validPixels) > hp.UNSEEN) &
                               (maskMap.getValuePixel(validPixels) == 0))
        testing.assert_almost_equal(maskedMap.getValuePixel(validPixels[stillGood0]), 1.0)

        # Pixels in the original map should all be 1
        testing.assert_almost_equal(floatMap.getValuePixel(validPixels), 1.0)

        # Mask specific bits (in the mask)
        maskedMap = floatMap.applyMask(maskMap, maskBits=4, inPlace=False)
        maskedPixels = maskMap.validPixels

        # Masked pixels should be zero
        testing.assert_almost_equal(maskedMap.getValuePixel(maskedPixels), hp.UNSEEN)

        # Pixels that are in the original but are not in the masked pixels should be 1
        stillGood, = np.where((floatMap.getValuePixel(validPixels) > 0) &
                              (maskMap.getValuePixel(validPixels) == 0))
        testing.assert_almost_equal(maskedMap.getValuePixel(validPixels[stillGood]), 1.0)

        # Mask specific bits (not in the mask)
        maskedMap = floatMap.applyMask(maskMap, maskBits=16, inPlace=False)
        maskedPixels = (maskMap & 16).validPixels

        testing.assert_equal(maskedPixels.size, 0)

        stillGood, = np.where((floatMap.getValuePixel(validPixels) > 0) &
                              ((maskMap.getValuePixel(validPixels) & 16) == 0))
        testing.assert_almost_equal(maskedMap.getValuePixel(validPixels[stillGood]), 1.0)

        # Final test is in-place

        floatMap.applyMask(maskMap, inPlace=True)
        maskedPixels = maskMap.validPixels

        # Masked pixels should be zero
        testing.assert_almost_equal(floatMap.getValuePixel(maskedPixels), hp.UNSEEN)

        # Pixels that are in the original but are not in the masked pixels should be 1
        testing.assert_almost_equal(floatMap.getValuePixel(validPixels[stillGood0]), 1.0)


if __name__=='__main__':
    unittest.main()

