import unittest
import numpy.testing as testing
import numpy as np
import hpgeom as hpg

import healsparse


class ApplyMaskTestCase(unittest.TestCase):
    def test_apply_mask_int(self):
        """
        Test apply_mask on an integer map
        """

        # Create an integer mask map, using a box...

        nside_coverage = 32
        nside_sparse = 2**15

        box = healsparse.geom.Polygon(ra=[200.0, 200.2, 200.2, 200.0],
                                      dec=[10.0, 10.0, 10.2, 10.2],
                                      value=4)

        mask_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_sparse, np.int16, sentinel=0)
        healsparse.geom.realize_geom(box, mask_map)

        # Create an integer value map, using a bigger box...

        box2 = healsparse.geom.Polygon(ra=[199.8, 200.4, 200.4, 199.8],
                                       dec=[9.8, 9.8, 10.4, 10.4],
                                       value=1)
        int_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_sparse, np.int16, sentinel=0)
        healsparse.geom.realize_geom(box2, int_map)

        # First tests are with duplicates for reuse
        valid_pixels = int_map.valid_pixels

        # Default, mask all bits
        masked_map = int_map.apply_mask(mask_map, in_place=False)
        masked_pixels = mask_map.valid_pixels

        # Masked pixels should be zero
        testing.assert_array_equal(masked_map.get_values_pix(masked_pixels), 0)

        # Pixels that are in the original but are not in the masked pixels should be 1
        still_good0, = np.where((int_map.get_values_pix(valid_pixels) > 0) &
                                (mask_map.get_values_pix(valid_pixels) == 0))
        testing.assert_array_equal(masked_map.get_values_pix(valid_pixels[still_good0]), 1)

        # Pixels in the original map should all be 1
        testing.assert_array_equal(int_map.get_values_pix(valid_pixels), 1)

        # Mask specific bits (in the mask)
        masked_map = int_map.apply_mask(mask_map, mask_bits=4, in_place=False)
        masked_pixels = mask_map.valid_pixels

        # Masked pixels should be zero
        testing.assert_array_equal(masked_map.get_values_pix(masked_pixels), 0)

        # Pixels that are in the original but are not in the masked pixels should be 1
        still_good, = np.where((int_map.get_values_pix(valid_pixels) > 0) &
                               (mask_map.get_values_pix(valid_pixels) == 0))
        testing.assert_array_equal(masked_map.get_values_pix(valid_pixels[still_good]), 1)

        # Mask specific bits (not in the mask)
        masked_map = int_map.apply_mask(mask_map, mask_bits=16, in_place=False)
        masked_pixels = (mask_map & 16).valid_pixels

        testing.assert_equal(masked_pixels.size, 0)

        still_good, = np.where((int_map.get_values_pix(valid_pixels) > 0) &
                               ((mask_map.get_values_pix(valid_pixels) & 16) == 0))
        testing.assert_array_equal(masked_map.get_values_pix(valid_pixels[still_good]), 1)

        # Final test is in-place

        int_map.apply_mask(mask_map, in_place=True)
        masked_pixels = mask_map.valid_pixels

        # Masked pixels should be zero
        testing.assert_array_equal(int_map.get_values_pix(masked_pixels), 0)

        # Pixels that are in the original but are not in the masked pixels should be 1
        testing.assert_array_equal(int_map.get_values_pix(valid_pixels[still_good0]), 1)

    def test_apply_mask_float(self):
        """
        Test apply_mask on a float map
        """

        # Create an integer mask map, using a box...

        nside_coverage = 32
        nside_sparse = 2**15

        box = healsparse.geom.Polygon(ra=[200.0, 200.2, 200.2, 200.0],
                                      dec=[10.0, 10.0, 10.2, 10.2],
                                      value=4)

        mask_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_sparse, np.int16, sentinel=0)
        healsparse.geom.realize_geom(box, mask_map)

        # Create a float value map, using a bigger box...

        box2 = healsparse.geom.Polygon(ra=[199.8, 200.4, 200.4, 199.8],
                                       dec=[9.8, 9.8, 10.4, 10.4],
                                       value=1)
        int_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_sparse, np.int16, sentinel=0)
        healsparse.geom.realize_geom(box2, int_map)

        float_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_sparse, np.float32)
        float_map.update_values_pix(int_map.valid_pixels,
                                    np.ones(int_map.valid_pixels.size, dtype=np.float32))

        # First tests are with duplicates for reuse
        valid_pixels = float_map.valid_pixels

        # Default, mask all bits
        masked_map = float_map.apply_mask(mask_map, in_place=False)
        masked_pixels = mask_map.valid_pixels

        # Masked pixels should be zero
        testing.assert_almost_equal(masked_map.get_values_pix(masked_pixels), hpg.UNSEEN)

        # Pixels that are in the original but are not in the masked pixels should be 1
        still_good0, = np.where((float_map.get_values_pix(valid_pixels) > hpg.UNSEEN) &
                                (mask_map.get_values_pix(valid_pixels) == 0))
        testing.assert_almost_equal(masked_map.get_values_pix(valid_pixels[still_good0]), 1.0)

        # Pixels in the original map should all be 1
        testing.assert_almost_equal(float_map.get_values_pix(valid_pixels), 1.0)

        # Mask specific bits (in the mask)
        masked_map = float_map.apply_mask(mask_map, mask_bits=4, in_place=False)
        masked_pixels = mask_map.valid_pixels

        # Masked pixels should be zero
        testing.assert_almost_equal(masked_map.get_values_pix(masked_pixels), hpg.UNSEEN)

        # Pixels that are in the original but are not in the masked pixels should be 1
        still_good, = np.where((float_map.get_values_pix(valid_pixels) > 0) &
                               (mask_map.get_values_pix(valid_pixels) == 0))
        testing.assert_almost_equal(masked_map.get_values_pix(valid_pixels[still_good]), 1.0)

        # Mask specific bits (not in the mask)
        masked_map = float_map.apply_mask(mask_map, mask_bits=16, in_place=False)
        masked_pixels = (mask_map & 16).valid_pixels

        testing.assert_equal(masked_pixels.size, 0)

        still_good, = np.where((float_map.get_values_pix(valid_pixels) > 0) &
                               ((mask_map.get_values_pix(valid_pixels) & 16) == 0))
        testing.assert_almost_equal(masked_map.get_values_pix(valid_pixels[still_good]), 1.0)

        # Final test is in-place

        float_map.apply_mask(mask_map, in_place=True)
        masked_pixels = mask_map.valid_pixels

        # Masked pixels should be zero
        testing.assert_almost_equal(float_map.get_values_pix(masked_pixels), hpg.UNSEEN)

        # Pixels that are in the original but are not in the masked pixels should be 1
        testing.assert_almost_equal(float_map.get_values_pix(valid_pixels[still_good0]), 1.0)


if __name__ == '__main__':
    unittest.main()
