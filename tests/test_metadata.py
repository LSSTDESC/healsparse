import unittest
import numpy as np
import tempfile
import shutil
import os
import warnings

import healsparse


class MetadataTestCase(unittest.TestCase):
    """
    Test code for metadata reading/writing.
    """
    def test_metadata_make_empty(self):
        """
        Test metadata passing on make_empty and make_empty_like
        """
        nside_coverage = 32
        nside_map = 64

        metadata_in = {'A': 15,
                       'B': 'test',
                       'LONGKEYNAME': 4.5}

        test_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                       np.float64, metadata=metadata_in)
        metadata_out = test_map.metadata

        for key in metadata_in:
            self.assertEqual(metadata_out[key], metadata_in[key])

        test_map2 = healsparse.HealSparseMap.make_empty_like(test_map)
        metadata_out = test_map2.metadata

        for key in metadata_in:
            self.assertEqual(metadata_out[key], metadata_in[key])

        metadata_in2 = {'A': 15,
                        'B': 'test',
                        'LONGKEYNAME': 5.5}

        test_map3 = healsparse.HealSparseMap.make_empty_like(test_map, metadata=metadata_in2)
        metadata_out = test_map3.metadata

        for key in metadata_in2:
            self.assertEqual(metadata_out[key], metadata_in2[key])

    def test_metadata_bad(self):
        """
        Test setting bad metadata
        """
        nside_coverage = 32
        nside_map = 64

        # Not a dict
        metadata = '5'
        self.assertRaises(ValueError,
                          healsparse.HealSparseMap.make_empty,
                          nside_coverage, nside_map, np.float64,
                          metadata=metadata)

        # Not a string key
        metadata = {7: 7}
        self.assertRaises(ValueError,
                          healsparse.HealSparseMap.make_empty,
                          nside_coverage, nside_map, np.float64,
                          metadata=metadata)

        # Not upper-case
        metadata = {'test': 7}
        self.assertRaises(ValueError,
                          healsparse.HealSparseMap.make_empty,
                          nside_coverage, nside_map, np.float64,
                          metadata=metadata)

    def test_metadata_getset(self):
        """
        Test setting and getting metadata
        """
        nside_coverage = 32
        nside_map = 64

        metadata_in = {'A': 15,
                       'B': 'test',
                       'LONGKEYNAME': 4.5}

        test_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map, np.float64)
        test_map.metadata = metadata_in
        metadata_out = test_map.metadata

        for key in metadata_in:
            self.assertEqual(metadata_out[key], metadata_in[key])

    def test_metadata_writeread(self):
        """
        Test writing and reading metadata
        """
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        nside_coverage = 32
        nside_map = 64

        metadata_in = {'A': 15,
                       'B': 'test',
                       'LONGKEYNAME': 4.5}
        test_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                       np.float64, metadata=metadata_in)

        fname = os.path.join(self.test_dir, 'healsparse_map_metadata.fits')

        # Make sure LONGKEYNAME doesn't give any warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            test_map.write(fname)

        test_map = healsparse.HealSparseMap.read(fname)
        metadata_out = test_map.metadata

        for key in metadata_in:
            self.assertEqual(metadata_out[key], metadata_in[key])

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__ == '__main__':
    unittest.main()
