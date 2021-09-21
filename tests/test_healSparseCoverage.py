import unittest
import numpy.testing as testing
import numpy as np
import healpy as hp
import tempfile
import shutil
import os
import pytest

import healsparse


class HealSparseCoverageTestCase(unittest.TestCase):
    def test_read_fits_coverage(self):
        """
        Test reading healSparseCoverage from a fits file.
        """
        nside_coverage = 32
        nside_map = 64

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')
        fname = os.path.join(self.test_dir, 'healsparse_map_test.hsp')

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         dtype=np.float32)

        sparse_map[0: 20000] = np.random.random(size=20000).astype(np.float32)

        sparse_map.write(fname)

        # Generate a coverage mask from the 0: 20000
        cov_mask_test = np.zeros(hp.nside2npix(nside_coverage), dtype=np.bool_)
        theta, phi = hp.pix2ang(nside_map, np.arange(20000), nest=True)
        ipnest = np.unique(hp.ang2pix(nside_coverage, theta, phi, nest=True))
        cov_mask_test[ipnest] = True

        cov_map = healsparse.HealSparseCoverage.read(fname)

        # Ensure that the coverage mask is what we think it should be
        testing.assert_array_equal(cov_map.coverage_mask, cov_mask_test)

        # Ensure that we can address the cov_map by index
        testing.assert_array_equal(cov_map[:], cov_map._cov_index_map)
        testing.assert_array_equal(cov_map[0: 100], cov_map._cov_index_map[0: 100])
        testing.assert_array_equal([cov_map[0]], [cov_map._cov_index_map[0]])

        # Make a healpy file and make sure we can't read it
        test_map = np.zeros(hp.nside2npix(nside_coverage))
        fname = os.path.join(self.test_dir, 'healpy_map_test.fits')
        hp.write_map(fname, test_map)

        self.assertRaises(RuntimeError, healsparse.HealSparseCoverage.read, fname)

    @pytest.mark.skipif(not healsparse.parquet_shim.use_pyarrow, reason='Requires pyarrow')
    def test_read_parquet_coverage(self):
        """
        Test reading healSparseCoverage from a parquet dataset.
        """
        nside_coverage = 32
        nside_map = 64

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')
        fname = os.path.join(self.test_dir, 'healsparse_map_test.hsparquet')

        sparse_map = healsparse.HealSparseMap.make_empty(nside_coverage, nside_map,
                                                         dtype=np.float32)

        sparse_map[0: 20000] = np.random.random(size=20000).astype(np.float32)

        sparse_map.write(fname, format='parquet')

        # Generate a coverage mask from the 0: 20000
        cov_mask_test = np.zeros(hp.nside2npix(nside_coverage), dtype=np.bool_)
        theta, phi = hp.pix2ang(nside_map, np.arange(20000), nest=True)
        ipnest = np.unique(hp.ang2pix(nside_coverage, theta, phi, nest=True))
        cov_mask_test[ipnest] = True

        cov_map = healsparse.HealSparseCoverage.read(fname)

        # Ensure that the coverage mask is what we think it should be
        testing.assert_array_equal(cov_map.coverage_mask, cov_mask_test)

        # Ensure that we can address the cov_map by index
        testing.assert_array_equal(cov_map[:], cov_map._cov_index_map)
        testing.assert_array_equal(cov_map[0: 100], cov_map._cov_index_map[0: 100])
        testing.assert_array_equal([cov_map[0]], [cov_map._cov_index_map[0]])

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__ == '__main__':
    unittest.main()
