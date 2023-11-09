import unittest
import numpy.testing as testing
import numpy as np
import tempfile
import shutil
import os

import healsparse
from healsparse.fits_shim import HealSparseFits


class FitsShimTestCase(unittest.TestCase):
    def test_read_header(self):
        """
        Test reading a fits header
        """
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')
        filename = os.path.join(self.test_dir, 'test_array.fits')

        data0 = np.zeros(10, dtype=np.int32)
        data1 = np.zeros(10, dtype=np.float64)
        header = healsparse.fits_shim._make_header({'AA': 0,
                                                    'BB': 1.0,
                                                    'CC': 'test'})
        self.write_testfile(filename, data0, data1, header)

        with HealSparseFits(filename) as fits:
            exts = [0, 1, 'COV', 'SPARSE']
            for ext in exts:
                header_test = fits.read_ext_header(ext)
                for key in header:
                    self.assertEqual(header_test[key], header[key])

    def test_read_dtype(self):
        """
        Test reading a dtype
        """
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        header = None

        for t in ['array', 'recarray', 'widemask']:
            filename = os.path.join(self.test_dir, 'test_%s.fits' % (t))

            data0 = np.zeros(10, dtype=np.int32)

            if t == 'array':
                dtype = np.dtype(np.float64)
            elif t == 'recarray':
                dtype = np.dtype([('a', 'f4'),
                                  ('b', 'i4')])
            else:
                dtype = np.dtype(healsparse.WIDE_MASK)

            data1 = np.zeros(10, dtype=dtype)

            self.write_testfile(filename, data0, data1, header)

            with HealSparseFits(filename) as fits:
                exts = [1, 'SPARSE']
                for ext in exts:
                    dtype_test = fits.get_ext_dtype(ext)
                    if t == 'recarray':
                        # We need to allow for byte-order swapping
                        var_in = np.zeros(1, dtype=dtype)
                        var_out = np.zeros(1, dtype=dtype_test)

                        self.assertEqual(len(dtype_test), len(dtype))
                        for n in dtype.names:
                            self.assertEqual(var_in[n][0].__class__,
                                             var_out[n][0].__class__)
                    else:
                        self.assertEqual(dtype_test, dtype)

    def test_read_data(self):
        """
        Test reading data
        """
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        header = None

        for t in ['array', 'recarray', 'widemask']:
            filename = os.path.join(self.test_dir, 'test_%s.fits' % (t))

            data0 = np.ones(10, dtype=np.int32)

            if t == 'array':
                dtype = np.dtype(np.float64)
            elif t == 'recarray':
                dtype = np.dtype([('a', 'f4'),
                                  ('b', 'i4')])
            else:
                dtype = np.dtype(healsparse.WIDE_MASK)

            data1 = np.ones(10, dtype=dtype)

            self.write_testfile(filename, data0, data1, header)

            with HealSparseFits(filename) as fits:
                exts = [0, 'COV', 1, 'SPARSE']
                for ext in exts:
                    data_test = fits.read_ext_data(ext)
                    data_sub = fits.read_ext_data(ext, row_range=[2, 5])

                    if ext == 0 or ext == 'COV':
                        testing.assert_array_almost_equal(data_test, data0)
                        testing.assert_array_almost_equal(data_sub, data0[2: 5])
                    else:
                        if t == 'recarray':
                            for n in dtype.names:
                                testing.assert_array_almost_equal(data_test[n], data1[n])
                                testing.assert_array_almost_equal(data_sub[n], data1[n][2: 5])
                        else:
                            testing.assert_array_almost_equal(data_test, data1)
                            testing.assert_array_almost_equal(data_sub, data1[2: 5])

    def test_is_image(self):
        """
        Test if an extension is an image.
        """
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')
        filename = os.path.join(self.test_dir, 'test_array.fits')

        data0 = np.zeros(10, dtype=np.int32)
        data1 = np.zeros(10, dtype=[('a', 'f8'), ('b', 'i4')])
        header = None

        self.write_testfile(filename, data0, data1, header)

        with HealSparseFits(filename) as fits:
            exts = [0, 1, 'COV', 'SPARSE']
            is_images = [True, False, True, False]
            for i, ext in enumerate(exts):
                self.assertEqual(fits.ext_is_image(ext), is_images[i])

    def test_append(self):
        """
        Test if we can append
        """

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')
        header = None

        if healsparse.fits_shim.use_fitsio:
            # Test appending to image and recarray
            for t in ['array', 'recarray', 'widemask']:
                filename = os.path.join(self.test_dir, 'test_%s.fits' % (t))

                data0 = np.zeros(10, dtype=np.int32)

                if t == 'array':
                    dtype = np.float64
                elif t == 'recarray':
                    dtype = np.dtype([('a', 'f4'),
                                      ('b', 'i4')])
                else:
                    dtype = healsparse.WIDE_MASK

                data1 = np.zeros(10, dtype=dtype)

                self.write_testfile(filename, data0, data1, header)

                with HealSparseFits(filename, mode='rw') as fits:
                    extra_data1 = np.ones(10, dtype=dtype)
                    fits.append_extension(1, extra_data1)
                    full_data = np.append(data1, extra_data1)

                with HealSparseFits(filename) as fits:
                    data_test = fits.read_ext_data(1)
                    if t == 'recarray':
                        for n in dtype.names:
                            testing.assert_array_almost_equal(data_test[n], full_data[n])
                    else:
                        testing.assert_array_almost_equal(data_test, full_data)
        else:
            # Test that we get an error with astropy.io.fits
            filename = os.path.join(self.test_dir, 'test_array.fits')

            data0 = np.zeros(10, dtype=np.int32)
            data1 = np.zeros(10, dtype=np.float64)

            self.write_testfile(filename, data0, data1, header)

            self.assertRaises(RuntimeError, HealSparseFits, filename, mode='rw')

    def write_testfile_unused(self, filename, data0, data1, header):
        """
        Write a testfile, using astropy.io.fits only.  This is in place
        until we get full compression support working in both.
        """
        _header = healsparse.fits_shim._make_header(header)
        _header['EXTNAME'] = 'COV'
        healsparse.fits_shim.fits.writeto(filename, data0,
                                          header=_header)
        _header['EXTNAME'] = 'SPARSE'
        healsparse.fits_shim.fits.append(filename, data1,
                                         header=_header, overwrite=False)

    def write_testfile(self, filename, data0, data1, header):
        """
        Write a testfile.
        """
        if healsparse.fits_shim.use_fitsio:
            healsparse.fits_shim.fitsio.write(filename, data0,
                                              header=header, extname='COV')
            healsparse.fits_shim.fitsio.write(filename, data1,
                                              header=header, extname='SPARSE')
        else:
            _header = healsparse.fits_shim._make_header(header)
            _header['EXTNAME'] = 'COV'
            healsparse.fits_shim.fits.writeto(filename, data0,
                                              header=_header)
            _header['EXTNAME'] = 'SPARSE'
            healsparse.fits_shim.fits.append(filename, data1,
                                             header=_header, overwrite=False)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__ == '__main__':
    unittest.main()
