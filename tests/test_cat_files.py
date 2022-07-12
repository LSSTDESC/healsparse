import unittest
import numpy.testing as testing
import numpy as np
import hpgeom as hpg
import tempfile
import shutil
import os
import healsparse
from healsparse import cat_healsparse_files


class CatFilesTestCase(unittest.TestCase):
    def test_cat_maps(self):
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        # Make 3 maps of the same type.  These will have different nside_coverage
        # to exercise the up-scale and down-scaling.  The output nside_coverage
        # will be in the middle

        nside_coverage1 = 32
        nside_coverage2 = 64
        nside_coverage3 = 128

        nside_sparse = 1024

        nside_coverage_out = 64

        for t in ['array', 'recarray', 'widemask']:
            if t == 'array':
                dtype = np.float64
                primary = None
                wide_mask_maxbits = None
                sentinel = hpg.UNSEEN
                data = np.array([100.0], dtype=dtype)
            elif t == 'recarray':
                dtype = [('a', 'f4'),
                         ('b', 'i4')]
                primary = 'a'
                wide_mask_maxbits = None
                sentinel = hpg.UNSEEN
                data = np.zeros(1, dtype=dtype)
                data['a'] = 100.0
                data['b'] = 100
            else:
                dtype = healsparse.WIDE_MASK
                primary = None
                wide_mask_maxbits = 24
                sentinel = 0

            map1 = healsparse.HealSparseMap.make_empty(nside_coverage1, nside_sparse,
                                                       dtype, primary=primary,
                                                       sentinel=sentinel,
                                                       wide_mask_maxbits=wide_mask_maxbits)
            if t == 'widemask':
                map1.set_bits_pix(np.arange(10000, 15000), [1, 12])
            else:
                map1[10000: 15000] = data

            map2 = healsparse.HealSparseMap.make_empty(nside_coverage2, nside_sparse,
                                                       dtype, primary=primary,
                                                       sentinel=sentinel,
                                                       wide_mask_maxbits=wide_mask_maxbits)
            if t == 'widemask':
                map2.set_bits_pix(np.arange(15001, 20000), [1, 12])
            else:
                map2[15001: 20000] = data

            map3 = healsparse.HealSparseMap.make_empty(nside_coverage3, nside_sparse,
                                                       dtype, primary=primary,
                                                       sentinel=sentinel,
                                                       wide_mask_maxbits=wide_mask_maxbits)
            if t == 'widemask':
                map3.set_bits_pix(np.arange(20001, 25000), [1, 12])
            else:
                map3[20001: 25000] = data

            # Make a combined map for comparison

            # This should have the same nside_coverage as map2
            combined_map = healsparse.HealSparseMap.make_empty_like(map1,
                                                                    nside_coverage=nside_coverage_out)
            combined_map[map1.valid_pixels] = map1[map1.valid_pixels]
            combined_map[map2.valid_pixels] = map2[map2.valid_pixels]
            combined_map[map3.valid_pixels] = map3[map3.valid_pixels]

            filename1 = os.path.join(self.test_dir, 'test_%s_1.hs' % (t))
            filename2 = os.path.join(self.test_dir, 'test_%s_2.hs' % (t))
            filename3 = os.path.join(self.test_dir, 'test_%s_3.hs' % (t))
            map1.write(filename1)
            map2.write(filename2)
            map3.write(filename3)

            file_list = [filename1, filename2, filename3]

            for in_mem in [False, True]:
                outfile = os.path.join(self.test_dir, 'test_%s_combined_%d.hs' %
                                       (t, int(in_mem)))

                if not healsparse.fits_shim.use_fitsio and not in_mem:
                    # We cannot use out-of-memory option with astropy.io.fits
                    self.assertRaises(RuntimeError, cat_healsparse_files,
                                      file_list, outfile, in_memory=in_mem,
                                      nside_coverage_out=nside_coverage_out)
                else:
                    cat_healsparse_files(file_list, outfile, in_memory=in_mem,
                                         nside_coverage_out=nside_coverage_out)

                    map_test = healsparse.HealSparseMap.read(outfile)

                    testing.assert_array_equal(map_test.valid_pixels, combined_map.valid_pixels)
                    if t == 'recarray':
                        for col in map_test.dtype.names:
                            testing.assert_array_almost_equal(map_test[:][col],
                                                              combined_map[:][col])
                    else:
                        testing.assert_array_almost_equal(map_test[:], combined_map[:])

    def test_cat_maps_overlap(self):
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHealSparse-')

        # Make 3 maps of the same type.  These will have different nside_coverage
        # to exercise the up-scale and down-scaling.  The output nside_coverage
        # will be in the middle, now we test explicitly for overlap and should
        # raise an error unless the maps are integer

        nside_coverage1 = 32
        nside_coverage2 = 64
        nside_coverage3 = 128

        nside_sparse = 1024

        nside_coverage_out = 64

        for t in ['array']:  # 'recarray', 'widemask']:
            if t == 'array':
                dtype = np.int32
                primary = None
                wide_mask_maxbits = None
                sentinel = -9999
                data = np.array([100.0], dtype=dtype)
            elif t == 'recarray':
                dtype = [('a', 'f4'),
                         ('b', 'i4')]
                primary = 'a'
                wide_mask_maxbits = None
                sentinel = hpg.UNSEEN
                data = np.zeros(1, dtype=dtype)
                data['a'] = 100.0
                data['b'] = 100
            else:
                dtype = healsparse.WIDE_MASK
                primary = None
                wide_mask_maxbits = 24
                sentinel = 0
            map1 = healsparse.HealSparseMap.make_empty(nside_coverage1, nside_sparse,
                                                       dtype, primary=primary,
                                                       sentinel=sentinel,
                                                       wide_mask_maxbits=wide_mask_maxbits)
            if t == 'widemask':
                map1.set_bits_pix(np.arange(10000, 15000), [1, 12])
            else:
                map1[10000: 15000] = data

            map2 = healsparse.HealSparseMap.make_empty(nside_coverage2, nside_sparse,
                                                       dtype, primary=primary,
                                                       sentinel=sentinel,
                                                       wide_mask_maxbits=wide_mask_maxbits)
            if t == 'widemask':
                map2.set_bits_pix(np.arange(14500, 20000), [1, 12])
            else:
                map2[14500: 20000] = data

            map3 = healsparse.HealSparseMap.make_empty(nside_coverage3, nside_sparse,
                                                       dtype, primary=primary,
                                                       sentinel=sentinel,
                                                       wide_mask_maxbits=wide_mask_maxbits)
            if t == 'widemask':
                map3.set_bits_pix(np.arange(19500, 25000), [1, 12])
            else:
                map3[19500: 25000] = data

            # Make a combined map for comparison

            # This should have the same nside_coverage as map2
            combined_map = healsparse.HealSparseMap.make_empty_like(map1,
                                                                    nside_coverage=nside_coverage_out,
                                                                    sentinel=sentinel)
            if combined_map.is_integer_map:
                combined_map[map1.valid_pixels] = map1[map1.valid_pixels]
                overlap = np.in1d(map2.valid_pixels, combined_map.valid_pixels)
                combined_map[map2.valid_pixels[overlap]] = map2[map2.valid_pixels[overlap]] | \
                    combined_map[map2.valid_pixels[overlap]]
                combined_map[map2.valid_pixels[~overlap]] = map2[map2.valid_pixels[~overlap]]
                overlap = np.in1d(map3.valid_pixels, combined_map.valid_pixels)
                combined_map[map3.valid_pixels[overlap]] = map3[map3.valid_pixels[overlap]] | \
                    combined_map[map3.valid_pixels[overlap]]
                combined_map[map3.valid_pixels[~overlap]] = map3[map3.valid_pixels[~overlap]]

            filename1 = os.path.join(self.test_dir, 'test_%s_1.hs' % (t))
            filename2 = os.path.join(self.test_dir, 'test_%s_2.hs' % (t))
            filename3 = os.path.join(self.test_dir, 'test_%s_3.hs' % (t))
            map1.write(filename1)
            map2.write(filename2)
            map3.write(filename3)

            file_list = [filename1, filename2, filename3]

            for in_mem in [False, True]:
                outfile = os.path.join(self.test_dir, 'test_%s_combined_%d.hs' %
                                       (t, int(in_mem)))

                if not healsparse.fits_shim.use_fitsio and not in_mem:
                    # We cannot use out-of-memory option with astropy.io.fits
                    self.assertRaises(RuntimeError, cat_healsparse_files,
                                      file_list, outfile, in_memory=in_mem,
                                      nside_coverage_out=nside_coverage_out,
                                      check_overlap=True, or_overlap=True)
                else:
                    if not combined_map.is_integer_map:
                        self.assertRaises(RuntimeError, cat_healsparse_files(file_list,
                                          outfile, in_memory=in_mem,
                                          nside_coverage_out=nside_coverage_out,
                                          check_overlap=True, or_overlap=True))
                    else:
                        cat_healsparse_files(file_list,
                                             outfile, in_memory=in_mem,
                                             nside_coverage_out=nside_coverage_out,
                                             check_overlap=True, or_overlap=True)
                        map_test = healsparse.HealSparseMap.read(outfile)
                        testing.assert_array_equal(map_test.valid_pixels, combined_map.valid_pixels)
                        testing.assert_array_almost_equal(map_test[map_test.valid_pixels],
                                                          combined_map[combined_map.valid_pixels])

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__ == '__main__':
    unittest.main()
