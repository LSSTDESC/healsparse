import numpy as np
import mmap
from .utils import is_integer_value
# We need this for compression before a newer version of fitsio arrives
import astropy.io.fits as fits

use_fitsio = False
try:
    import fitsio
    use_fitsio = True
except ImportError:
    pass


_image_bitpix2npy = {
    8: 'u1',
    10: 'i1',
    16: 'i2',
    20: 'u2',
    32: 'i4',
    40: 'u4',
    64: 'i8',
    -32: 'f4',
    -64: 'f8'}


FITS_RESERVED = ['TFIELDS', 'TTYPE1', 'TFORM1', 'ZIMAGE',
                 'ZTENSION', 'ZBITPIX', 'ZNAXIS', 'ZNAXIS1',
                 'ZPCOUNT', 'ZGCOUNT', 'ZTILE1', 'ZCMPTYPE',
                 'ZNAME1', 'ZVAL1', 'ZQUANTIZ',
                 'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2',
                 'PCOUNT', 'GCOUNT']


class HealSparseFits(object):
    """
    Wrapper class to handle fitsio or astropy.io.fits
    """
    def __init__(self, filename, mode='r'):
        """
        Instantiate a HealSparseFits object

        Parameters
        ----------
        filename : `str`
           Name of file to open
        mode : `str`
           'r' or 'rw'

        Returns
        -------
        fits_object : `healsparse.HealSparseFits`
        """
        self._filename = filename
        self._mode = mode

        if use_fitsio:
            self.fits_object = fitsio.FITS(filename, mode=mode)
        else:
            if mode == 'r':
                fits_mode = 'readonly'
            else:
                raise RuntimeError('Readonly is only useful mode supported for astropy.io.fits')
            self.fits_object = fits.open(filename, memmap=True, lazy_load_hdus=True,
                                         mode=fits_mode)

    def read_ext_header(self, extension):
        """
        Read the header from a given extension (name or number)

        Parameters
        ----------
        extension : `int` or `str`
           Extension name or number

        Returns
        -------
        header : `fitsio.FITSHDR` or `FIXME`
        """
        if use_fitsio:
            return self.fits_object[extension].read_header()
        else:
            return self.fits_object[extension].header

    def get_ext_dtype(self, extension):
        """
        Get the datatype for a given extension

        Parameters
        ----------
        extension : `int` or `str`
           Extension name or number

        Returns
        -------
        dtype : `np.dtype`
        """
        if use_fitsio:
            hdu = self.fits_object[extension]
            if hdu.get_exttype() == 'IMAGE_HDU':
                return _image_bitpix2npy[hdu.get_info()['img_equiv_type']]
            else:
                return self.fits_object[extension].get_rec_dtype()[0]
        else:
            hdu = self.fits_object[extension]
            if hdu.is_image:
                return _image_bitpix2npy[hdu._bitpix]
            else:
                return hdu.data[0: 1].dtype

    def read_ext_data(self, extension, row_range=None):
        """
        Get data from a fits extension.

        Parameters
        ----------
        extension : `int` or `str`
           Extension name or number
        row_range : `list`, 2 elements, optional
           row range to create a slice.  Default is None (read all).

        Returns
        -------
        data : `np.ndarray`
        """
        if use_fitsio:
            hdu = self.fits_object[extension]
            if row_range is None:
                return hdu.read()
            else:
                return hdu[slice(row_range[0], row_range[1])]
        else:
            hdu = self.fits_object[extension]
            if row_range is None:
                return hdu.data.view(np.ndarray)
            else:
                return hdu.data[slice(row_range[0], row_range[1])].view(np.ndarray)

    def ext_is_image(self, extension):
        """
        Is a given extension an image HDU?

        Parameters
        ----------
        extension : `int` or `str`
           Extension name or number

        Returns
        -------
        is_image : `bool`
        """
        hdu = self.fits_object[extension]
        if use_fitsio:
            if hdu.get_exttype() == 'IMAGE_HDU':
                return True
            else:
                return False
        else:
            return hdu.is_image

    def append_extension(self, extension, data):
        """
        Append data to extension.

        Parameters
        ----------
        extension : `int` or `str`
           Extension name or number
        data : `np.ndarray`
           Data to append
        """
        if self._mode != 'rw':
            raise RuntimeError("Appending only allowed in rw mode")

        hdu = self.fits_object[extension]
        if use_fitsio:
            if hasattr(hdu, 'append'):
                # A recarray that we can append to
                hdu.append(data)
            else:
                # An image that we cannot append to
                firstrow = (hdu.get_dims()[0], )
                hdu.write(data, start=firstrow)
        else:
            raise RuntimeError("Appending is not supported by astropy.io.fits")

    def close(self):
        self.fits_object.close()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.fits_object.close()


def _write_filename(filename, c_hdr, s_hdr, cov_index_map, sparse_map,
                    compress=False, compress_tilesize=None):
    """
    Write to a filename, using fitsio or astropy.io.fits.

    This assumes that you want to overwrite any existing file (as
    should be checked in calling function).

    Parameters
    ----------
    filename : `str`
       Name of file to write to.
    c_hdr : `fitsio.FITSHDR` or `astropy.io.fits.Header`
       Coverage index map header
    s_hdr : `fitsio.FITSHDR` or `astropy.io.fits.Header`
       Sparse map header
    cov_index_map : `np.ndarray`
       Coverage index map
    sparse_map : `np.ndarray`
       Sparse map
    compress : `bool`, optional
       Write with FITS compression?
    """
    # Currently, all writing is done with astropy.io.fits because it supports
    # lossless compression of floating point data.  Unfortunately, the header
    # is wrong so we have a header patch below.

    c_hdr['EXTNAME'] = 'COV'
    s_hdr['EXTNAME'] = 'SPARSE'

    hdu_list = fits.HDUList()

    hdu = fits.PrimaryHDU(data=cov_index_map, header=fits.Header())
    for n in c_hdr:
        if n not in FITS_RESERVED:
            hdu.header[n] = c_hdr[n]
    hdu_list.append(hdu)

    if compress:
        hdu = fits.CompImageHDU(data=sparse_map, header=fits.Header(),
                                compression_type='GZIP_2',
                                tile_size=(compress_tilesize, ),
                                quantize_level=0.0)
    else:
        if sparse_map.dtype.fields is not None:
            hdu = fits.BinTableHDU(data=sparse_map, header=fits.Header())
        else:
            hdu = fits.ImageHDU(data=sparse_map, header=fits.Header())

    for n in s_hdr:
        if n not in FITS_RESERVED:
            hdu.header[n] = s_hdr[n]
    hdu_list.append(hdu)

    hdu_list.writeto(filename, overwrite=True)

    # When writing a gzip unquantized (lossless) floating point image,
    # current versions of astropy (4.0.1 and earlier, at least) write
    # the ZQUANTIZ header value as NO_DITHER, while cfitsio expects
    # this to be NONE for unquantized data.  The only way to overwrite
    # this reserved header keyword is to manually overwrite the bytes
    # in the file.  The following code uses mmap to overwrite the
    # necessary header keyword without loading the full image into
    # memory.  Note that healsparse files only have one compressed
    # extension, so there will only be one use of ZQUANTIZ in the file.
    if compress and not is_integer_value(sparse_map[0]):
        with open(filename, "r+b") as f:
            try:
                mm = mmap.mmap(f.fileno(), 0)
                loc = mm.find(b"ZQUANTIZ= 'NO_DITHER'")
                if loc >= 0:
                    mm.seek(loc)
                    mm.write(b"ZQUANTIZ= 'NONE     '")
            except OSError:
                # Some systems do not have the mmap available,
                # we need to read in the full file.
                data = f.read()
                loc = data.find(b"ZQUANTIZ= 'NO_DITHER'")
                if loc >= 0:
                    f.seek(loc)
                    f.write(b"ZQUANTIZ= 'NONE     '")


def _make_header(metadata):
    """
    Make a fits header.

    Parameters
    ----------
    metadata : `dict`-like object
       Input metadata

    Returns
    -------
    header : `fitsio.FITSHDR` or `astropy.io.fits.Header`
    """
    # All headers are astropy headers until we update fitsio
    # if use_fitsio:
    #     hdr = fitsio.FITSHDR(metadata)
    if metadata is None:
        hdr = fits.Header()
    else:
        hdr = fits.Header(metadata)

    return hdr


def _write_healpix_filename(filename, hdr, output_struct):
    """
    Write to a filename, HEALPix EXPLICIT format, using astropy.io.fits.

    This assumes that you want to overwrite any existing file (as should be
    checked in the calling function.)

    Parameters
    ----------
    filename : `str`
        Name of file to write to.
    hdr : `astropy.io.fits.Header`
        Correctly formatted header.
    output_struct : `numpy.recarray`
        Correctly formatted output struct.
    """
    hdu_list = fits.HDUList()

    hdu = fits.BinTableHDU(data=output_struct, header=fits.Header())

    for n in hdr:
        if n not in FITS_RESERVED:
            hdu.header[n] = hdr[n]
    hdu_list.append(hdu)

    hdu_list.writeto(filename, overwrite=True)
