import copy
import numpy as np
from .utils import is_integer_value

use_rustfits = False
use_fitsio = False
has_astropy = False

try:
    import rustfits
    use_rustfits = True
except ImportError:
    pass

if not use_rustfits:
    try:
        import fitsio
        use_fitsio = True
    except ImportError:
        pass

try:
    import astropy.io.fits as astropy_fits
    has_astropy = True
except ImportError:
    pass

if not use_rustfits and not use_fitsio and not has_astropy:
    raise ImportError("HealSparse requires rustfits or fitsio or astropy to be installed.")


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
                 'ZTENSION', 'ZBITPIX', 'ZNAXIS', 'ZNAXIS1', 'ZNAXIS2',
                 'ZPCOUNT', 'ZGCOUNT', 'ZTILE1', 'ZTILE2', 'ZCMPTYPE',
                 'ZNAME1', 'ZNAME2', 'ZVAL1', 'ZVAL2', 'ZQUANTIZ',
                 'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2',
                 'PCOUNT', 'GCOUNT', 'XTENSION']


class HealSparseFits(object):
    """
    Wrapper class to handle rustfits, fitsio, or astropy.io.fits
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

        if use_rustfits:
            if mode == "r":
                rustfits_mode = "r"
            elif mode == "rw":
                rustfits_mode = "r+"

            self.fits_object = rustfits.FITS(filename, mode=rustfits_mode)

            try:
                _ = self.fits_object[0]
            except ValueError:
                raise IOError("File %s does not appear to be a fits file." % (filename))
        elif use_fitsio:
            self.fits_object = fitsio.FITS(filename, mode=mode)
        elif has_astropy:
            if mode == 'r':
                fits_mode = 'readonly'
            else:
                raise RuntimeError('Readonly is only useful mode supported for astropy.io.fits')
            self.fits_object = astropy_fits.open(filename, memmap=True, lazy_load_hdus=True,
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
            # This works for rustfits and astropy fits.
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
        if use_rustfits:
            hdu = self.fits_object[extension]
            if self.ext_is_image(extension):
                return hdu.dtype.str
            else:
                return hdu.dtype
        elif use_fitsio:
            hdu = self.fits_object[extension]
            if self.ext_is_image(extension):
                return _image_bitpix2npy[hdu.get_info()['img_equiv_type']]
            else:
                return self.fits_object[extension].get_rec_dtype()[0]
        elif has_astropy:
            hdu = self.fits_object[extension]
            if hdu.is_image:
                return _image_bitpix2npy[hdu._bitpix]
            else:
                return hdu.data[0: 1].dtype

    def read_ext_data(self, extension, row_range=None, col_range=None):
        """
        Get data from a fits extension.

        Parameters
        ----------
        extension : `int` or `str`
           Extension name or number
        row_range : `list`, 2 elements, optional
           row range to create a slice.  Default is None (read all).
        col_range: `list`, 2 elements, optional
           column range to create a slice.  Default is None (read all).
           only used if row_range is also set.

        Returns
        -------
        data : `np.ndarray`
        """
        if use_rustfits or use_fitsio:
            hdu = self.fits_object[extension]
            if row_range is None:
                return hdu.read()
            elif col_range is None:
                return hdu[slice(row_range[0], row_range[1])]
            else:
                return hdu[slice(col_range[0], col_range[1]),
                           slice(row_range[0], row_range[1])]
        elif has_astropy:
            # Note that for astropy this does not actually seem to work
            # read a subregion from a tile-compressed image; it reads
            # the full thing.
            hdu = self.fits_object[extension]
            if row_range is None:
                return hdu.data.view(np.ndarray)
            elif col_range is None:
                try:
                    return hdu.section[slice(row_range[0], row_range[1])].view(np.ndarray)
                except AttributeError:
                    return hdu.data[slice(row_range[0], row_range[1])].view(np.ndarray)
            else:
                try:
                    return hdu.section[slice(col_range[0], col_range[1]),
                                       slice(row_range[0], row_range[1])].view(np.ndarray)
                except AttributeError:
                    return hdu.data[slice(col_range[0], col_range[1]),
                                    slice(row_range[0], row_range[1])].view(np.ndarray)

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
        if use_rustfits:
            return isinstance(hdu, rustfits.ImageHDU)
        elif use_fitsio:
            if hdu.get_exttype() == 'IMAGE_HDU':
                return True
            else:
                return False
        elif has_astropy:
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
        if use_rustfits:
            if hasattr(hdu, 'extend'):
                # We can append
                hdu.extend(data)
            else:
                firstrow = (hdu.get_dims()[0], )
                hdu.write(data, start=firstrow)
        elif use_fitsio:
            if hasattr(hdu, 'append'):
                # A recarray that we can append to
                hdu.append(data)
            else:
                # An image that we cannot append to
                firstrow = (hdu.get_dims()[0], )
                hdu.write(data, start=firstrow)
        elif has_astropy:
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
    c_hdr['EXTNAME'] = 'COV'
    s_hdr['EXTNAME'] = 'SPARSE'

    integer_map = sparse_map.dtype.fields is None and is_integer_value(sparse_map.dtype.type(0))
    if integer_map:
        compression = "RICE_1"
    else:
        compression = "GZIP_2"

    if compress:
        # The maximum that ZNAXIS1 can be is 2**31 - 1 when using FITS
        # compression.  Therefore, we do a reshape here if necessary,
        # and also record that there has been a reshape in the header.

        if len(sparse_map) > (2**31 - 1):
            _sparse_map = sparse_map.reshape((len(sparse_map)//compress_tilesize,
                                              compress_tilesize))
            _tile_shape = (1, compress_tilesize)
            s_hdr['RESHAPED'] = True
        else:
            _sparse_map = sparse_map
            _tile_shape = (compress_tilesize, )
            s_hdr['RESHAPED'] = False

    if use_rustfits:
        with rustfits.FITS(filename, mode="w+") as f:
            f.write_image(cov_index_map, extname=c_hdr["EXTNAME"], header=c_hdr)

            if compress:
                if compression == "GZIP_2":
                    comp = rustfits.Gzip2(tile_shape=_tile_shape)
                elif compression == "RICE_1":
                    comp = rustfits.Rice1(tile_shape=_tile_shape)

                f.write_image(
                    _sparse_map,
                    extname=s_hdr["EXTNAME"],
                    header=s_hdr,
                    compress=comp,
                )
            else:
                f.write(sparse_map, extname=s_hdr["EXTNAME"], header=s_hdr)

    elif use_fitsio:
        # Preferred because it is faster for integer writes.
        # Floating point writing with compression has only just
        # been fixed and I don't want to put a lower limit on
        # fitsio versioning yet.

        with fitsio.FITS(filename, mode="rw", clobber=True) as f:
            f.write(cov_index_map, extname=c_hdr["EXTNAME"], header=c_hdr)

            if compress:
                f.write(
                    _sparse_map,
                    extname=s_hdr["EXTNAME"],
                    header=s_hdr,
                    compress=compression,
                    tile_dims=_tile_shape,
                    qlevel=0.0,
                    qmethod=None,
                )
            else:
                f.write(sparse_map, extname=s_hdr["EXTNAME"], header=s_hdr)

    else:
        hdu_list = astropy_fits.HDUList()

        hdu = astropy_fits.PrimaryHDU(data=cov_index_map, header=astropy_fits.Header())
        _make_hierarch_header(c_hdr, hdu.header)
        hdu_list.append(hdu)

        if compress:
            # Try new tile_shape API (astropy>=5.3).
            hdu = astropy_fits.CompImageHDU(
                data=_sparse_map,
                header=astropy_fits.Header(),
                compression_type=compression,
                tile_shape=_tile_shape,
                quantize_level=0.0,
            )
        else:
            if sparse_map.dtype.fields is not None:
                hdu = astropy_fits.BinTableHDU(data=sparse_map, header=astropy_fits.Header())
            else:
                hdu = astropy_fits.ImageHDU(data=sparse_map, header=astropy_fits.Header())

        _make_hierarch_header(s_hdr, hdu.header)
        hdu_list.append(hdu)

        hdu_list.writeto(filename, overwrite=True)


def _make_header(metadata, force_astropy=False):
    """
    Make a fits header.

    Parameters
    ----------
    metadata : `dict`-like object
        Input metadata
    force_astropy : `bool`, optional
        Force astropy header usage; used for string processing.

    Returns
    -------
    header : `fitsio.FITSHDR` or `astropy.io.fits.Header`
    """
    if use_rustfits and not force_astropy:
        if metadata is None:
            return {}
        else:
            hdr = copy.copy(metadata)
            for reserved in FITS_RESERVED:
                hdr.pop(reserved, None)
            return hdr
    if use_fitsio and not force_astropy:
        hdr = fitsio.FITSHDR(metadata)
    else:
        hdr = astropy_fits.Header()

    if metadata is not None:
        _make_hierarch_header(metadata, hdr)

    return hdr


def _write_healpix_filename(filename, hdr, output_struct):
    """
    Write to a filename, HEALPix EXPLICIT format.

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
    if use_rustfits:
        rustfits.write(filename, output_struct, header=hdr, mode="w+")
    elif use_fitsio:
        fitsio.write(filename, output_struct, header=hdr, clobber=True)
    elif has_astropy:
        hdu_list = astropy_fits.HDUList()

        hdu = astropy_fits.BinTableHDU(data=output_struct, header=astropy_fits.Header())

        _make_hierarch_header(hdr, hdu.header, skip_reserved=False)
        hdu_list.append(hdu)

        hdu_list.writeto(filename, overwrite=True)


def _make_hierarch_header(hdr_in, hdr_out, skip_reserved=True):
    """Make a header with HIERARCH keywords to appease astropy.

    Parameters
    ----------
    hdr_in : `astropy.fits.Header` or `dict`
    hdr_out : `astropy.fits.Header`
    skip_reserved : `bool`, optional
    """
    for n in hdr_in:
        if not skip_reserved or n not in FITS_RESERVED:
            if len(n) > 8:
                hdr_out[f"HIERARCH {n}"] = hdr_in[n]
            else:
                hdr_out[n] = hdr_in[n]
