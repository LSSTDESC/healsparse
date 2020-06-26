import numpy as np
import healpy as hp

from .utils import _compute_bitshift
from .fits_shim import HealSparseFits


class HealSparseCoverage(object):
    """
    Class to define a HealSparseCoverage map
    """

    def __init__(self, cov_index_map, nside_sparse):
        """
        Instantiate a HealSparseCoverage map.

        Parameters
        ----------

        Returns
        -------
        healSparseCoverage : `HealSparseCoverage`
        """
        self._nside_coverage = hp.npix2nside(cov_index_map.size)
        self._nside_sparse = nside_sparse
        self._cov_index_map = cov_index_map
        self._bit_shift = _compute_bitshift(self._nside_coverage, self._nside_sparse)
        self._nfine_per_cov = 2**self._bit_shift
        self._compute_block_to_cov_index()

    @classmethod
    def read(cls, filename_or_fits):
        """
        Read in HealSparseCoverage.

        Parameters
        ----------

        Returns
        -------
        healSparseCoverage : `HealSparseCoverage`
           HealSparseCoverage from file
        """
        if isinstance(filename_or_fits, str):
            fits = HealSparseFits(filename_or_fits)
        else:
            fits = filename_or_fits

        try:
            cov_index_map = fits.read_ext_data('COV')
        except (OSError, KeyError):
            raise RuntimeError("File is not a HealSparseMap")

        s_hdr = fits.read_ext_header('SPARSE')

        if isinstance(filename_or_fits, str):
            fits.close()

        return cls(cov_index_map, s_hdr['NSIDE'])

    @classmethod
    def make_empty(cls, nside_coverage, nside_sparse):
        """
        Make an empty coverage map.

        Parameters
        ----------
        nside_coverage : `int`
           Healpix nside for the coverage map
        nside_sparse : `int`
           Healpix nside for the sparse map

        Returns
        -------
        healSparseCoverage : `HealSparseCoverage`
           HealSparseCoverage from file
        """
        bit_shift = _compute_bitshift(nside_coverage, nside_sparse)
        nfine_per_cov = 2**bit_shift

        cov_index_map = -1*np.arange(hp.nside2npix(nside_coverage), dtype=np.int64)*nfine_per_cov

        return cls(cov_index_map, nside_sparse)

    @classmethod
    def make_from_pixels(cls, nside_coverage, nside_sparse, cov_pixels):
        """
        Make an empty coverage map.

        Parameters
        ----------
        nside_coverage : `int`
           Healpix nside for the coverage map
        nside_sparse : `int`
           Healpix nside for the sparse map
        cov_pixels : `np.ndarray`
           Array of coverage pixels

        Returns
        -------
        healSparseCoverage : `HealSparseCoverage`
           HealSparseCoverage from file
        """
        cov_map = cls.make_empty(nside_coverage, nside_sparse)
        cov_map.initialize_pixels(cov_pixels)

        return cov_map

    def initialize_pixels(self, cov_pix):
        """
        Initialize pixels in the index map

        Parameters
        ----------
        cov_pix : `np.ndarray`
           Array of coverage pixels
        """
        self._cov_index_map[cov_pix] += np.arange(1, len(cov_pix) + 1)*self.nfine_per_cov
        self._compute_block_to_cov_index()

    def append_pixels(self, sparse_map_size, new_cov_pix, check=True, copy=True):
        """
        Append new pixels to the coverage map

        Parameters
        ----------
        sparse_map_size : `int`
           Size of current sparse map
        new_cov_pix : `np.ndarray`
           Array of new coverage pixels
        """
        if check:
            if np.max(self.cov_mask[new_cov_pix]) > 0:
                raise RuntimeError("New coverage pixels are already in the map.")

        if copy:
            new_cov_map = self.copy()
        else:
            new_cov_map = self

        # Reset to "defaults"
        cov_index_map_temp = new_cov_map._cov_index_map + np.arange(hp.nside2npix(self.nside_coverage),
                                                                    dtype=np.int64)*self.nfine_per_cov
        # set the new pixels
        cov_index_map_temp[new_cov_pix] = (np.arange(new_cov_pix.size)*self.nfine_per_cov +
                                           sparse_map_size)
        # Restore the offset
        cov_index_map_temp -= np.arange(hp.nside2npix(self.nside_coverage),
                                        dtype=np.int64)*self.nfine_per_cov

        new_cov_map._cov_index_map[:] = cov_index_map_temp
        new_cov_map._compute_block_to_cov_index()

        return new_cov_map

    def cov_pixels(self, sparse_pixels):
        """
        Get coverage pixel numbers (nest) from a set of sparse pixels.

        Parameters
        ----------
        sparse_pixels : `np.ndarray`
           Array of sparse pixels

        Returns
        -------
        cov_pixels : `np.ndarray`
           Coverage pixel numbers (nest format)
        """
        return np.right_shift(sparse_pixels, self._bit_shift)

    def cov_pixels_from_index(self, index):
        """
        Get the coverage pixels from the sparse map index.

        Parameters
        ----------
        index : `np.ndarray`
           Array of indices in sparse map

        Returns
        -------
        cov_pixels : `np.ndarray`
           Coverage pixel numbers (nest format)
        """
        return self._block_to_cov_index[(index // self.nfine_per_cov) - 1]

    @property
    def coverage_mask(self):
        """
        Get the boolean mask of the coverage map.

        Returns
        -------
        cov_mask : `np.ndarray`
           Boolean array of coverage mask.
        """
        cov_mask = (self._cov_index_map[:] +
                    np.arange(hp.nside2npix(self._nside_coverage)) *
                    self._nfine_per_cov) >= self.nfine_per_cov
        return cov_mask

    @property
    def nside_coverage(self):
        """
        Get the nside of the coverage map

        Returns
        -------
        nside_coverage : `int`
        """
        return self._nside_coverage

    @property
    def nside_sparse(self):
        """
        Get the nside of the associated sparse map

        Returns
        -------
        nside_sparse : `int`
        """
        return self._nside_sparse

    @property
    def bit_shift(self):
        """
        Get the bit_shift for the coverage map

        Returns
        -------
        bit_shift : `int`
           Number of bits to shift from coarse to fine maps
        """
        return self._bit_shift

    @property
    def nfine_per_cov(self):
        """
        Get the number of fine (sparse) pixels per coarse (coverage) pixel

        Returns
        -------
        nfine_per_cov : `int`
           Number of fine (sparse) pixels per coverage pixel
        """
        return self._nfine_per_cov

    def _compute_block_to_cov_index(self):
        """
        Compute the mapping from block number to cov_index
        """
        offset_map = (self._cov_index_map[:] +
                      np.arange(hp.nside2npix(self._nside_coverage)) *
                      self._nfine_per_cov)
        cov_mask = (offset_map >= self.nfine_per_cov)
        cov_pixels, = np.where(cov_mask)

        block_number = (offset_map[cov_pixels] // self.nfine_per_cov) - 1
        st = np.argsort(block_number)
        self._block_to_cov_index = cov_pixels[st]

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return HealSparseCoverage(self._cov_index_map.copy(), self._nside_sparse)

    # Pass through to the underlying map
    def __getitem__(self, key):
        return self._cov_index_map[key]

    def __setitem__(self, key, value):
        self._cov_index_map[key] = value
