from __future__ import division, absolute_import, print_function
import numpy as np
import healpy as hp
import fitsio
import os
from .utils import reduce_array

class HealSparseMap(object):
    """
    Class to define a HealSparseMap
    """

    def __init__(self, covIndexMap=None, sparseMap=None, nsideSparse=None, healpixMap=None, nsideCoverage=None, primary=None, nest=True):

        if covIndexMap is not None and sparseMap is not None and nsideSparse is not None:
            # this is a sparse map input
            self._covIndexMap = covIndexMap
            self._sparseMap = sparseMap
        elif healpixMap is not None and nsideCoverage is not None:
            # this is a healpixMap input
            self._covIndexMap, self._sparseMap = self.convertHealpixMap(healpixMap,
                                                                        nsideCoverage=nsideCoverage, nest=nest)
            nsideSparse = hp.npix2nside(healpixMap.size)
        else:
            raise RuntimeError("Must specify either covIndexMap/sparseMap or healpixMap/nsideCoverage")

        self._nsideCoverage = hp.npix2nside(self._covIndexMap.size)
        self._nsideSparse = nsideSparse

        self._isRecArray = False
        self._primary = primary
        if self._sparseMap.dtype.fields is not None:
            self._isRecArray = True
            if self._primary is None:
                raise RuntimeError("Must specify `primary` field when using a recarray for the sparseMap.")

        self._bitShift = 2 * int(np.round(np.log(self._nsideSparse / self._nsideCoverage) / np.log(2)))

    @classmethod
    def read(cls, filename, nsideCoverage=None, pixels=None, header=False):
        """
        Read in a HealSparseMap.

        Parameters
        ----------
        filename: `str`
           Name of the file to read.  May be either a regular HEALPIX
           map or a HealSparseMap
        nsideCoverage: `int`, optional
           Nside of coverage map to generate if input file is healpix map.
        pixels: `list`, optional
           List of coverage map pixels to read.  Only used if input file
           is a HealSparseMap
        header: `bool`, optional
           Return the fits header as well as map?  Default is False.

        Returns
        -------
        healSparseMap: `HealSparseMap`
           HealSparseMap from file, covered by pixels
        header: `fitsio.FITSHDR` (if header=True)
           Fits header for the map file.
        """

        # Check to see if the filename is a healpix map or a sparsehealpix map

        hdr = fitsio.read_header(filename, ext=1)
        if 'PIXTYPE' in hdr and hdr['PIXTYPE'].rstrip() == 'HEALPIX':
            if nsideCoverage is None:
                raise RuntimeError("Must specify nsideCoverage when reading healpix map")

            # This is a healpix format
            # We need to determine the datatype, preserving it.
            if hdr['OBJECT'].rstrip() == 'PARTIAL':
                row = fitsio.read(filename, ext=1, rows=[0])
                dtype = row[0]['SIGNAL'].dtype.type
            else:
                row = fitsio.read(filename, ext=1, rows=[0])
                dtype = row[0][0][0].dtype.type

            healpixMap = hp.read_map(filename, nest=True, verbose=False, dtype=dtype)
            if header:
                return (cls(healpixMap=healpixMap, nsideCoverage=nsideCoverage, nest=True), hdr)
            else:
                return cls(healpixMap=healpixMap, nsideCoverage=nsideCoverage, nest=True)
        elif 'PIXTYPE' in hdr and hdr['PIXTYPE'].rstrip() == 'HEALSPARSE':
            # This is a sparse map type.  Just use fits for now.
            covIndexMap, sparseMap, nsideSparse, primary = cls._readHealSparseFile(filename, pixels=pixels)
            if header:
                return (cls(covIndexMap=covIndexMap, sparseMap=sparseMap, nsideSparse=nsideSparse, primary=primary), hdr)
            else:
                return cls(covIndexMap=covIndexMap, sparseMap=sparseMap, nsideSparse=nsideSparse, primary=primary)
        else:
            raise RuntimeError("Filename %s not in healpix or healsparse format." % (filename))

    @classmethod
    def makeEmpty(cls, nsideCoverage, nsideSparse, dtype, primary=None):
        """
        Make an empty map with nothing in it.

        Parameters
        ----------
        nsideCoverage: `int`
           Nside for the coverage map
        nsideSparse: `int`
           Nside for the sparse map
        dtype: `str` or `list` or `np.dtype`
           Datatype, any format accepted by numpy.
        primary: `str`, optional
           Primary key for recarray, required if dtype has fields.

        Returns
        -------
        healSparseMap: `HealSparseMap`
           HealSparseMap filled with UNSEEN values.
        """

        bitShift = 2 * int(np.round(np.log(nsideSparse / nsideCoverage) / np.log(2)))
        nFinePerCov = 2**bitShift

        covIndexMap = np.zeros(hp.nside2npix(nsideCoverage), dtype=np.int64)
        covIndexMap[:] -= np.arange(hp.nside2npix(nsideCoverage), dtype=np.int64) * nFinePerCov

        sparseMap = np.zeros(nFinePerCov, dtype=dtype)
        if sparseMap.dtype.fields is not None:
            if primary is None:
                raise RuntimeError("Must specify 'primary' field when using a recarray for the sparseMap.")
            # I am not sure what to do here for integer types...TBD
            for name in sparseMap.dtype.names:
                sparseMap[name][:] = hp.UNSEEN
        else:
            # fill with UNSEEN
            sparseMap[:] = hp.UNSEEN

        return cls(covIndexMap=covIndexMap, sparseMap=sparseMap, nsideSparse=nsideSparse, primary=primary)

    @staticmethod
    def _readHealSparseFile(filename, pixels=None):
        """
        Read a healsparse file, optionally with a set of coverage pixels.

        Parameters
        ----------
        filename: `str`
           Name of the file to read.
        pixels: `list`, optional
           List of integer pixels from the coverage map

        Returns
        -------
        covIndexMap: `np.array`
           Integer array for coverage index values
        sparseMap: `np.array`
           Sparse map with map dtype
        nsideSparse: `int`
           Nside of the coverage map
        primary: `str`
           Primary key field for recarray map.  Default is None.
        """
        covIndexMap = fitsio.read(filename, ext='COV')
        primary = None

        if pixels is None:
            # Read the full map
            sparseMap, sHdr = fitsio.read(filename, ext='SPARSE', header=True)
            nsideSparse = sHdr['NSIDE']
            if 'PRIMARY' in sHdr:
                primary = sHdr['PRIMARY'].rstrip()
        else:
            if len(np.unique(pixels)) < len(pixels):
                raise RuntimeError("Input list of pixels must be unique.")

            # Read part of a map
            with fitsio.FITS(filename) as fits:

                hdu = fits['SPARSE']
                sHdr = hdu.read_header()

                nsideSparse = sHdr['NSIDE']
                nsideCoverage = hp.npix2nside(covIndexMap.size)

                bitShift = 2 * int(np.round(np.log(nsideSparse / nsideCoverage) / np.log(2)))
                nFinePerCov = 2**bitShift

                imageType = False
                if hdu.get_exttype() == 'IMAGE_HDU':
                    # This is an image extension
                    sparseMapSize = hdu.get_dims()[0]
                    imageType = True
                else:
                    # This is a table extension
                    primary = sHdr['PRIMARY'].rstrip()
                    sparseMapSize = hdu.get_nrows()

                nCovPix = sparseMapSize // nFinePerCov - 1

                # This is the map without the offset
                covIndexMapTemp = covIndexMap + np.arange(hp.nside2npix(nsideCoverage), dtype=np.int64) * nFinePerCov
                covPix, = np.where(covIndexMapTemp >= nFinePerCov)

                # Find which pixels are in the coverage map
                sub = np.clip(np.searchsorted(covPix, pixels), 0, covPix.size - 1)
                ok, = np.where(covPix[sub] == pixels)
                if ok.size == 0:
                    raise RuntimeError("None of the specified pixels are in the coverage map")
                sub = np.sort(sub[ok])

                # It is not 100% sure this is the most efficient way to read in using
                # fitsio, but it does work.
                sparseMap = np.zeros((sub.size + 1) * nFinePerCov, dtype=fits['SPARSE'][0:1].dtype)
                # Read in the overflow bin
                sparseMap[0: nFinePerCov] = hdu[0: nFinePerCov]
                # And read in the pixels
                for i, pix in enumerate(covPix[sub]):
                    sparseMap[(i + 1)*nFinePerCov: (i + 2)*nFinePerCov] = hdu[covIndexMapTemp[pix]: covIndexMapTemp[pix] + nFinePerCov]

                # Set the coverage index map for the pixels that we read in
                covIndexMap[:] = 0
                covIndexMap[covPix[sub]] = np.arange(1, sub.size + 1) * nFinePerCov
                covIndexMap[:] -= np.arange(hp.nside2npix(nsideCoverage), dtype=np.int64) * nFinePerCov

        return covIndexMap, sparseMap, nsideSparse, primary

    @staticmethod
    def convertHealpixMap(healpixMap, nsideCoverage, nest=True):
        """
        Convert a healpix map to a healsparsemap.

        Parameters
        ----------
        healpixMap: `np.array`
           Numpy array that describes a healpix map.
        nsideCoverage: `int`
           Nside for the coverage map to construct
        nest: `bool`, optional
           Is the input map in nest format?  Default is True.

        Returns
        -------
        covIndexMap: `np.array`
           Coverage map with pixel indices
        sparseMap: `np.array`
           Sparse map of input values.
        """
        if not nest:
            # must convert map to ring format
            healpixMap = hp.reorder(healpixMap, r2n=True)

        # Compute the coverage map...
        ipnest, = np.where(healpixMap > hp.UNSEEN)

        bitShift = 2 * int(np.round(np.log(hp.npix2nside(healpixMap.size) / nsideCoverage) / np.log(2)))
        ipnestCov = np.right_shift(ipnest, bitShift)

        covPix = np.unique(ipnestCov)

        nFinePerCov = int(healpixMap.size / hp.nside2npix(nsideCoverage))

        # This initializes as zeros, that's the location of the overflow bins
        covIndexMap = np.zeros(hp.nside2npix(nsideCoverage), dtype=np.int64)

        # The default for the covered pixels is the location in the array (below)
        # Note that we have a 1-index here to have the 0-index overflow bin
        covIndexMap[covPix] = np.arange(1, covPix.size + 1) * nFinePerCov
        # And then subtract off the starting fine pixel for each coarse pixel
        covIndexMap[:] -= np.arange(hp.nside2npix(nsideCoverage), dtype=np.int64) * nFinePerCov

        sparseMap = np.zeros((covPix.size + 1) * nFinePerCov, dtype=healpixMap.dtype) + hp.UNSEEN
        sparseMap[ipnest + covIndexMap[ipnestCov]] = healpixMap[ipnest]

        return covIndexMap, sparseMap

    def write(self, filename, clobber=False, header=None):
        """
        Write heal HealSparseMap to filename

        Parameters
        ----------
        filename: `str`
           Name of file to save
        clobber: `bool`, optional
           Clobber existing file?  Default is False.
        header: `fitsio.FITSHDR` or `dict`, optional
           Header to put in first extension with additional metadata.
           Default is None.
        """
        if os.path.isfile(filename) and not clobber:
            raise RuntimeError("Filename %s exists and clobber is False." % (filename))

        # Note that we put the requested header information in each of the extensions.
        cHdr = fitsio.FITSHDR(header)
        cHdr['PIXTYPE'] = 'HEALSPARSE'
        cHdr['NSIDE'] = self._nsideCoverage
        fitsio.write(filename, self._covIndexMap, header=cHdr, extname='COV', clobber=True)
        sHdr = fitsio.FITSHDR(header)
        sHdr['PIXTYPE'] = 'HEALSPARSE'
        sHdr['NSIDE'] = self._nsideSparse
        if self._isRecArray:
            sHdr['PRIMARY'] = self._primary
        fitsio.write(filename, self._sparseMap, header=sHdr, extname='SPARSE')

    def updateValues(self, pixel, values, nest=True):
        """
        Update the values in the sparsemap.

        Parameters
        ----------
        pixel: `np.array`
           Integer array of sparseMap pixel values
        values: `np.array`
           Array of values.  Must be same type as sparseMap
        """

        # First, check if these are the same type
        if not isinstance(values, np.ndarray):
            raise RuntimeError("Values are not a numpy ndarray")

        if not nest:
            _pix = hp.ring2nest(pixel)
        else:
            _pix = pixel

        if self._sparseMap.dtype != values.dtype:
            raise RuntimeError("Data-type mismatch between sparseMap and values")

        # Compute the coverage pixels
        ipnestCov = np.right_shift(_pix, self._bitShift)

        # Check which pixels are in the coverage map
        covMask = self.coverageMask
        inCov, = np.where(covMask[ipnestCov])
        outCov, = np.where(~covMask[ipnestCov])

        # Replace values for those pixels in the coverage map
        self._sparseMap[_pix[inCov] + self._covIndexMap[ipnestCov[inCov]]] = values[inCov]

        # Update the coverage map for the rest of the pixels (if necessary)
        if outCov.size > 0:
            # This requires data copying. (Even numpy appending does)
            # I don't want to overthink this and prematurely optimize, but
            # I want it to be able to work when the map isn't being held
            # in memory.  So that will require an append and non-contiguous
            # pixels, which I *think* should be fine.

            nFinePerCov = 2**self._bitShift

            newCovPix = np.unique(ipnestCov[outCov])
            sparseAppend = np.zeros(newCovPix.size * nFinePerCov, dtype=self._sparseMap.dtype)
            # Fill with the empty defaults (generally UNSEEN)
            sparseAppend[:] = self._sparseMap[0]

            # Update covIndexMap
            # These are pixels that are at the end of the previous sparsemap

            # First reset the map to the base pixel indices
            covIndexMapTemp = self._covIndexMap + np.arange(hp.nside2npix(self._nsideCoverage), dtype=np.int64) * nFinePerCov
            # Put in the appended pixels
            covIndexMapTemp[newCovPix] = np.arange(newCovPix.size) * nFinePerCov + self._sparseMap.size
            # And put the offset back in
            covIndexMapTemp[:] -= np.arange(hp.nside2npix(self._nsideCoverage), dtype=np.int64) * nFinePerCov

            # Fill in the pixels to append
            sparseAppend[_pix[outCov] + covIndexMapTemp[ipnestCov[outCov]] - self._sparseMap.size] = values[outCov]

            # And set the values in the map
            self._covIndexMap = covIndexMapTemp
            self._sparseMap = np.append(self._sparseMap, sparseAppend)

    def getValueRaDec(self, ra, dec):
        """
        Get the map value for a ra/dec in degrees (for now)

        Parameters
        ----------
        ra: `np.array`
           Float array of RA (degrees)
        dec: `np.array`
           Float array of dec (degrees)

        Returns
        -------
        values: `np.array`
           Array of values from the map.
        """

        return self.getValueThetaPhi(np.radians(90.0 - dec), np.radians(ra))

    def getValueThetaPhi(self, theta, phi):
        """
        Get the map value for a theta/phi.

        Parameters
        ----------
        theta: `np.array`
           Float array of healpix theta (np.radians(90.0 - dec))
        phi: `np.array`
           Float array of healpix phi (np.radians(ra))

        Returns
        -------
        values: `np.array`
           Array of values from the map.
        """

        ipnest = hp.ang2pix(self._nsideSparse, theta, phi, nest=True)

        return self.getValuePixel(ipnest, nest=True)

    def getValuePixel(self, pixel, nest=True):
        """
        Get the map value for a pixel.

        Parameters
        ----------
        pixel: `np.array`
           Integer array of healpix pixels.
        nest: `bool`, optional
           Are the pixels in nest scheme?  Default is True.

        Returns
        -------
        values: `np.array`
           Array of values from the map.

        """

        if not nest:
            _pix = hp.ring2nest(self._nsideSparse, pixel)
        else:
            _pix = pixel

        ipnestCov = np.right_shift(_pix, self._bitShift)

        return self._sparseMap[_pix + self._covIndexMap[ipnestCov]]

    @property
    def coverageMap(self):
        """
        Get the fractional area covered by the sparse map
        in the resolution of the coverage map

        Returns
        -------
        covMap: `np.array`
           Float array of fractional coverage of each pixel
        """

        covMap = np.zeros_like(self.coverageMask, dtype=np.double)
        covMask = self.coverageMask
        npop_pix = np.count_nonzero(covMask)
        if self._isRecArray:
            spMap_T = self._sparseMap[self._primary].reshape((npop_pix+1, -1))
        else:
            spMap_T = self._sparseMap.reshape((npop_pix+1, -1))
        counts = np.sum((spMap_T > hp.UNSEEN), axis=1).astype(np.double)
        covMap[covMask] = counts[1:] / 2**self._bitShift
        return covMap

    @property
    def coverageMask(self):
        """
        Get the boolean mask of the coverage map.

        Returns
        -------
        covMask: `np.array`
           Boolean array of coverage mask.
        """

        nfine = 2**self._bitShift
        covMask = (self._covIndexMap[:] + np.arange(hp.nside2npix(self._nsideCoverage))*nfine) >= nfine
        return covMask

    @property
    def nsideCoverage(self):
        """
        Get the nside of the coverage map

        Returns
        -------
        nsideCoverage: `int`
        """

        return self._nsideCoverage

    @property
    def nsideSparse(self):
        """
        Get the nside of the sparse map

        Returns
        -------
        nsideSparse: `int`
        """

        return self._nsideSparse

    def generateHealpixMap(self, nside=None, reduction='mean', key=None):
        """
        Generate the associated healpix map

        if nside is specified, then reduce

        Args:
        -----
        nside: `int`
            Output nside resolution parameter (should be a multiple of 2). If not specified
            the output resolution will be equal to the parent's sparsemap nsideSparse
        reduction: `str`
            If a change in resolution is requested, this controls the method to reduce the
            map computing the mean, median, std, max or min of the neighboring pixels to compute
            the `degraded` map.
        key: `str`
            If the parent HealSparseMap contains `recarrays`, key selects the field that will be
            transformed into a HEALPix map.

        Returns:
        --------
        hp_map: `ndarray`
            Output HEALPix map with the requested resolution.
        """

        # If no nside is passed, we generate a map with the same resolution as the original
        if nside is None:
            nside = self._nsideSparse

        # Check if we are dealing with recArrays and try to preserve the data types
        if self._isRecArray:
            if key is None:
                raise ValueError('key should be specified for HealSparseMaps including `recarray`')
            else:
                if isinstance(self._sparseMap[key],int):
                    aux_dtype = np.float #Force to float
                else:
                    aux_dtype = self._sparseMap[key].dtype
                aux = self._sparseMap[key].astype(aux_dtype)
                # We create one HealSparseMap in order to avoid keys other than the specified (in the case of having more than one)
                aux_spmap = HealSparseMap(covIndexMap=self._covIndexMap, sparseMap=aux, nsideSparse=self._nsideSparse,
                                          nsideCoverage=self._nsideCoverage)
        # If it is not a recArray then, try to preserve the type anyway
        else:
            if isinstance(self._sparseMap,int):
                aux_dtype = np.float #Force to float
                aux = self._sparseMap.astype(aux_dtype)
                # `self.degrade` would have taken care of this but, in case that we preserve the resolution...
                aux_spmap = HealSparseMap(covIndexMap=self._covIndexMap, sparseMap=aux, nsideSparse=self._nsideSparse,
                                          nsideCoverage=self._nsideCoverage)
            else:
                aux_spmap = self
                aux_dtype = self._sparseMap.dtype
        # Create empty HEALPix map
        hp_map = np.zeros(hp.nside2npix(nside), dtype=aux_dtype) + hp.UNSEEN # remember that empty means UNSEEN!
        # We generate a HealSparseMap with the requested size before converting to healpix
        if nside < self._nsideSparse:
            aux_spmap = aux_spmap.degrade(nside, reduction=reduction) # We change the resolution of the map
        if nside > self._nsideSparse:
            raise ValueError('The generation of HEALPix maps with higher resolution than the original \
                             sparse map is not yet implemented')
        aux = aux_spmap._sparseMap
        pop_idx, = np.where(aux > hp.UNSEEN)
        cov_idx = np.right_shift(pop_idx, aux_spmap._bitShift)
        hp_map[pop_idx - aux_spmap._covIndexMap[aux_spmap.coverageMask][cov_idx-1]]=aux[pop_idx]
        return hp_map

    @property
    def validPixels(self):
        """
        Get an array of valid pixels in the sparse map.

        Returns
        -------
        validPixels: `np.array`
        """

        if self._isRecArray:
            validSparsePixels, = np.where(self._sparseMap[self._primary] > hp.UNSEEN)
        else:
            validSparsePixels, = np.where(self._sparseMap > hp.UNSEEN)

        coverageIndex = np.right_shift(validSparsePixels, self._bitShift)

        validPixels = validSparsePixels - self._covIndexMap[self.coverageMask][coverageIndex - 1]

        return validPixels

    def degrade(self, nside_out, reduction='mean'):
        """
        Reduce the resolution, i.e., increase the pixel size
        of a given sparse map

        Args:
        ----
        nside_out: `int`, output Nside resolution parameter.
        reduction: `str`, reduction method (mean, median, std, max, min).
        """

        if self._nsideSparse < nside_out:
            raise ValueError('nside_out should be smaller than nside for the sparseMap')
        # Count the number of filled pixels in the coverage mask
        npop_pix = np.count_nonzero(self.coverageMask)
        # We need the new bitShifts and we have to build a new CovIndexMap
        bitShift = 2 * int(np.round(np.log(nside_out / self._nsideCoverage) / np.log(2)))
        nFinePerCov = 2**bitShift
        # Work with RecArray (we have to change the resolution to all maps...)
        if self._isRecArray:
            dtype = []
            # We should avoid integers
            for key, value in self._sparseMap.dtype.fields.items():
                if isinstance(self._sparseMap[key],int):
                    dtype.append((key,np.float))
                else:
                    dtype.append((key,value[0]))
            dtype = self._sparseMap.dtype
            # Allocate new map
            newsparseMap = np.zeros((npop_pix+1)*nFinePerCov, dtype=dtype)
            for key, value in newsparseMap.dtype.fields.items():
                aux = self._sparseMap[key].astype(float)
                aux = aux.reshape((npop_pix+1, (nside_out//self._nsideCoverage)**2, -1))
                aux[aux==hp.UNSEEN] = np.nan
                # Perform the reduction operation (check utils.reduce_array)
                aux = reduce_array(aux, reduction=reduction)
                # Transform back to UNSEEN
                aux[np.isnan(aux)] = hp.UNSEEN
                newsparseMap[key] = aux

        # Work with ndarray
        else:
            aux = self._sparseMap
            aux = aux.reshape((npop_pix+1, (nside_out//self._nsideCoverage)**2, -1))
            aux[aux==hp.UNSEEN] = np.nan
            aux = reduce_array(aux, reduction=reduction)
            # NaN are converted to UNSEEN
            aux[np.isnan(aux)] = hp.UNSEEN
            newsparseMap = aux
        # The coverage index map is now offset, we have to build a new one
        newIndexMap = np.zeros(hp.nside2npix(self._nsideCoverage), dtype=np.int64)
        newIndexMap[self.coverageMask] = np.arange(1, npop_pix + 1) * nFinePerCov
        newIndexMap[:] -= np.arange(hp.nside2npix(self._nsideCoverage), dtype=np.int64) * nFinePerCov
        return HealSparseMap(covIndexMap=newIndexMap, sparseMap=newsparseMap, nsideCoverage=self._nsideCoverage,
                             nsideSparse=nside_out, primary=self._primary)
