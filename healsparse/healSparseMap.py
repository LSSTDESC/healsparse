from __future__ import division, absolute_import, print_function

import numpy as np
import healpy as hp
import fitsio
import os

class HealSparseMap(object):
    """
    Class to define a HealSparseMap
    """

    def __init__(self, covIndexMap=None, sparseMap=None, nsideSparse=None, healpixMap=None, nsideCoverage=None, nest=True):

        if covIndexMap is not None and sparseMap is not None and nsideSparse is not None:
            # this is a sparse map input
            self._covIndexMap = covIndexMap
            self._sparseMap = sparseMap
        elif healpixMap is not None and nsideCoverage is not None:
            # this is a healpxMap input
            self._covIndexMap, self._sparseMap = self.convertHealpixMap(healpixMap,
                                                                   nsideCoverage=nsideCoverage, nest=nest)
            nsideSparse = hp.npix2nside(healpixMap.size)
        else:
            raise RuntimeError("Must specify either covIndexMap/sparseMap or healpixMap/nsideCoverage")

        self._nsideCoverage = hp.npix2nside(self._covIndexMap.size)
        self._nsideSparse = nsideSparse

        self._bitShift = 2 * int(np.round(np.log(self._nsideSparse / self._nsideCoverage) / np.log(2)))

    @classmethod
    def read(cls, filename, nsideCoverage=None, pixels=None):
        """
        Read in a HealSparseMap
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
                dtype = row[0]['T'][0].dtype.type

            healpixMap = hp.read_map(filename, nest=True, verbose=False, dtype=dtype)
            return cls(healpixMap=healpixMap, nsideCoverage=nsideCoverage, nest=True)
        elif 'PIXTYPE' in hdr and hdr['PIXTYPE'].rstrip() == 'HEALSPARSE':
            # This is a sparse map type.  Just use fits for now.
            covIndexMap, sparseMap, nsideSparse = cls._readHealSparseFile(filename, pixels=pixels)
            return cls(covIndexMap=covIndexMap, sparseMap=sparseMap, nsideSparse=nsideSparse)
        else:
            raise RuntimeError("Filename %s not in healpix or healsparse format." % (filename))

    @staticmethod
    def _readHealSparseFile(filename, pixels=None):
        """
        Read a healsparse file, optionally with a set of coverage pixels
        """
        covIndexMap = fitsio.read(filename, ext='COV')

        if pixels is None:
            # Read the full map
            sparseMap, sHdr = fitsio.read(filename, ext='SPARSE', header=True)
            nsideSparse = sHdr['NSIDE']
        else:
            if len(np.unique(pixels)) < len(pixels):
                raise RuntimeError("Input list of pixels must be unique.")

            # Read a partial map
            #fits = fitsio.FITS(filename)
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
                    # Currently, this is the only supported type.
                    sparseMapSize = hdu.get_dims()[0]
                    imageType = True
                else:
                    # This is a table extension
                    raise RuntimeError("Table extension not yet supported.")
                    sparseMapSize = hdu.get_nrows()

                nCovPix = sparseMapSize / nFinePerCov - 1
                covPix, = np.where((sparseMapSize - 2**bitShift) !=
                                   (np.arange(hp.nside2npix(nsideCoverage), dtype=np.int64) * 2**bitShift) + covIndexMap)

                # Find which pixels are in the coverage map
                sub = np.clip(np.searchsorted(covPix, pixels), 0, covPix.size - 1)
                ok, = np.where(covPix[sub] == pixels)
                if ok.size == 0:
                    raise RuntimeError("None of the specified pixels are in the coverage map")
                sub = np.sort(sub[ok])

                if imageType:
                    sparseMap = np.zeros((sub.size + 1) * nFinePerCov, dtype=fits['SPARSE'][0:1].dtype) + hp.UNSEEN
                    for i, p in enumerate(sub):
                        sparseMap[i*nFinePerCov: (i + 1)*nFinePerCov] = hdu[p*nFinePerCov: (p + 1)*nFinePerCov]

                else:
                    # This is all preliminary work, table reading is not yet implemented.

                    # This indexing selects out just the rows that we want to grab
                    rows = np.tile(np.arange(nFinePerCov), b.size) + np.repeat(b, nFinePerCov) * nFinePerCov

                    # This will have to be updated when supporting table types
                    sparseMap = np.zeros((b.size + 1) * nFinePerCov, dtype=fits['SPARSE'][0:1].dtype)
                    sparseMap[0: rows.size] = hdu.read_rows(rows)

                # Set the coverage index map for the pixels that we read in
                covIndexMap[:] = sub.size * nFinePerCov
                covIndexMap[covPix[sub]] = np.arange(sub.size) * nFinePerCov
                covIndexMap[:] -= np.arange(hp.nside2npix(nsideCoverage), dtype=np.int64) * nFinePerCov

        return covIndexMap, sparseMap, nsideSparse


    @staticmethod
    def convertHealpixMap(healpixMap, nsideCoverage, nest=True):
        """
        Convert a healpix map

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

        covIndexMap = np.zeros(hp.nside2npix(nsideCoverage), dtype=np.int64)
        # This points to the overflow bins
        covIndexMap[:] = covPix.size * nFinePerCov

        # The default for the covered pixels is the location in the array (below)
        covIndexMap[covPix] = np.arange(covPix.size) * nFinePerCov
        # And then subtract off the starting fine pixel for each coarse pixel
        covIndexMap[:] -= np.arange(hp.nside2npix(nsideCoverage), dtype=np.int64) * nFinePerCov

        sparseMap = np.zeros((covPix.size + 1) * nFinePerCov, dtype=healpixMap.dtype) + hp.UNSEEN

        sparseMap[ipnest + covIndexMap[ipnestCov]] = healpixMap[ipnest]

        return covIndexMap, sparseMap

    def write(self, filename, clobber=False):
        """
        Write heal HealSparseMap to filename
        """
        if os.path.isfile(filename) and not clobber:
            raise RuntimeError("Filename %s exists and clobber is False." % (filename))

        cHdr = fitsio.FITSHDR()
        cHdr['PIXTYPE'] = 'HEALSPARSE'
        cHdr['NSIDE'] = self._nsideCoverage
        fitsio.write(filename, self._covIndexMap, header=cHdr, extname='COV', clobber=True)
        sHdr = fitsio.FITSHDR()
        sHdr['PIXTYPE'] = 'HEALSPARSE'
        sHdr['NSIDE'] = self._nsideSparse
        fitsio.write(filename, self._sparseMap, header=sHdr, extname='SPARSE')

    def getValueRaDec(self, ra, dec):
        """
        Get the map value for a ra/dec in degrees (for now)
        """

        return self.getValueThetaPhi(np.radians(90.0 - dec), np.radians(ra))

    def getValueThetaPhi(self, theta, phi):
        """
        Get the map value for a theta/phi
        """

        ipnest = hp.ang2pix(self._nsideSparse, theta, phi, nest=True)

        return self.getValuePixel(ipnest, nest=True)

    def getValuePixel(self, pixel, nest=True):
        """
        Get the map value for a pixel
        """

        if not nest:
            _pix = hp.ring2nest(pixel)
        else:
            _pix = pixel

        ipnestCov = np.right_shift(_pix, self._bitShift)

        return self._sparseMap[_pix + self._covIndexMap[ipnestCov]]

    @property
    def coverageMap(self):
        """
        Get the fractional area covered by the sparse map
        in the resolution of the coverage map
        """

        covMap = np.zeros_like(self.coverageMask, dtype=np.double)
        covMask = self.coverageMask
        npop_pix = np.count_nonzero(covMask)
        spMap_T = self._sparseMap.reshape((npop_pix+1, -1))
        counts = np.sum((spMap_T > hp.UNSEEN), axis=1).astype(np.double)
        covMap[covMask] = counts[:-1] / 2**self._bitShift
        return covMap

    @property
    def coverageMask(self):
        """
        Get the boolean mask
        """

        nfine = 2**self._bitShift
        covMask = (self._covIndexMap[:] + np.arange(hp.nside2npix(self._nsideCoverage))*nfine) < (len(self._sparseMap) - nfine)
        return covMask

    def generateHealpixMap(self, nside=None, reduction='mean'):
        """
        Generate the associated healpix map

        if nside is specified, then reduce
        """

        pass

