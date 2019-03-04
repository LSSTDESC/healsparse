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
            healpixMap = hp.read_map(filename, nest=True, verbose=False)
            return cls(healpixMap=healpixMap, nsideCoverage=nsideCoverage, nest=True)
        elif 'PIXTYPE' in hdr and hdr['PIXTYPE'].rstrip() == 'HEALSPARSE':
            # This is a sparse map type.  Just use fits for now.

            covIndexMap = fitsio.read(filename, ext='COV')
            sparseMap, sHdr = fitsio.read(filename, ext='SPARSE', header=True)
            return cls(covIndexMap=covIndexMap, sparseMap=sparseMap, nsideSparse=sHdr['NSIDE'])
        else:
            raise RuntimeError("Filename %s not in healpix or healsparse format." % (filename))

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
        npix = hp.nside2npix(self._nsideCoverage) 
        covMap = np.zeros(npix, dtype=np.double) 
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
        covMask = np.zeros(hp.nside2npix(self._nsideCoverage), dtype=np.bool)
        nfine = 2**self._bitShift    
        covMask[:] = (self._covIndexMap[:] + np.arange(hp.nside2npix(self._nsideCoverage))*nfine) < (len(self._sparseMap) - nfine)
        return covMask

    def generateHealpixMap(self, nside=None, reduction='mean'):
        """
        Generate the associated healpix map

        if nside is specified, then reduce
        """

        pass

