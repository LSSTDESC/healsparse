import os
import numpy as np
from .utils import WIDE_MASK
from .packedBoolArray import _PackedBoolArray
import warnings
from .healSparseCoverage import HealSparseCoverage
import astropy.io.fits as fits
import hpgeom as hpg

use_hdf5 = False
try:
    import h5py

    use_hdf5 = True
except ImportError:
    pass


def _write_map_hdf5(hsp_map, filepath, hdf5_group="map", clobber=False):
    """
    Internal method to write a HealSparseMap to an HDF5 file in a specified group.

    Supports regular, recarray, wide-mask, and bit-packed maps.

    Parameters
    ----------
    hsp_map : HealSparseMap
        Map to save.
    filepath : str
        HDF5 file path.
    hdf5_group : str, optional
        Name of the HDF5 group to store the map.
    clobber : bool, optional
        Overwrite the file/group if it exists.
    """
    if os.path.isfile(filepath) and not clobber:
        with h5py.File(filepath, 'r') as f:
            group_exists = hdf5_group in f
        if group_exists:
            raise RuntimeError(f"Filename {filepath} with group {hdf5_group} exists and clobber is False.")

    mode = "a"  # append mode, if file doesn't exist it will be made
    with h5py.File(filepath, mode) as f:
        if hdf5_group in f and clobber:
            del f[hdf5_group]
        grp = f.create_group(hdf5_group)

        # Coverage map - save coverage index map
        grp.create_dataset("cov_index_map", data=hsp_map._cov_map[:], compression="gzip")

        # Sparse map - save the _sparse_map (occupied coverage pixels only+overflow)
        # re-shape sparse_map data so each coverage pixel is a different row
        # chunk the dataset such that each chunk is 1 coverage pixel
        ncov_in_sparse = sum(hsp_map.coverage_mask) + 1  # include buffer pixel
        nfine_per_cov = hsp_map._cov_map._nfine_per_cov

        if hsp_map.is_rec_array:
            # for recarray, save each field separately
            for name in hsp_map._sparse_map.dtype.names:
                sparse_map_reshape = hsp_map[name]._sparse_map.reshape(ncov_in_sparse, nfine_per_cov)
                field_grp = grp.create_group(name)
                field_grp.create_dataset(
                    "sparse_map",
                    data=sparse_map_reshape,
                    chunks=(1, nfine_per_cov),
                    compression="gzip",
                )
        elif hsp_map.is_bit_packed_map:
            # for bit packed maps, save packed array
            sparse_map_reshape = hsp_map._sparse_map.data_array.reshape(ncov_in_sparse, nfine_per_cov//8)
            grp.create_dataset(
                "sparse_map",
                data=sparse_map_reshape,
                chunks=(1, nfine_per_cov//8),
                compression="gzip",
                dtype=np.uint8,
            )

        elif hsp_map.is_wide_mask_map:
            # wide mask, save 2D values
            sparse_map_reshape = hsp_map[name]._sparse_map.reshape(
                ncov_in_sparse, nfine_per_cov, hsp_map.wide_mask_width
            )
            grp.create_dataset(
                "sparse_map",
                data=hsp_map._sparse_map,
                chunks=(1, nfine_per_cov, hsp_map.wide_mask_width),
                compression="gzip",
            )
        else:
            # "regular" map
            sparse_map_reshape = hsp_map._sparse_map.reshape(ncov_in_sparse, nfine_per_cov)
            grp.create_dataset(
                "sparse_map",
                data=sparse_map_reshape,
                chunks=(1, nfine_per_cov),
                compression="gzip",
            )

        # Metadata
        grp.attrs["nside_sparse"] = hsp_map.nside_sparse
        grp.attrs["nside_coverage"] = hsp_map.nside_coverage
        grp.attrs["sentinel"] = hsp_map._sentinel
        grp.attrs["primary"] = "" if hsp_map._primary is None else hsp_map._primary
        grp.attrs["nest"] = True  # always True

        # Map type flags
        grp.attrs["is_rec_array"] = hsp_map.is_rec_array
        grp.attrs["is_bit_packed"] = hsp_map.is_bit_packed_map
        grp.attrs["is_wide_mask"] = hsp_map.is_wide_mask_map
        grp.attrs["wide_mask_width"] = getattr(hsp_map, "_wide_mask_width", 0)

        if hsp_map.metadata is not None:
            for k, v in hsp_map.metadata.items():
                grp.attrs[k] = v


def _read_map_hdf5(
    healsparse_class,
    filename,
    hdf5_group="map",
    pixels=None,
    header=False,
    degrade_nside=None,
    weightfile=None,
    reduction="mean",
):
    """
    Internal method to read a HealSparseMap from an HDF5 file in a specified group.

    Parameters
    ----------
    healsparse_class : class
        HealSparseMap class
    filename : str
        HDF5 file path
    group : str
        HDF5 group containing the map
    pixels : `list`, optional
        List of coverage map pixels to read.
    header : `bool`, optional
        Return stored metadata/header as well as map?
    degrade_nside : `int`, optional
        Degrade map to this nside on read.
    weightfile : `str`, optional
        Weight map for weighted degrade.
    reduction : `str`, optional
        Reduction method for degrade-on-read.

    Returns
    -------
    HealSparseMap instance
    """
    with h5py.File(filename, "r") as f:
        if hdf5_group not in f:
            raise RuntimeError(f"Group '{hdf5_group}' not found in file '{filename}'")
        grp = f[hdf5_group]

        cov_index_map = grp["cov_index_map"][:]
        nside_sparse = grp.attrs["nside_sparse"]
        nside_coverage = grp.attrs["nside_coverage"]

        # this is the coverage map of the *full* map
        cov_map = HealSparseCoverage(cov_index_map, nside_sparse)

        ncov_in_sparse = sum(cov_map.coverage_mask) + 1  # including overflow pixel
        nfine_per_cov = cov_map._nfine_per_cov

        is_rec_array = grp.attrs.get("is_rec_array", False)
        is_bit_packed = grp.attrs.get("is_bit_packed", False)
        is_wide_mask = grp.attrs.get("is_wide_mask", False)
        wide_mask_width = grp.attrs.get("wide_mask_width", 0)

        # sentinel handling
        sentinel = grp.attrs["sentinel"]

        if pixels is not None:
            # check the requested pixels are ok
            _pixels = np.atleast_1d(pixels)
            if len(np.unique(_pixels)) < len(_pixels):
                raise RuntimeError("Input list of pixels must be unique.")

            # Which pixels are in the coverage map?
            (cov_pix,) = np.where(cov_map.coverage_mask)
            sub = np.clip(np.searchsorted(cov_pix, _pixels), 0, cov_pix.size - 1)
            (ok,) = np.where(cov_pix[sub] == _pixels)
            if ok.size == 0:
                raise RuntimeError("None of the specified pixels are in the coverage map.")
            _pixels = np.sort(_pixels[ok])

            # translate the _pixel index to the row in the hdf5 file
            cov_index_map_temp = (
                cov_map[:] +
                np.arange(hpg.nside_to_npixel(nside_coverage), dtype=np.int64) * cov_map.nfine_per_cov
            )
            cov_index_in_sparse = np.append(
                0, cov_index_map_temp[_pixels] // cov_map.nfine_per_cov
            )  # pixel 0 is the overflow pixel

            # hdf5 has to read rows in order
            order = np.argsort(cov_index_in_sparse)
            cov_index_in_sparse_ordered = cov_index_in_sparse[order]
            inv = np.empty_like(order)
            inv[order] = np.arange(order.size)

            # make sub coverage map
            _cov_map = HealSparseCoverage.make_from_pixels(
                nside_coverage,
                nside_sparse,
                _pixels,
            )

            # how many cov pixels(+overflow) are in the sub map
            ncov_in_sparse_sub = len(_pixels) + 1

            sparse_size = ncov_in_sparse_sub * nfine_per_cov
        else:
            cov_index_in_sparse_ordered = slice(None)
            inv = slice(None)
            sparse_size = ncov_in_sparse * nfine_per_cov
            _cov_map = cov_map

        # load the data
        if is_rec_array:
            dtype = []
            for name in grp:
                if name in ["cov_index_map"]:
                    continue
                dtype.append((name, grp[name]["sparse_map"].dtype))

            sparse_map = np.zeros(sparse_size, dtype=dtype)
            for name, _ in dtype:
                sparse_map[name] = grp[name]["sparse_map"][cov_index_in_sparse_ordered, :][inv].reshape(-1)
        elif is_wide_mask:
            sparse_map = (
                grp["sparse_map"][cov_index_in_sparse_ordered, :][inv]
                .reshape((-1, wide_mask_width))
                .astype(WIDE_MASK)
            )
        elif is_bit_packed:
            bit_packed_map = grp["sparse_map"][cov_index_in_sparse_ordered, :][inv].reshape(-1)
            sparse_map = _PackedBoolArray(data_buffer=bit_packed_map)
            sentinel = bool(sentinel)  # has to be python bool, not numpy bool
        else:
            # is regular map
            sparse_map = grp["sparse_map"][cov_index_in_sparse_ordered, :][inv].reshape(-1)

        # metadata
        metadata = {
            k: grp.attrs[k]
            for k in grp.attrs
            if k
            not in [
                "nside_sparse",
                "nside_coverage",
                "sentinel",
                "primary",
                "nest",
                "is_rec_array",
                "is_bit_packed",
                "is_wide_mask",
                "wide_mask_width",
            ]
        }

        hsp_map = healsparse_class(
            cov_map=_cov_map,
            sparse_map=sparse_map,
            nside_sparse=grp.attrs["nside_sparse"],
            primary=grp.attrs.get("primary", None),
            sentinel=sentinel,
            metadata=metadata,
        )

        if degrade_nside is not None:
            hsp_map = hsp_map.degrade(degrade_nside, reduction=reduction, weightfile=weightfile)

        if header:
            hdr = fits.Header(hsp_map.metadata)
            return (hsp_map, hdr)
        else:
            return hsp_map


def check_hdf5_file(filepath):
    """
    Check if a filepath points to an hdf5 file

    Parameters
    ----------
    filepath : `str`
        File path to check.

    Returns
    -------
    is_hdf5_file : `bool`
        True if it is an hdf5 file.

    Raises
    ------
    Warns if hdf5 is not installed.
    """
    if not use_hdf5:
        warnings.warn("Cannot access hdf5 files without h5py", UserWarning)
        return False

    return h5py.is_hdf5(filepath)
