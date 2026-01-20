import os
import numpy as np
from .utils import WIDE_MASK
from .packedBoolArray import _PackedBoolArray
import warnings

use_hdf5 = False
try:
    import h5py
    use_hdf5 = True
except ImportError:
    pass

def _write_map_hdf5(hsp_map, filepath, group='map', clobber=False):
    """
    Internal method to write a HealSparseMap to an HDF5 file in a specified group.

    Supports regular, recarray, wide-mask, and bit-packed maps.

    Parameters
    ----------
    hsp_map : HealSparseMap
        Map to save
    filepath : str
        HDF5 file path
    group : str, optional
        Name of the HDF5 group to store the map (default 'map')
    clobber : bool, optional
        Overwrite the file/group if it exists (default False)
    """
    if os.path.isfile(filepath) and not clobber:
        raise RuntimeError("Filename %s exists and clobber is False." % (filepath))
    
    mode='a' #append mode so we can save to an existing file
    with h5py.File(filepath, mode) as f:
        if group in f:
            if clobber:
                del f[group]
            else:
                raise RuntimeError(f"Group '{group}' in file '{filepath}' exists. Use clobber=True to overwrite.")
        grp = f.create_group(group)

        # Coverage map - only save valid pixels
        coverage_pixels = np.where(hsp_map.coverage_mask)[0]
        coverage_values = hsp_map.coverage_mask[coverage_pixels].astype(bool)
        grp.create_dataset('coverage_pixel', data=coverage_pixels, compression='gzip')
        grp.create_dataset('coverage_value', data=coverage_values, compression='gzip')

        # Sparse map - only save valid pixels
        valid_pixels = hsp_map.valid_pixels
        grp.create_dataset('pixel', data=valid_pixels, compression='gzip')

        if hsp_map.is_rec_array:
            # for recarray, save each field separately
            sparse_values = hsp_map[valid_pixels]
            for name in sparse_values.dtype.names:
                field_grp = grp.create_group(name)
                field_grp.create_dataset('value', data=sparse_values[name], compression='gzip')
        elif hsp_map.is_bit_packed_map:
            # for bit-packed, save packed buffer
            sparse_values = hsp_map._sparse_map.data_array[
                hsp_map._cov_map.cov_index_map[valid_pixels]
            ]
            grp.create_dataset('value', data=sparse_values, compression='gzip')
        elif hsp_map.is_wide_mask_map:
            # wide mask, save 2D values
            sparse_values = hsp_map[valid_pixels]
            grp.create_dataset('value', data=sparse_values, compression='gzip')
        else:
            sparse_values = hsp_map[valid_pixels]
            grp.create_dataset('value', data=sparse_values, compression='gzip')

        # Metadata
        grp.attrs['nside_sparse'] = hsp_map._nside_sparse
        grp.attrs['sentinel'] = float(hsp_map._sentinel) if np.isscalar(hsp_map._sentinel) else str(hsp_map._sentinel)
        grp.attrs['primary'] = '' if hsp_map._primary is None else hsp_map._primary
        grp.attrs['nest'] = True  # always True

        # Map type flags
        grp.attrs['is_rec_array'] = hsp_map.is_rec_array
        grp.attrs['is_bit_packed'] = hsp_map.is_bit_packed_map
        grp.attrs['is_wide_mask'] = hsp_map.is_wide_mask_map
        grp.attrs['wide_mask_width'] = getattr(hsp_map, '_wide_mask_width', 0)

        if hsp_map.metadata is not None:
            for k, v in hsp_map.metadata.items():
                grp.attrs[k] = str(v)


def _read_map_hdf5(healsparse_class, filename, group='map', pixels=None, header=False, 
                   degrade_nside=None, weightfile=None, reduction='mean' ):
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
        Return stored metadata/header as well as map?  Default is False.
        Not implemented for hdf5
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
    with h5py.File(filename, 'r') as f:
        if group not in f:
            raise RuntimeError(f"Group '{group}' not found in file '{filename}'")
        grp = f[group]

        coverage_pixels = grp['coverage_pixel'][:]
        coverage_values = grp['coverage_value'][:]
        coverage_mask = np.zeros(coverage_pixels.max() + 1, dtype=bool)
        coverage_mask[coverage_pixels] = coverage_values

        if pixels is not None:
            pixels = np.atleast_1d(pixels)
        else:
            pixels = grp['pixel'][:]

        is_rec_array = grp.attrs.get('is_rec_array', False)
        is_bit_packed = grp.attrs.get('is_bit_packed', False)
        is_wide_mask = grp.attrs.get('is_wide_mask', False)
        wide_mask_width = grp.attrs.get('wide_mask_width', 0)

        # Allocate full sparse map with sentinel
        sentinel = grp.attrs['sentinel']
        try:
            sentinel = float(sentinel)
        except ValueError:
            if sentinel == 'UNSEEN':
                import hpgeom as hpg
                sentinel = hpg.UNSEEN
            elif sentinel == 'False':
                sentinel = False
            elif sentinel == 'True':
                sentinel = True

        if is_rec_array:
            dtype = []
            for name in grp:
                if name in ['pixel', 'coverage_pixel', 'coverage_value']:
                    continue
                dtype.append((name, grp[name]['value'].dtype))

            sparse_map = np.zeros(coverage_mask.size, dtype=dtype)
            for name, _ in dtype:
                sparse_map[name][:] = sentinel
                sparse_map[name][pixels] = grp[name]['value'][:]
        else:
            values = grp['value'][:]
            sparse_map = np.full(coverage_mask.size, sentinel, dtype=values.dtype)
            sparse_map[pixels] = values

            if is_wide_mask:
                sparse_map = sparse_map.reshape((-1, wide_mask_width)).astype(WIDE_MASK)
            elif is_bit_packed:
                sparse_map = _PackedBoolArray(data_buffer=sparse_map)

        # metadata
        metadata = {k: grp.attrs[k] for k in grp.attrs
                        if k not in ['nside_sparse', 'sentinel', 'primary',
                                    'nest', 'is_rec_array', 'is_bit_packed',
                                    'is_wide_mask', 'wide_mask_width']}

        hsp_map = healsparse_class(cov_map=coverage_mask,
                                     sparse_map=sparse_map,
                                     nside_sparse=grp.attrs['nside_sparse'],
                                     primary=grp.attrs.get('primary', None),
                                     sentinel=sentinel,
                                     metadata=metadata)
    
        if degrade_nside is not None:
            hsp_map = hsp_map.degrade(
                degrade_nside,
                reduction=reduction,
                weightfile=weightfile
            )

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
        True if it is an hdf5 file

    Raises
    ------
    Warns if hdf5 is not installed.
    """
    if not use_hdf5:
        warnings.warn("Cannot access hdf5 files without h5py",
                      UserWarning)
        return False

    return h5py.is_hdf5(filepath)