import fsspec

use_hdf5 = False
try:
    import h5py

    use_hdf5 = True
except ImportError:
    pass


def _read_coverage_hdf5(coverage_class, filepath, hdf5_group="map"):
    """
    Internal method to read in a HealSparseCoverage map from
    an hdf5 file.

    Parameters
    ----------
    coverage_class : `type`
        Type value of the HealSparseCoverage class.
    filepath : `str`
        Name of filepath.
    hdf5_group : str
        HDF5 group containing the map

    Returns
    -------
    cov_map : `HealSparseCoverage`
        HealSparseCoverage map from file.
    """
    fs, path = fsspec.core.url_to_fs(filepath)

    with fs.open(path, mode="rb") as fsf:
        with h5py.File(fsf, "r") as f:
            if hdf5_group not in f:
                raise RuntimeError(f"Group '{hdf5_group}' not found in file '{filepath}'")
            grp = f[hdf5_group]

            cov_index_map = grp["cov_index_map"][:]
            nside_sparse = grp.attrs["nside_sparse"]

            cov_map = coverage_class(cov_index_map, nside_sparse)

    return cov_map
