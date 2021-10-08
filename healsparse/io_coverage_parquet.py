import os

use_pyarrow = False
try:
    from pyarrow import parquet
    from pyarrow import dataset
    use_pyarrow = True
except ImportError:
    pass


def _read_coverage_parquet(coverage_class, filepath, use_threads=False):
    """
    Internal method to read in a HealSparseCoverage map from
    a parquet dataset.

    Parameters
    ----------
    coverage_class : `type`
        Type value of the HealSparseCoverage class.
    filepath : `str`
        Name of filepath.
    use_threads : `bool`
        Use multithreaded reading.  This should not be necessary
        with coverage maps.

    Returns
    -------
    cov_map : `HealSparseCoverage`
        HealSparseCoverage map from file.
    """
    ds = dataset.dataset(filepath, format='parquet', partitioning='hive')
    schema = ds.schema
    # Convert from byte strings
    md = {key.decode(): schema.metadata[key].decode()
          for key in schema.metadata}

    cov_fname = os.path.join(filepath, '_coverage.parquet')
    if not os.path.isfile(cov_fname):
        # Note that this could be reconstructed from the information in the file
        # inefficiently.  This feature could be added in the future.
        raise RuntimeError("Filepath %s is missing coverage map %s" % (filepath, cov_fname))

    nside_sparse = int(md['healsparse::nside_sparse'])
    nside_coverage = int(md['healsparse::nside_coverage'])

    cov_tab = parquet.read_table(cov_fname, use_threads=use_threads, columns=['cov_pix'])
    cov_pixels = cov_tab['cov_pix'].to_numpy()

    cov_map = coverage_class.make_from_pixels(nside_coverage, nside_sparse, cov_pixels)

    return cov_map
