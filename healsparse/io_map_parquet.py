import os
import numpy as np
import hpgeom as hpg
import astropy.io.fits as fits

from .utils import is_integer_value, _compute_bitshift, WIDE_MASK
from .utils import check_sentinel
from .healSparseCoverage import HealSparseCoverage
from .fits_shim import _make_header
from .packedBoolArray import _PackedBoolArray

use_pyarrow = False
try:
    import pyarrow as pa
    from pyarrow import parquet
    from pyarrow import dataset
    from .parquet_shim import to_numpy_dtype
    use_pyarrow = True
except ImportError:
    pass


def _read_map_parquet(healsparse_class, filepath, pixels=None, header=False,
                      degrade_nside=None, weightfile=None, reduction='mean',
                      use_threads=False):
    """
    Internal function to read in a HealSparseMap from a parquet dataset.

    Parameters
    ----------
    healsparse_class : `type`
        Type value of the HealSparseMap class.
    filepath : `str`
        Name of the file path to read.  Must be a parquet dataset.
    pixels : `list`, optional
        List of coverage map pixels to read.
    header : `bool`, optional
        Return the parquet metadata as well as map?  Default is False.
    degrade_nside : `int`, optional
        Degrade map to this nside on read.  None means leave as-is.
        Not yet implemented for parquet.
    weightfile : `str`, optional
        Floating-point map to supply weights for degrade wmean.  Must
        be a HealSparseMap (weighted degrade not supported for
        healpix degrade-on-read).
        Not yet implemented for parquet.
    reduction : `str`, optional
        Reduction method with degrade-on-read.
        (mean, median, std, max, min, and, or, sum, prod, wmean).
        Not yet implemented for parquet.
    use_threads : `bool`, optional
        Use multithreaded reading.

    Returns
    -------
    healSparseMap : `HealSparseMap`
        HealSparseMap from file, covered by pixels
    header : `astropy.io.fits.Header` (if header=True)
        Header metadata for the map file.
    """
    ds = dataset.dataset(filepath, format='parquet', partitioning='hive')
    schema = ds.schema
    # Convert from byte strings
    md = {key.decode(): schema.metadata[key].decode()
          for key in schema.metadata}

    if 'healsparse::filetype' not in md:
        raise RuntimeError("Filepath %s is not a healsparse parquet map." % (filepath))
    if md['healsparse::filetype'] != 'healsparse':
        raise RuntimeError("Filepath %s is not a healsparse parquet map." % (filepath))
    cov_fname = os.path.join(filepath, '_coverage.parquet')
    if not os.path.isfile(cov_fname):
        # Note that this could be reconstructed from the information in the file
        # inefficiently.  This feature could be added in the future.
        raise RuntimeError("Filepath %s is missing coverage map %s" % (filepath, cov_fname))
    nside_sparse = int(md['healsparse::nside_sparse'])
    nside_coverage = int(md['healsparse::nside_coverage'])
    nside_io = int(md['healsparse::nside_io'])
    bitshift_io = _compute_bitshift(nside_io, nside_coverage)

    cov_tab = parquet.read_table(cov_fname, use_threads=use_threads)
    cov_pixels = cov_tab['cov_pix'].to_numpy()
    row_groups = cov_tab['row_group'].to_numpy()

    if pixels is not None:
        _pixels = np.atleast_1d(pixels)
        if len(np.unique(_pixels)) < len(_pixels):
            raise RuntimeError("Input list of pixels must be unique.")

        sub = np.clip(np.searchsorted(cov_pixels, _pixels), 0, cov_pixels.size - 1)
        ok, = np.where(cov_pixels[sub] == _pixels)
        if ok.size == 0:
            raise RuntimeError("None of the specified pixels are in the coverage map.")
        _pixels = np.sort(_pixels[ok])

        _pixels_io = np.right_shift(_pixels, bitshift_io)

        # Figure out row groups...
        matches = np.searchsorted(cov_pixels, _pixels)
        _row_groups_io = row_groups[matches]
    else:
        _pixels = cov_pixels
        _pixels_io = None
        _row_groups_io = None

    cov_map = HealSparseCoverage.make_from_pixels(nside_coverage, nside_sparse, _pixels)

    if md['healsparse::widemask'] == 'True':
        is_wide_mask = True
        wmult = int(md['healsparse::wwidth'])
    else:
        is_wide_mask = False
        wmult = 1

    if md['healsparse::bitpacked'] == 'True':
        is_bit_packed = True
        wdiv = 8
    else:
        is_bit_packed = False
        wdiv = 1

    if md['healsparse::primary'] != '':
        # This is a multi-column table.
        is_rec_array = True
        primary = md['healsparse::primary']
        columns = [name for name in schema.names if name not in ['iopix', 'cov_pix']]
        dtype = [(name, to_numpy_dtype(schema.field(name).type)) for
                 name in columns]
        primary_dtype = to_numpy_dtype(schema.field(primary).type)
    else:
        is_rec_array = False
        primary = None
        dtype = to_numpy_dtype(schema.field('sparse').type)
        primary_dtype = dtype
        columns = ['sparse']

    if md['healsparse::sentinel'] == 'UNSEEN':
        sentinel = primary_dtype(hpg.UNSEEN)
    elif md['healsparse::sentinel'] == 'False':
        sentinel = False
    elif md['healsparse::sentinel'] == 'True':
        sentinel = True
    else:
        sentinel = primary_dtype(md['healsparse::sentinel'])

        if is_integer_value(sentinel):
            sentinel = int(sentinel)
        elif not isinstance(sentinel, np.bool_):
            sentinel = float(sentinel)

    if is_rec_array:
        sparse_map = np.zeros((_pixels.size + 1)*cov_map.nfine_per_cov, dtype=dtype)
        # Fill in the overflow (primary)
        sparse_map[primary][: cov_map.nfine_per_cov] = sentinel
        # Fill in the overflow (not primary)
        for d in dtype:
            if d[0] == primary:
                continue
            sparse_map[d[0]][: cov_map.nfine_per_cov] = check_sentinel(d[1], None)
    else:
        sparse_map = np.zeros((_pixels.size + 1)*cov_map.nfine_per_cov*wmult//wdiv, dtype=dtype)
        sparse_map[: cov_map.nfine_per_cov*wmult//wdiv] = sentinel

    if _pixels_io is None:
        # Read the full table
        tab = ds.to_table(columns=columns, use_threads=use_threads)
    else:
        _pixels_io_unique = list(np.unique(_pixels_io))

        fragments = list(ds.get_fragments(filter=dataset.field('iopix').isin(_pixels_io_unique)))
        group_fragments = []
        for pixel_io, fragment in zip(_pixels_io_unique, fragments):
            groups = fragment.split_by_row_group()
            # Only append groups that are relevant
            use, = np.where(_pixels_io == pixel_io)
            for ind in use:
                group_fragments.append(groups[_row_groups_io[ind]])

        ds2 = dataset.FileSystemDataset(group_fragments, schema, ds.format)
        tab = ds2.to_table(columns=columns, use_threads=use_threads)

    if is_rec_array:
        for name in columns:
            sparse_map[name][cov_map.nfine_per_cov:] = tab[name].to_numpy()
    else:
        sparse_map[cov_map.nfine_per_cov*wmult//wdiv:] = tab['sparse'].to_numpy()

        if is_wide_mask:
            sparse_map = sparse_map.reshape((sparse_map.size // wmult,
                                             wmult)).astype(WIDE_MASK)
        elif is_bit_packed:
            sparse_map = _PackedBoolArray(data_buffer=sparse_map)

    healsparse_map = healsparse_class(cov_map=cov_map, sparse_map=sparse_map,
                                      nside_sparse=nside_sparse, primary=primary,
                                      sentinel=sentinel)

    if header:
        if 'healsparse::header' in md:
            hdr_string = md['healsparse::header']
            hdr = fits.Header.fromstring(hdr_string)
        else:
            hdr = fits.Header()

        return (healsparse_map, hdr)
    else:
        return healsparse_map


def _write_map_parquet(hsp_map, filepath, clobber=False, nside_io=4):
    """
    Internal method to write a HealSparseMap to a parquet dataset.
    use the `metadata` property from the map to persist additional
    information in the parquet metadata.

    Parameters
    ----------
    hsp_map : `HealSparseMap`
        HealSparseMap to write to a file.
    filepath : `str`
        Name of dataset to save
    clobber : `bool`, optional
        Clobber existing file?  Not supported.
    nside_io : `int`, optional
        The healpix nside to partition the output map files in parquet.
        Must be less than or equal to nside_coverage, and not greater than 16.

    Raises
    ------
    RuntimeError if file exists.
    ValueError if nside_io is out of range.
    """
    if os.path.isfile(filepath) or os.path.isdir(filepath):
        raise RuntimeError("Filepath %s exists and clobber is not supported." % (filepath))

    if nside_io > hsp_map.nside_coverage:
        raise ValueError("nside_io must be <= nside_coverage.")
    elif nside_io > 16:
        raise ValueError("nside_io must be <= 16")
    elif nside_io < 0:
        raise ValueError("nside_io must be >= 0")

    # Make the path
    os.makedirs(filepath)

    # Create the nside_io paths
    cov_mask = hsp_map.coverage_mask

    cov_pixels = np.where(cov_mask)[0].astype(np.int32)
    bitshift_io = _compute_bitshift(nside_io, hsp_map.nside_coverage)
    cov_pixels_io = np.right_shift(cov_pixels, bitshift_io)

    if hsp_map.is_wide_mask_map:
        wmult = hsp_map.wide_mask_width
    else:
        wmult = 1

    if hsp_map.is_bit_packed_map:
        wdiv = 8
    else:
        wdiv = 1

    if np.isclose(hsp_map._sentinel, hpg.UNSEEN):
        sentinel_string = 'UNSEEN'
    else:
        sentinel_string = str(hsp_map._sentinel)

    metadata = {'healsparse::version': '1',
                'healsparse::nside_sparse': str(hsp_map.nside_sparse),
                'healsparse::nside_coverage': str(hsp_map.nside_coverage),
                'healsparse::nside_io': str(nside_io),
                'healsparse::filetype': 'healsparse',
                'healsparse::primary': '' if hsp_map.primary is None else hsp_map.primary,
                'healsparse::sentinel': sentinel_string,
                'healsparse::widemask': str(hsp_map.is_wide_mask_map),
                'healsparse::wwidth': str(hsp_map._wide_mask_width),
                'healsparse::bitpacked': str(hsp_map.is_bit_packed_map)}

    # Add additional metadata
    if hsp_map.metadata is not None:
        # Use the fits header serialization for compatibility
        hdr_string = str(_make_header(hsp_map.metadata, force_astropy=True))
        metadata['healsparse::header'] = hdr_string

    if hsp_map.is_bit_packed_map:
        sparse_map = hsp_map._sparse_map.data_array
    else:
        sparse_map = hsp_map._sparse_map.ravel()

    if not hsp_map.is_rec_array:
        schema = pa.schema([('cov_pix', pa.from_numpy_dtype(np.int32)),
                            ('sparse', pa.from_numpy_dtype(sparse_map.dtype))],
                           metadata=metadata)
    else:
        type_list = [(name, pa.from_numpy_dtype(hsp_map.dtype[name].type)) for
                     name in hsp_map.dtype.names]
        type_list[0: 0] = [('cov_pix', pa.from_numpy_dtype(np.int32))]
        schema = pa.schema(type_list, metadata=metadata)

    cov_map = hsp_map._cov_map
    cov_index_map_temp = cov_map[:] + np.arange(hpg.nside_to_npixel(hsp_map.nside_coverage),
                                                dtype=np.int64)*cov_map.nfine_per_cov

    cpix_arr = np.zeros(cov_map.nfine_per_cov*wmult//wdiv, dtype=np.int32)

    last_cpix_io = -1
    writer = None
    row_groups = np.zeros_like(cov_pixels)
    for ctr, (cpix_io, cpix) in enumerate(zip(cov_pixels_io, cov_pixels)):
        # These are always going to be sorted
        if cpix_io > last_cpix_io:
            last_cpix_io = cpix_io

            if writer is not None:
                writer.close()
                writer = None

            # Create a new file
            iopixpath = os.path.join(filepath, f'iopix={cpix_io:03d}')
            os.makedirs(iopixpath)

            iopixfile = os.path.join(iopixpath, f'{cpix_io:03d}.parquet')
            writer = parquet.ParquetWriter(iopixfile, schema)
            row_group_ctr = 0

        sparsepix = sparse_map[cov_index_map_temp[cpix]*wmult//wdiv:
                               (cov_index_map_temp[cpix] + cov_map.nfine_per_cov)*wmult//wdiv]
        cpix_arr[:] = cpix
        if not hsp_map.is_rec_array:
            arrays = [pa.array(cpix_arr),
                      pa.array(sparsepix)]
        else:
            arrays = [pa.array(sparsepix[name]) for
                      name in hsp_map.dtype.names]
            arrays[0: 0] = [pa.array(cpix_arr)]
        tab = pa.Table.from_arrays(arrays, schema=schema)

        # Ensure we write this as one row group.
        writer.write_table(tab, row_group_size=len(tab))
        row_groups[ctr] = row_group_ctr
        row_group_ctr += 1

    if writer is not None:
        writer.close()

    # And write the coverage pixels and row groups
    tab = pa.Table.from_pydict({'cov_pix': pa.array(cov_pixels),
                                'row_group': pa.array(row_groups)})
    parquet.write_table(tab, os.path.join(filepath, '_coverage.parquet'))

    # And write the metadata
    parquet.write_metadata(schema, os.path.join(filepath, '_common_metadata'))
    parquet.write_metadata(schema, os.path.join(filepath, '_metadata'))
