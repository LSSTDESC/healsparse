import socketserver
import threading
from functools import partial

import numpy as np
import pytest
from RangeHTTPServer import RangeRequestHandler

import healsparse


if not healsparse.io_map_hdf5.use_hdf5:
    pytest.skip("Skipping hdf5 tests", allow_module_level=True)


def _get_simple_hsp_map():
    nside_coverage = 32
    nside_map = 1024

    n_pt = 100000
    pix = np.arange(n_pt) + 100000

    m = healsparse.HealSparseMap.make_empty(
        nside_coverage=nside_coverage,
        nside_sparse=nside_map,
        dtype=np.float64,
    )
    m[pix] = np.arange(n_pt, dtype=np.float64)

    return m


@pytest.fixture(scope="session")
def served_healsparse_hdf5(tmp_path_factory):
    """Write a small healsparse file and serve its directory over localhost HTTP."""
    root = tmp_path_factory.mktemp("hdf5")

    fname = "test_healsparse_map.hdf5"

    m = _get_simple_hsp_map()
    m.write(root / fname, format="hdf5")

    handler = partial(RangeRequestHandler, directory=str(root))
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as httpd:
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            yield f"http://127.0.0.1:{port}/{fname}"
        finally:
            httpd.shutdown()


def test_remote_read_over_https(served_healsparse_hdf5):
    """Test remote reading over https (partial and full)."""

    m_local = _get_simple_hsp_map()

    # Read in the coverage map only.
    cov_map_remote = healsparse.HealSparseCoverage.read(served_healsparse_hdf5)

    assert cov_map_remote.nside_sparse == m_local.nside_sparse
    assert cov_map_remote.nside_coverage == m_local.nside_coverage
    np.testing.assert_array_equal(cov_map_remote.coverage_mask, m_local.coverage_mask)

    # Read in one pixel only.
    pixels, = np.where(m_local.coverage_mask)

    m_remote_sub = healsparse.HealSparseMap.read(served_healsparse_hdf5, pixels=[pixels[0]])

    m_local_sub = m_local.get_single_covpix_map(pixels[0])

    assert m_remote_sub == m_local_sub
