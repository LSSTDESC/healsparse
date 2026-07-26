import socketserver
import threading
from functools import partial

import numpy as np
import pytest
from RangeHTTPServer import RangeRequestHandler

import healsparse


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
def served_healsparse_fits(tmp_path_factory):
    """Write a small healsparse file and serve its directory over localhost HTTP."""
    root = tmp_path_factory.mktemp("hsp")

    fname = "test_healsparse_map.hsp"

    m = _get_simple_hsp_map()
    m.write(root / fname)

    handler = partial(RangeRequestHandler, directory=str(root))
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as httpd:
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            yield f"http://127.0.0.1:{port}/{fname}"
        finally:
            httpd.shutdown()


def test_remote_uri_open(served_healsparse_fits):
    """Test remote open (directly)."""
    fits = healsparse.fits_shim.HealSparseFits(served_healsparse_fits)

    assert fits._use_astropy
    assert not fits._use_rustfits
    assert not fits._use_fitsio

    hdr = fits.read_ext_header("SPARSE")

    m_local = _get_simple_hsp_map()

    assert hdr["NSIDE"] == m_local.nside_sparse


def test_remote_uri_open_fail(served_healsparse_fits):
    """Test remote open (no file)."""
    remote = served_healsparse_fits

    with pytest.raises(IOError):
        _ = healsparse.fits_shim.HealSparseFits(remote + ".notafile")


def test_local_uri_open_fail(tmp_path):
    """Test local open (no file)."""
    with pytest.raises(IOError):
        _ = healsparse.fits_shim.HealSparseFits(tmp_path / "notafile")


def test_local_uri_open(tmp_path):
    """Test local open (directly) with a file:// URI."""
    fname = "test_healsparse_map.hsp"

    m_local = _get_simple_hsp_map()
    m_local.write(tmp_path / fname)

    fits = healsparse.fits_shim.HealSparseFits(tmp_path / fname)

    hdr = fits.read_ext_header("SPARSE")

    assert hdr["NSIDE"] == m_local.nside_sparse


def test_remote_read_over_https(served_healsparse_fits):
    """Test remote reading over https (partial and full)."""

    m_local = _get_simple_hsp_map()

    # Read in the coverage map only.
    cov_map_remote = healsparse.HealSparseCoverage.read(served_healsparse_fits)

    assert cov_map_remote.nside_sparse == m_local.nside_sparse
    assert cov_map_remote.nside_coverage == m_local.nside_coverage
    np.testing.assert_array_equal(cov_map_remote.coverage_mask, m_local.coverage_mask)

    # Read in one pixel only.
    pixels, = np.where(m_local.coverage_mask)

    m_remote_sub = healsparse.HealSparseMap.read(served_healsparse_fits, pixels=[pixels[0]])

    m_local_sub = m_local.get_single_covpix_map(pixels[0])

    assert m_remote_sub == m_local_sub
