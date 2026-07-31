import os
import subprocess
import sys
import time
import urllib

import numpy as np
import pytest
import healsparse


_SERVER = os.path.join(os.path.dirname(__file__), "_range_server.py")


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
    """Serve a healsparse file over localhost HTTP from a separate process."""
    root = tmp_path_factory.mktemp("hsp")
    fname = "test_healsparse_map.hsp"
    _get_simple_hsp_map().write(root / fname)

    proc = subprocess.Popen(
        [sys.executable, _SERVER, str(root)],
        stdout=subprocess.PIPE, text=True,
    )

    port_line = proc.stdout.readline()
    if not port_line.strip():
        proc.wait(timeout=5)
        raise RuntimeError("HTTP server subprocess failed to start")
    port = int(port_line.strip())
    base = f"http://127.0.0.1:{port}/"

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(base, timeout=0.25).close()
            break
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError("HTTP server subprocess exited early")
            time.sleep(0.05)
    else:
        proc.terminate()
        raise RuntimeError("HTTP server did not start in time")

    try:
        yield f"{base}{fname}"
    finally:
        proc.terminate()
        proc.wait()


def test_remote_uri_open(served_healsparse_fits):
    """Test remote open (directly)."""
    fits = healsparse.fits_shim.HealSparseFits(served_healsparse_fits)

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
