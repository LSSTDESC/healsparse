import subprocess
import sys
import time
import urllib.request

import numpy as np
import pytest
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
    """Serve a healsparse file over localhost HTTP from a separate process."""
    root = tmp_path_factory.mktemp("hdf5")
    fname = "test_healsparse_map.hdf5"
    _get_simple_hsp_map().write(root / fname, format="hdf5")

    code = (
        "from http.server import ThreadingHTTPServer;"
        "from functools import partial;"
        "from RangeHTTPServer import RangeRequestHandler;"
        "RangeRequestHandler.protocol_version = 'HTTP/1.1';"
        f"h = partial(RangeRequestHandler, directory={str(root)!r});"
        "srv = ThreadingHTTPServer(('127.0.0.1', 0), h);"
        "print(srv.server_address[1], flush=True);"
        "srv.serve_forever()"
    )
    # stdout piped only for the port line; stderr inherited so request
    # logs go to pytest's capture (never pipe stderr here — the child
    # logs every request to it, and an undrained pipe would fill and
    # wedge the server).
    proc = subprocess.Popen([sys.executable, "-c", code],
                            stdout=subprocess.PIPE, text=True)

    port_line = proc.stdout.readline()  # blocks until the child binds
    if not port_line.strip():
        proc.wait(timeout=5)
        raise RuntimeError("HTTP server subprocess failed to start")
    port = int(port_line.strip())
    base = f"http://127.0.0.1:{port}/"

    # The port line means the constructor returned, i.e. the socket is
    # already bound and listening; this poll just waits for the accept
    # loop, and normally passes on the first try.
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
