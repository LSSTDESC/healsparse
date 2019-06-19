import numpy as np
import healsparse


def test_polygon_smoke():
    """
    just test we can make a polygon and a map from it
    """
    nside = 2**15

    # counter clockwise
    ra = [200.0, 200.2, 200.3, 200.2, 200.1]
    dec = [0.0,     0.1,   0.2,   0.25, 0.13]
    poly = healsparse.Polygon(
        ra=ra,
        dec=dec,
        value=64,
    )

    smap = poly.get_map(nside=nside, dtype=np.int16)

    ra = np.array([200.1, 200.15])
    dec = np.array([0.05, 0.015])
    vals = smap.getValueRaDec(ra, dec)  # noqa


def test_polygon_values():
    """
    make sure we get out the value we used for the map

    Use a box so "truth" is easy to calculate.  Note however
    that we do not use inclusive intersections, we we will
    test values from a slightly smaller box
    """
    nside = 2**15
    rng = np.random.RandomState(8312)
    nrand = 10000

    # make a box
    ra_range = 200.0, 200.1
    dec_range = 0.1, 0.2

    ra = [ra_range[0], ra_range[1], ra_range[1], ra_range[0]]
    dec = [dec_range[0], dec_range[0], dec_range[1], dec_range[1]]
    poly = healsparse.Polygon(
        ra=ra,
        dec=dec,
        value=64,
    )

    smap = poly.get_map(nside=nside, dtype=np.int16)

    rad = 0.1*(ra_range[1] - ra_range[0])
    decd = 0.1*(dec_range[1] - dec_range[0])

    rra = rng.uniform(
        low=ra_range[0]+rad,
        high=ra_range[1]-rad,
        size=nrand,
    )
    rdec = rng.uniform(
        low=dec_range[0]+decd,
        high=dec_range[1]-decd,
        size=nrand,
    )

    vals = smap.getValueRaDec(rra, rdec)  # noqa

    assert np.all(vals == poly.value)
