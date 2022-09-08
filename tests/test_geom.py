import unittest
import numpy.testing as testing

import numpy as np

import healsparse
from healsparse import Circle, Polygon, Ellipse


def atbound(longitude, minval, maxval):
    w, = np.where(longitude < minval)
    while w.size > 0:
        longitude[w] += 360.0
        w, = np.where(longitude < minval)

    w, = np.where(longitude > maxval)
    while w.size > 0:
        longitude[w] -= 360.0
        w, = np.where(longitude > maxval)

    return


def _randcap(rng, nrand, ra, dec, rad, get_radius=False):
    """
    Generate random points in a spherical cap

    parameters
    ----------

    nrand:
        The number of random points
    ra,dec:
        The center of the cap in degrees.  The ra should be within [0,360) and
        dec from [-90,90]
    rad:
        radius of the cap, same units as ra,dec

    get_radius: bool, optional
        if true, return radius of each point in radians
    """
    # generate uniformly in r**2
    rand_r = rng.uniform(size=nrand)
    rand_r = np.sqrt(rand_r)*rad

    # put in degrees
    np.deg2rad(rand_r, rand_r)

    # generate position angle uniformly 0,2*PI
    rand_posangle = rng.uniform(nrand)*2*np.pi

    theta = np.array(dec, dtype='f8', ndmin=1, copy=True)
    phi = np.array(ra, dtype='f8', ndmin=1, copy=True)
    theta += 90

    np.deg2rad(theta, theta)
    np.deg2rad(phi, phi)

    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    sinr = np.sin(rand_r)
    cosr = np.cos(rand_r)

    cospsi = np.cos(rand_posangle)
    costheta2 = costheta*cosr + sintheta*sinr*cospsi

    np.clip(costheta2, -1, 1, costheta2)

    # gives [0,pi)
    theta2 = np.arccos(costheta2)
    sintheta2 = np.sin(theta2)

    cosDphi = (cosr - costheta*costheta2)/(sintheta*sintheta2)

    np.clip(cosDphi, -1, 1, cosDphi)
    Dphi = np.arccos(cosDphi)

    # note fancy usage of where
    phi2 = np.where(rand_posangle > np.pi, phi+Dphi, phi-Dphi)

    np.rad2deg(phi2, phi2)
    np.rad2deg(theta2, theta2)
    rand_ra = phi2
    rand_dec = theta2-90.0

    atbound(rand_ra, 0.0, 360.0)

    if get_radius:
        np.rad2deg(rand_r, rand_r)
        return rand_ra, rand_dec, rand_r
    else:
        return rand_ra, rand_dec


class GeomTestCase(unittest.TestCase):
    def test_circle_smoke(self):
        """
        just test we can make a circle and a map from it
        """
        ra, dec = 200.0, 0.0
        radius = 30.0/3600.0
        nside = 2**17
        circle = Circle(
            ra=ra,
            dec=dec,
            radius=radius,
            value=2**4,
        )

        pixels = circle.get_pixels(nside=nside)
        self.assertGreater(pixels.size, 0)

        smap = circle.get_map(nside_coverage=32, nside_sparse=nside, dtype=np.int16)
        self.assertTrue(isinstance(smap, healsparse.HealSparseMap))

        smap2 = circle.get_map_like(smap)
        self.assertEqual(smap2.nside_coverage, smap.nside_coverage)
        self.assertEqual(smap2.nside_sparse, smap.nside_sparse)
        self.assertEqual(smap2.dtype, smap.dtype)

    def test_circle_values(self):
        """
        make sure we get out the value we used for the map

        Note however that we do not use inclusive intersections, we we will test
        values from a slightly smaller circle
        """

        rng = np.random.RandomState(7812)
        nside = 2**17

        ra, dec = 200.0, 0.0
        radius = 30.0/3600.0
        circle = Circle(
            ra=ra,
            dec=dec,
            radius=radius,
            value=2**4,
        )

        smap = circle.get_map(nside_coverage=32, nside_sparse=nside, dtype=np.int16)

        # test points we expect to be inside
        smallrad = radius*0.95
        nrand = 10000
        rra, rdec = _randcap(rng, nrand, ra, dec, smallrad)

        vals = smap.get_values_pos(rra, rdec, lonlat=True)

        testing.assert_array_equal(vals, circle.value)

        # test points we expect to be outside
        bigrad = radius*2
        nrand = 10000
        rra, rdec, rrand = _randcap(
            rng,
            nrand,
            ra,
            dec,
            bigrad,
            get_radius=True,
        )
        w, = np.where(rrand > 1.1*radius)

        vals = smap.get_values_pos(rra[w], rdec[w], lonlat=True)

        testing.assert_array_equal(vals, 0)

        # And test floating point values
        circle = Circle(
            ra=ra,
            dec=dec,
            radius=radius,
            value=2.0,
        )
        smap2 = circle.get_map(nside_coverage=32, nside_sparse=nside, dtype=np.float32)
        testing.assert_array_equal(smap.valid_pixels, smap2.valid_pixels)
        testing.assert_array_equal(smap2.get_values_pix(smap2.valid_pixels), 2.0)

    def test_bad_values(self):
        """
        test that we can only set Circles with scalars
        and that the values should be either scalar or bits if the map is
        a wide-mask
        """
        ra = [1., 2.]
        dec = 1.
        radius = 1.
        value = 1.
        testing.assert_raises(ValueError, Circle, ra=ra, dec=dec, radius=radius, value=value)

        testing.assert_raises(
            ValueError,
            Ellipse,
            ra=ra,
            dec=dec,
            semi_major=radius,
            semi_minor=radius,
            alpha=0.0,
            value=1.0,
        )

    def test_polygon_smoke(self):
        """
        just test we can make a polygon and a map from it
        """
        nside = 2**15

        # counter clockwise
        ra = [200.0, 200.2, 200.3, 200.2, 200.1]
        dec = [0.0, 0.1, 0.2, 0.25, 0.13]
        poly = Polygon(
            ra=ra,
            dec=dec,
            value=64,
        )

        smap = poly.get_map(nside_coverage=32, nside_sparse=nside, dtype=np.int16)

        ra = np.array([200.1, 200.15])
        dec = np.array([0.05, 0.015])
        vals = smap.get_values_pos(ra, dec, lonlat=True)

        testing.assert_array_equal(vals, [poly.value, 0])

        smap2 = poly.get_map_like(smap)
        self.assertEqual(smap2.nside_coverage, smap.nside_coverage)
        self.assertEqual(smap2.nside_sparse, smap.nside_sparse)
        self.assertEqual(smap2.dtype, smap.dtype)

    def test_polygon_values(self):
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
        poly = Polygon(
            ra=ra,
            dec=dec,
            value=64,
        )

        smap = poly.get_map(nside_coverage=32, nside_sparse=nside, dtype=np.int16)

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

        vals = smap.get_values_pos(rra, rdec, lonlat=True)

        testing.assert_array_equal(vals, poly.value)

        # And test floating point values
        poly = Polygon(
            ra=ra,
            dec=dec,
            value=2.0,
        )
        smap2 = poly.get_map(nside_coverage=32, nside_sparse=nside, dtype=np.float32)
        testing.assert_array_equal(smap.valid_pixels, smap2.valid_pixels)
        testing.assert_array_equal(smap2.get_values_pix(smap2.valid_pixels), 2.0)

    def test_ellipse_smoke(self):
        """
        Test that we can make an ellipse and a map from it.
        """
        ra, dec = 200.0, 0.0
        semi_major = 30.0/3600.0
        semi_minor = 15.0/3600.0
        alpha = 45.0
        nside = 2**17
        ellipse = Ellipse(
            ra=ra,
            dec=dec,
            semi_major=semi_major,
            semi_minor=semi_minor,
            alpha=alpha,
            value=2**4,
        )

        pixels = ellipse.get_pixels(nside=nside)
        self.assertGreater(pixels.size, 0)

        smap = ellipse.get_map(nside_coverage=32, nside_sparse=nside, dtype=np.int16)
        self.assertTrue(isinstance(smap, healsparse.HealSparseMap))

        smap2 = ellipse.get_map_like(smap)
        self.assertEqual(smap2.nside_coverage, smap.nside_coverage)
        self.assertEqual(smap2.nside_sparse, smap2.nside_sparse)
        self.assertEqual(smap2.dtype, smap.dtype)

    def test_ellipse_values(self):
        """
        Make sure we get out the values we used for the map.

        Note however that we do not use inclusive intersections, we will test values
        from a slightly smaller ellipse.
        """
        rng = np.random.RandomState(7812)
        nside = 2**17

        ra, dec = 200.0, 0.0
        semi_major = 30.0/3600.0
        semi_minor = 15.0/3600.0
        alpha = 45.0
        ellipse = Ellipse(
            ra=ra,
            dec=dec,
            semi_major=semi_major,
            semi_minor=semi_minor,
            alpha=alpha,
            value=2**4,
        )

        smap = ellipse.get_map(nside_coverage=32, nside_sparse=nside, dtype=np.int16)

        # Test points we know to be inside.
        smallrad = semi_minor*0.95
        nrand = 10000
        rra, rdec = _randcap(rng, nrand, ra, dec, smallrad)

        values = smap.get_values_pos(rra, rdec)

        testing.assert_array_equal(values, ellipse.value)

        # test points we expect to be outside
        bigrad = semi_major*2
        nrand = 10000
        rra, rdec, rrand = _randcap(
            rng,
            nrand,
            ra,
            dec,
            bigrad,
            get_radius=True,
        )
        w, = np.where(rrand > 1.1*semi_major)

        vals = smap.get_values_pos(rra[w], rdec[w])
        testing.assert_array_equal(vals, 0)

        # And test floating point values
        ellipse = Ellipse(
            ra=ra,
            dec=dec,
            semi_major=semi_major,
            semi_minor=semi_minor,
            alpha=alpha,
            value=2.0,
        )
        smap2 = ellipse.get_map(nside_coverage=32, nside_sparse=nside, dtype=np.float32)
        testing.assert_array_equal(smap.valid_pixels, smap2.valid_pixels)
        testing.assert_array_equal(smap2.get_values_pix(smap2.valid_pixels), 2.0)

    def test_realize_geom_values(self):
        """
        test "or"ing two geom objects
        """
        nside = 2**17

        for dtype in [np.int16, np.uint16]:
            radius1 = 0.075
            radius2 = 0.075
            ra1, dec1 = 200.0, 0.0
            ra2, dec2 = 200.1, 0.0
            value1 = 2**2
            value2 = 2**4

            circle1 = Circle(
                ra=ra1,
                dec=dec1,
                radius=radius1,
                value=value1,
            )
            circle2 = Circle(
                ra=ra2,
                dec=dec2,
                radius=radius2,
                value=value2,
            )

            smap = healsparse.HealSparseMap.make_empty(
                nside_coverage=32,
                nside_sparse=nside,
                dtype=dtype,
                sentinel=0,
            )
            healsparse.realize_geom([circle1, circle2], smap)

            out_ra, out_dec = 190.0, 25.0
            in1_ra, in1_dec = 200.02, 0.0
            in2_ra, in2_dec = 200.095, 0.0
            both_ra, both_dec = 200.05, 0.0

            out_vals = smap.get_values_pos(out_ra, out_dec, lonlat=True)
            in1_vals = smap.get_values_pos(in1_ra, in1_dec, lonlat=True)
            in2_vals = smap.get_values_pos(in2_ra, in2_dec, lonlat=True)
            both_vals = smap.get_values_pos(both_ra, both_dec, lonlat=True)

            testing.assert_array_equal(out_vals, 0)
            testing.assert_array_equal(in1_vals, value1)
            testing.assert_array_equal(in2_vals, value2)
            testing.assert_array_equal(both_vals, (value1 | value2))

    def test_repr(self):
        """
        Test representations
        """
        # The following is needed to eval the repr.
        from numpy import array # noqa

        ra, dec = 200.0, 0.0
        radius = 30.0/3600.0
        circle = Circle(
            ra=ra,
            dec=dec,
            radius=radius,
            value=2**4,
        )

        rep = repr(circle)
        circle_rep = eval(rep)

        testing.assert_almost_equal(circle._ra, circle_rep._ra)
        testing.assert_almost_equal(circle._dec, circle_rep._dec)
        testing.assert_almost_equal(circle._radius, circle_rep._radius)
        testing.assert_array_equal(circle._value, circle_rep._value)

        circle._value = 5.6
        rep = repr(circle)
        circle_rep = eval(rep)
        testing.assert_array_equal(circle._value, circle_rep._value)

        # And multi-bit version
        circle = Circle(
            ra=ra,
            dec=dec,
            radius=radius,
            value=[5, 10],
        )

        rep = repr(circle)
        circle_rep = eval(rep)

        testing.assert_almost_equal(circle._ra, circle_rep._ra)
        testing.assert_almost_equal(circle._dec, circle_rep._dec)
        testing.assert_almost_equal(circle._radius, circle_rep._radius)
        testing.assert_array_equal(circle._value, circle_rep._value)

        ra, dec = 200.0, 0.0
        semi_major = 30.0/3600.0
        semi_minor = 15.0/3600.0
        alpha = 45.0
        ellipse = Ellipse(
            ra=ra,
            dec=dec,
            semi_major=semi_major,
            semi_minor=semi_minor,
            alpha=alpha,
            value=2**4,
        )

        rep = repr(ellipse)
        ellipse_rep = eval(rep)

        testing.assert_almost_equal(ellipse._ra, ellipse_rep._ra)
        testing.assert_almost_equal(ellipse._dec, ellipse_rep._dec)
        testing.assert_almost_equal(ellipse._semi_major, ellipse_rep._semi_major)
        testing.assert_almost_equal(ellipse._semi_minor, ellipse_rep._semi_minor)
        testing.assert_almost_equal(ellipse._alpha, ellipse_rep._alpha)
        testing.assert_array_equal(ellipse._value, ellipse_rep._value)

        ra = [200.0, 200.2, 200.3, 200.2, 200.1]
        dec = [0.0, 0.1, 0.2, 0.25, 0.13]
        poly = Polygon(
            ra=ra,
            dec=dec,
            value=64,
        )

        rep = repr(poly)
        poly_rep = eval(rep)

        testing.assert_almost_equal(poly._ra, poly_rep._ra)
        testing.assert_almost_equal(poly._dec, poly_rep._dec)
        testing.assert_array_equal(poly._value, poly_rep._value)

        poly._value = 5.6
        rep = repr(poly)
        poly_rep = eval(rep)
        testing.assert_array_equal(poly._value, poly_rep._value)

        # And multi-bit version
        poly = Polygon(
            ra=ra,
            dec=dec,
            value=[5, 10],
        )

        rep = repr(poly)
        poly_rep = eval(rep)

        testing.assert_almost_equal(poly._ra, poly_rep._ra)
        testing.assert_almost_equal(poly._dec, poly_rep._dec)
        testing.assert_array_equal(poly._value, poly_rep._value)


if __name__ == '__main__':
    unittest.main()
