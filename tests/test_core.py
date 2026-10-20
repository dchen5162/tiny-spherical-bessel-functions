import math
import unittest

from tiny_spherical_bessel_functions import spherical_jn, spherical_yn


class TestSphericalJn(unittest.TestCase):

    def test_j0_known_value(self):
        # j_0(x) = sin(x)/x
        for x in [1.0, 2.5, 4.0, 10.0]:
            expected = math.sin(x) / x
            result = spherical_jn(0, x)
            self.assertAlmostEqual(result, expected, places=12)

    def test_j0_at_zero(self):
        self.assertEqual(spherical_jn(0, 0.0), 1.0)

    def test_jn_at_zero(self):
        for n in [1, 2, 5, 10]:
            self.assertEqual(spherical_jn(n, 0.0), 0.0)

    def test_j1_known_value(self):
        # j_1(x) = sin(x)/x^2 - cos(x)/x
        for x in [1.0, 3.0, 5.0, 8.0]:
            expected = math.sin(x) / (x * x) - math.cos(x) / x
            result = spherical_jn(1, x)
            self.assertAlmostEqual(result, expected, places=12)

    def test_j2_recurrence(self):
        # j_2(x) = (3/x) j_1(x) - j_0(x)
        for x in [2.0, 4.0, 6.0, 9.0]:
            j0 = math.sin(x) / x
            j1 = math.sin(x) / (x * x) - math.cos(x) / x
            expected = (3.0 / x) * j1 - j0
            result = spherical_jn(2, x)
            self.assertAlmostEqual(result, expected, places=12)

    def test_jn_negative_x(self):
        # j_n(-x) = (-1)^n j_n(x)
        x = 3.0
        for n in [0, 1, 2, 3]:
            pos = spherical_jn(n, x)
            neg = spherical_jn(n, -x)
            expected = ((-1) ** n) * pos
            self.assertAlmostEqual(neg, expected, places=10)

    def test_jn_small_argument(self):
        # For small x, j_n(x) ~ x^n / (2n+1)!! with one correction term
        x = 0.01
        for n in [0, 1, 2, 3]:
            df = 1.0
            for k in range(n + 1):
                df *= (2 * k + 1)
            leading = (x ** n) / df
            correction = 1.0 - x * x / (2.0 * (2 * n + 3))
            expected = leading * correction
            result = spherical_jn(n, x)
            self.assertAlmostEqual(result, expected, places=12)

    def test_jn_large_order_small_x(self):
        # j_10(0.01) is extremely small but not necessarily exactly 0.0
        result = spherical_jn(10, 0.01)
        self.assertTrue(abs(result) < 1e-20)

    def test_negative_order_raises(self):
        with self.assertRaises(ValueError):
            spherical_jn(-1, 1.0)


class TestSphericalYn(unittest.TestCase):

    def test_y0_known_value(self):
        # y_0(x) = -cos(x)/x
        for x in [1.0, 2.5, 4.0, 10.0]:
            expected = -math.cos(x) / x
            result = spherical_yn(0, x)
            self.assertAlmostEqual(result, expected, places=12)

    def test_y1_known_value(self):
        # y_1(x) = -cos(x)/x^2 - sin(x)/x
        for x in [1.0, 3.0, 5.0, 8.0]:
            expected = -math.cos(x) / (x * x) - math.sin(x) / x
            result = spherical_yn(1, x)
            self.assertAlmostEqual(result, expected, places=12)

    def test_y2_recurrence(self):
        # y_2(x) = (3/x) y_1(x) - y_0(x)
        for x in [2.0, 4.0, 6.0, 9.0]:
            y0 = -math.cos(x) / x
            y1 = -math.cos(x) / (x * x) - math.sin(x) / x
            expected = (3.0 / x) * y1 - y0
            result = spherical_yn(2, x)
            self.assertAlmostEqual(result, expected, places=12)

    def test_yn_at_zero_raises(self):
        with self.assertRaises(ValueError):
            spherical_yn(0, 0.0)

    def test_yn_small_argument(self):
        # For small x, y_n(x) ~ -(2n-1)!! / x^(n+1)
        x = 0.01
        for n in [1, 2, 3]:
            df = 1.0
            for k in range(n):
                df *= (2 * k + 1)
            expected = -df / (x ** (n + 1))
            result = spherical_yn(n, x)
            self.assertAlmostEqual(result, expected, places=6)

    def test_y0_small_argument(self):
        # y_0(x) ~ -1/x for small x
        x = 0.001
        expected = -1.0 / x
        result = spherical_yn(0, x)
        self.assertAlmostEqual(result, expected, places=6)

    def test_yn_negative_x(self):
        # y_n(-x) = (-1)^{n+1} y_n(x)
        x = 3.0
        for n in [0, 1, 2, 3]:
            pos = spherical_yn(n, x)
            neg = spherical_yn(n, -x)
            expected = ((-1) ** (n + 1)) * pos
            self.assertAlmostEqual(neg, expected, places=8)

    def test_negative_order_yn_raises(self):
        with self.assertRaises(ValueError):
            spherical_yn(-1, 1.0)


if __name__ == "__main__":
    unittest.main()
