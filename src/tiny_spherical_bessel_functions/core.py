"""Spherical Bessel functions of the first and second kind via recurrences.

All computation is done on plain Python floats. No NumPy, no arrays.
"""

import math


def _small_x_threshold(n):
    """Return the |x| below which the direct recurrence for j_n is unstable.

    For very small |x|, forward recurrence on j_n / j_{n-1} suffers from
    catastrophic cancellation because j_n is tiny and j_{n-1} is not. We
    switch to the small-argument series in that regime. The cutoff is
    heuristic; see the README for the trade-off.
    """
    return max(0.5, 0.3 * n)


def _small_argument_jn(n, x):
    """Compute j_n(x) for |x| << 1 using the leading terms of the series.

    j_n(x) ~ x^n / (2n+1)!!  [1 - x^2/(2(2n+3)) + O(x^4)]

    For small enough x the single leading factor is within float64 epsilon of
    the full series, so we use one correction term when x is not *that* small.
    """
    ax = abs(x)
    # (2n+1)!! = (2n+1)! / (2^n n!)
    # x^n / (2n+1)!! = x^n * 2^n * n! / (2n+1)!
    num = 1.0
    for k in range(n):
        num *= ax / (2 * (k + 1))
    # multiply by x^n * 2^n * n! / (2n+1)! ; we built x^n / (2n+1)!! directly
    # via the recurrence above, but let's re-derive cleanly.
    # Recompute from scratch to avoid confusion.
    double_factorial = 1.0
    for k in range(n + 1):
        double_factorial *= (2 * k + 1)
    leading = (ax ** n) / double_factorial
    correction = 1.0 - ax * ax / (2.0 * (2 * n + 3))
    val = leading * correction
    return val if x >= 0 else ((-1) ** n) * val


def _small_argument_yn(n, x):
    """Compute y_n(x) for x -> 0+ using the leading singular term.

    y_n(x) ~ -(2n-1)!! / x^(n+1)  for n >= 1, and y_0(x) ~ -1/x.

    x must be strictly positive for this to make sense; y_n is undefined
    (singular) at x=0.
    """
    if n == 0:
        return -1.0 / x
    # (2n-1)!!
    df = 1.0
    for k in range(1, n):
        df *= (2 * k + 1)
    # (2n-1)!! / x^(n+1) — note (2n-1)!! = product_{k=0}^{n-1} (2k+1)
    df2 = 1.0
    for k in range(n):
        df2 *= (2 * k + 1)
    return -df2 / (x ** (n + 1))


def spherical_jn(n, x):
    """Spherical Bessel function of the first kind, j_n(x).

    Parameters
    ----------
    n : int
        Order (n >= 0). Negative orders are not supported.
    x : float
        Argument (any real value, including zero and negatives).

    Returns
    -------
    float
        The value of j_n(x).

    Implementation
    --------------
    For |x| above a small cutoff we use upward recurrence
    j_n = ((2n-1)/x) j_{n-1} - j_{n-2}, seeded from j_0 = sin(x)/x and
    j_1 = sin(x)/x^2 - cos(x)/x. Below the cutoff we use the small-argument
    series, which avoids the cancellation that the recurrence suffers from
    when j_n is many orders of magnitude smaller than j_{n-1}.

    Edge cases
    ----------
    - n < 0 raises ValueError.
    - x = 0: j_0(0) = 1, j_n(0) = 0 for n >= 1.
    - Large n with moderate x: the recurrence is stable upward, but the
      value underflows to 0.0 for large enough n; this is correct.
    """
    if not isinstance(n, int):
        raise TypeError("n must be an int")
    if n < 0:
        raise ValueError("n must be >= 0")
    if not isinstance(x, (int, float)):
        raise TypeError("x must be a real number")
    x = float(x)

    if x == 0.0:
        return 1.0 if n == 0 else 0.0

    ax = abs(x)

    if ax < _small_x_threshold(n):
        return _small_argument_jn(n, x)

    # Upward recurrence for |x| not too small.
    sin_x = math.sin(x)
    cos_x = math.cos(x)
    inv_x = 1.0 / x
    j0 = sin_x * inv_x
    if n == 0:
        return j0
    j1 = sin_x * inv_x * inv_x - cos_x * inv_x
    if n == 1:
        return j1
    jm2 = j0
    jm1 = j1
    for k in range(2, n + 1):
        jk = (2 * k - 1) * inv_x * jm1 - jm2
        jm2 = jm1
        jm1 = jk
    return jm1


def spherical_yn(n, x):
    """Spherical Bessel function of the second kind, y_n(x).

    Parameters
    ----------
    n : int
        Order (n >= 0). Negative orders are not supported.
    x : float
        Argument. Must be nonzero; y_n(0) is singular.

    Returns
    -------
    float
        The value of y_n(x). For x < 0 the result uses the analytic
        continuation y_n(-|x|) = (-1)^{n+1} y_n(|x|).

    Implementation
    --------------
    Upward recurrence y_n = ((2n-1)/x) y_{n-1} - y_{n-2}, seeded from
    y_0 = -cos(x)/x and y_1 = -cos(x)/x^2 - sin(x)/x. For very small |x|
    we use the leading singular term instead to avoid loss of precision in
    the trig evaluation.

    Edge cases
    ----------
    - n < 0 raises ValueError.
    - x = 0 raises ValueError (y_n is singular at the origin).
    """
    if not isinstance(n, int):
        raise TypeError("n must be an int")
    if n < 0:
        raise ValueError("n must be >= 0")
    if not isinstance(x, (int, float)):
        raise TypeError("x must be a real number")
    x = float(x)

    if x == 0.0:
        raise ValueError("spherical_yn is singular at x = 0")

    sign = 1.0
    abs_x = x
    if x < 0:
        abs_x = -x
        # y_n(-x) = (-1)^{n+1} y_n(x)
        sign = -1.0 if (n % 2 == 0) else 1.0

    # For very small |x|, use the leading singular term.
    if abs_x < _small_x_threshold(n):
        val = _small_argument_yn(n, abs_x)
        return sign * val

    sin_x = math.sin(abs_x)
    cos_x = math.cos(abs_x)
    inv_x = 1.0 / abs_x
    y0 = -cos_x * inv_x
    if n == 0:
        return sign * y0
    y1 = -cos_x * inv_x * inv_x - sin_x * inv_x
    if n == 1:
        return sign * y1
    ym2 = y0
    ym1 = y1
    for k in range(2, n + 1):
        yk = (2 * k - 1) * inv_x * ym1 - ym2
        ym2 = ym1
        ym1 = yk
    return sign * ym1
