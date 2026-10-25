# tiny-spherical-bessel-functions

Evaluates spherical Bessel functions of the first kind (`spherical_jn`) and second kind (`spherical_yn`) for non-negative integer orders on plain Python floats.

## Usage

```python
from tiny_spherical_bessel_functions import spherical_jn, spherical_yn

# j_0(1.5) == sin(1.5)/1.5
print(spherical_jn(0, 1.5))

# y_0(1.5) == -cos(1.5)/1.5
print(spherical_yn(0, 1.5))

# Higher orders via recurrence
print(spherical_jn(3, 2.0))
print(spherical_yn(3, 2.0))
```

## Why this exists

The standard library does not ship spherical Bessel functions. SciPy does, but pulling in SciPy for two scalar functions is a heavy dependency. This library provides them with zero third-party requirements: pure Python, standard library only.

The trade-off is precision and range. The implementation uses upward recurrence on doubles, which is stable for the first-kind function `j_n` when `|x|` is not too small relative to `n`. Below a heuristic cutoff (`max(0.5, 0.3*n)`) the code switches to the small-argument series for `j_n` and the leading singular term for `y_n`. This avoids the catastrophic cancellation that plain forward recurrence suffers from when `j_n` is many orders of magnitude smaller than `j_{n-1}`.

## Awkward edges

- `y_n(0)` is singular; the function raises `ValueError` rather than returning `inf`.
- `j_n(0) = 1` for `n=0` and `0` for `n >= 1`.
- For negative arguments, `j_n(-x) = (-1)^n j_n(x)` and `y_n(-x) = (-1)^{n+1} y_n(x)` are returned directly.
- Negative orders are not supported; `n < 0` raises `ValueError`.
- For very large `n` at small `|x|`, `j_n` underflows to `0.0`. This is physically correct, not a bug.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

