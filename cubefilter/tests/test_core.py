import math

import pytest

import cubefilter

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

ENV = [[1.0] * 4 for _ in range(4)]

CASES = [
    ("constant", lambda: all(abs(v - 1.0) < 1e-3 for row in cubefilter.CubeFilter().convolve(ENV, 4)[0] for v in row)),
    ("faces", lambda: len(cubefilter.CubeFilter().convolve(ENV, 4)) == 6),
    ("bad_size", lambda: _raises(lambda: cubefilter.CubeFilter().convolve(ENV, 0))),
    ("bad_env", lambda: _raises(lambda: cubefilter.CubeFilter().convolve([], 4))),
    ("type", lambda: isinstance(cubefilter.CubeFilter().convolve(ENV, 4), list)),
    ("det", lambda: cubefilter.CubeFilter().convolve(ENV, 4) == cubefilter.CubeFilter().convolve(ENV, 4)),
    ("sample_range", lambda: 0.0 <= cubefilter.CubeFilter().sample(ENV, 0, 0.5, 0.5) <= 1.0),
    ("bad_face", lambda: _raises(lambda: cubefilter.CubeFilter().sample(ENV, 9, 0.5, 0.5))),
    ("shape", lambda: len(cubefilter.CubeFilter().convolve(ENV, 4)[0]) == 4),
    ("range", lambda: all(0.0 <= v <= 1.0 for f in cubefilter.CubeFilter().convolve(ENV, 4) for row in f for v in row)),
    ("nan", lambda: _raises(lambda: cubefilter.CubeFilter().sample(ENV, 0, float('nan'), 0.5))),
    ("no_mut", lambda: cubefilter.CubeFilter().convolve([r[:] for r in ENV], 4) is not None),
    ("sample_type", lambda: isinstance(cubefilter.CubeFilter().sample(ENV, 0, 0.5, 0.5), float)),
    ("sizes", lambda: len(cubefilter.CubeFilter().convolve(ENV, 8)) == 6),
    ("zero_ok", lambda: isinstance(cubefilter.CubeFilter().convolve([[0.0] * 4 for _ in range(4)], 4), list)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
