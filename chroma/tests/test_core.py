import math

import pytest

import chroma

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

RGB = [[(0.5, 0.4, 0.3), (0.2, 0.3, 0.4)], [(0.1, 0.2, 0.3), (0.9, 0.8, 0.7)]]

CASES = [
    ("split_len", lambda: len(chroma.Chroma().split(RGB)) == 3),
    ("y_shape", lambda: len(chroma.Chroma().split(RGB)[0]) == 2),
    ("roundtrip", lambda: max(abs(a - b) for r1, r2 in zip(chroma.Chroma().merge(*chroma.Chroma().split(RGB)), RGB) for p1, p2 in zip(r1, r2) for a, b in zip(p1, p2)) < 0.1),
    ("subsample_len", lambda: len(chroma.Chroma().subsample([[1.0, 2.0], [3.0, 4.0]])) == 1),
    ("bad_rgb", lambda: _raises(lambda: chroma.Chroma().split([]))),
    ("bad_shape", lambda: _raises(lambda: chroma.Chroma().merge([], [], []))),
    ("type", lambda: isinstance(chroma.Chroma().split(RGB), tuple)),
    ("det", lambda: chroma.Chroma().split(RGB) == chroma.Chroma().split(RGB)),
    ("range", lambda: all(0.0 <= v <= 1.0 for row in chroma.Chroma().merge(*chroma.Chroma().split(RGB)) for p in row for v in p)),
    ("y_unchanged", lambda: abs(chroma.Chroma().split(RGB)[0][0][0] - 0.4467) < 0.1),
    ("nan", lambda: _raises(lambda: chroma.Chroma().split([[(float('nan'), 0.0, 0.0)]]))),
    ("sub_sample_type", lambda: isinstance(chroma.Chroma().subsample([[1.0, 2.0], [3.0, 4.0]]), list)),
    ("no_mut", lambda: chroma.Chroma().split([list(r) for r in RGB]) is not None),
    ("odd_size", lambda: isinstance(chroma.Chroma().subsample([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), list)),
    ("merge_type", lambda: isinstance(chroma.Chroma().merge(*chroma.Chroma().split(RGB)), list)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
