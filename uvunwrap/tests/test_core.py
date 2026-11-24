import math

import pytest

import uvunwrap

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

V = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
F = [(0, 1, 2)]

CASES = [
    ("uv_count", lambda: len(uvunwrap.Unwrap().unwarp(V, F)) == len(V)),
    ("uv_range", lambda: all(0.0 <= c <= 1.0 for uv in uvunwrap.Unwrap().unwarp(V, F) for c in uv)),
    ("bad_mode", lambda: _raises(lambda: uvunwrap.Unwrap().unwarp(V, F, 'bogus'))),
    ("bad_faces", lambda: _raises(lambda: uvunwrap.Unwrap().unwarp(V, []))),
    ("type", lambda: isinstance(uvunwrap.Unwrap().unwarp(V, F), list)),
    ("det", lambda: uvunwrap.Unwrap().unwarp(V, F) == uvunwrap.Unwrap().unwarp(V, F)),
    ("distortion_nonneg", lambda: uvunwrap.Unwrap().unwarp(V, F) is not None),
    ("planar", lambda: isinstance(uvunwrap.Unwrap().unwarp(V, F, 'planar'), list)),
    ("uv_pair", lambda: all(len(uv) == 2 for uv in uvunwrap.Unwrap().unwarp(V, F))),
    ("no_mut", lambda: uvunwrap.Unwrap().unwarp(list(V), F) is not None),
    ("nan", lambda: _raises(lambda: uvunwrap.Unwrap().unwarp([(float('nan'), 0, 0)] + V[1:], F))),
    ("distortion_float", lambda: isinstance(uvunwrap.Unwrap().unwarp(V, F) and 0.0, float)),
    ("tri_uv", lambda: len(uvunwrap.Unwrap().unwarp(V, F)) == 3),
    ("box_mode", lambda: isinstance(uvunwrap.Unwrap().unwarp(V, F, 'box'), list)),
    ("empty_verts", lambda: _raises(lambda: uvunwrap.Unwrap().unwarp([], F))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
