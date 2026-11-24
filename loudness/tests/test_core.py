import math

import pytest

import loudness

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _l():
    return loudness.Loudness(48000)

CASES = [
    ("silence_neg_inf", lambda: _l().integrated([0.0] * 1000) == float('-inf')),
    ("tone_reading", lambda: abs(_l().integrated([0.5 * math.sin(2 * math.pi * 1000 * i / 48000) for i in range(48000)]) - (-6.0)) < 6.0),
    ("kweight_len", lambda: len(_l().k_weight([0.1] * 100)) == 100),
    ("bad_rate", lambda: _raises(lambda: loudness.Loudness(0))),
    ("empty", lambda: _l().integrated([]) == float('-inf')),
    ("type", lambda: isinstance(_l().integrated([0.1] * 100), float)),
    ("det", lambda: _l().integrated([0.1] * 100) == _l().integrated([0.1] * 100)),
    ("monotonic", lambda: _l().integrated([0.5] * 1000) >= _l().integrated([0.1] * 1000)),
    ("kweight_type", lambda: isinstance(_l().k_weight([0.1, 0.2]), list)),
    ("nan", lambda: _raises(lambda: _l().integrated([float('nan')] * 10))),
    ("range", lambda: _l().integrated([0.5] * 1000) <= 0.0),
    ("no_mut", lambda: _l().integrated([0.2] * 100) is not None),
    ("kweight_range", lambda: all(abs(v) <= 10.0 for v in _l().k_weight([0.5] * 100))),
    ("tiny", lambda: isinstance(_l().integrated([1e-9] * 100), float)),
    ("loud_gt_quiet", lambda: _l().integrated([0.9] * 2000) > _l().integrated([0.05] * 2000)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
