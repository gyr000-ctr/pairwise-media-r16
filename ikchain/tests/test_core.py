import math

import pytest

import ikchain

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

ROOT = (0.0, 0.0)
TARGET = (1.0, 1.0)
HINT = (0.0, -1.0)

CASES = [
    ("reach", lambda: abs(math.dist(ikchain.IkChain().solve(ROOT, TARGET, 1.0, 1.0, HINT)[-1], TARGET)) < 1e-9),
    ("len", lambda: len(ikchain.IkChain().solve(ROOT, TARGET, 1.0, 1.0, HINT)) == 2),
    ("unreachable", lambda: abs(math.dist(ikchain.IkChain().solve(ROOT, (10.0, 0.0), 1.0, 1.0, HINT)[-1], (10.0, 0.0))) <= 2.0),
    ("bad_len", lambda: _raises(lambda: ikchain.IkChain().solve(ROOT, TARGET, 0.0, 1.0, HINT))),
    ("bad_target", lambda: _raises(lambda: ikchain.IkChain().solve(ROOT, None, 1.0, 1.0, HINT))),
    ("type", lambda: isinstance(ikchain.IkChain().solve(ROOT, TARGET, 1.0, 1.0, HINT), list)),
    ("det", lambda: ikchain.IkChain().solve(ROOT, TARGET, 1.0, 1.0, HINT) == ikchain.IkChain().solve(ROOT, TARGET, 1.0, 1.0, HINT)),
    ("finite", lambda: all(math.isfinite(c) for p in ikchain.IkChain().solve(ROOT, TARGET, 1.0, 1.0, HINT) for c in p)),
    ("root_end", lambda: ikchain.IkChain().solve(ROOT, TARGET, 1.0, 1.0, HINT)[0] != ROOT),
    ("hint_effect", lambda: ikchain.IkChain().solve(ROOT, TARGET, 1.0, 1.0, (0.0, -1.0)) != ikchain.IkChain().solve(ROOT, TARGET, 1.0, 1.0, (0.0, 1.0))),
    ("nan_target", lambda: _raises(lambda: ikchain.IkChain().solve(ROOT, (float('nan'), 0.0), 1.0, 1.0, HINT))),
    ("no_mut", lambda: ikchain.IkChain().solve(ROOT, TARGET, 1.0, 1.0, HINT) is not None),
    ("zero_dist", lambda: isinstance(ikchain.IkChain().solve(ROOT, ROOT, 1.0, 1.0, HINT), list)),
    ("long_reach", lambda: abs(math.dist(ikchain.IkChain().solve(ROOT, (0.5, 0.0), 3.0, 4.0, HINT)[-1], (0.5, 0.0))) < 1e-9),
    ("joint_inside", lambda: all(math.dist(ROOT, p) <= 1.0 + 1.0 + 1e-6 for p in ikchain.IkChain().solve(ROOT, TARGET, 1.0, 1.0, HINT))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
