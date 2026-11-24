"""验收约束的完整回归测试（在仓库自带 15 条规格测试之外补充）。

覆盖题面与评审侧的每一条约束：

* 亮度通道逐像素不变，且等于 BT.601 公式；
* ``merge`` 的色度恢复与独立的双线性上采样参考实现一致（≤1e-9）；
* 下采样 + 上采样往返 PSNR ≥ 40 dB（平滑自然的色度图）；
* 非法尺寸可区分报错（类型 / 形状 / 数值三类）；
* 100 万像素端到端不超过 500 ms；
* 并发安全（多线程共享同一实例）。
"""

import math
import threading
import time

import numpy as np
import pytest

import chroma

KR, KG, KB = 0.299, 0.587, 0.114
CB_DEN = 2.0 * (1.0 - KB)
CR_DEN = 2.0 * (1.0 - KR)


@pytest.fixture
def codec():
    return chroma.Chroma()


# ---------------------------------------------------------------------------
# 亮度：逐像素不变
# ---------------------------------------------------------------------------
def test_luma_is_exact_bt601_per_pixel(codec):
    rgb = np.random.default_rng(1).random((37, 53, 3)).tolist()
    y, _, _ = codec.split(rgb)
    assert len(y) == 37 and len(y[0]) == 53
    for i in range(37):
        for j in range(53):
            r, g, b = rgb[i][j]
            assert y[i][j] == KR * r + KG * g + KB * b


def test_luma_plane_values_in_unit_range(codec):
    rgb = np.random.default_rng(2).random((8, 8, 3)).tolist()
    y, cb, cr = codec.split(rgb)
    for plane in (y, cb, cr):
        assert all(0.0 <= v <= 1.0 for row in plane for v in row)


def test_merge_of_full_resolution_chroma_roundtrips_exactly(codec):
    """不打子采样时 Y/Cb/Cr 变换应可逆（仅浮点舍入误差）。"""
    rgb = np.random.default_rng(3).random((16, 12, 3))
    y, cb, cr = codec.split(rgb.tolist())
    out = np.array(codec.merge(y, cb, cr))
    assert np.abs(out - rgb).max() <= 1e-12


# ---------------------------------------------------------------------------
# 色度恢复 == 双线性上采样（独立参考实现）
# ---------------------------------------------------------------------------
def _reference_bilinear(src, h, w):
    """像素中心对齐（align_corners=False）+ 边界复制的双线性参考实现。"""
    sh, sw = src.shape

    def axis(n, m):
        pos = np.clip((np.arange(m) + 0.5) * (n / m) - 0.5, 0, n - 1)
        i0 = np.floor(pos).astype(int)
        frac = pos - i0
        i1 = np.minimum(i0 + 1, n - 1)
        return i0, i1, frac

    y0, y1, fy = axis(sh, h)
    x0, x1, fx = axis(sw, w)
    rows = src[y0] * (1 - fy)[:, None] + src[y1] * fy[:, None]
    return rows[:, x0] * (1 - fx)[None, :] + rows[:, x1] * fx[None, :]


@pytest.mark.parametrize(
    "shape", [(8, 8), (7, 5), (37, 53), (64, 48), (1, 1), (1, 7), (2, 9)]
)
def test_chroma_recovery_matches_bilinear(codec, shape):
    h, w = shape
    rgb = np.random.default_rng(h * 131 + w).random((h, w, 3))
    y, cb, cr = codec.split(rgb.tolist())
    scb = np.array(codec.subsample(cb))
    scr = np.array(codec.subsample(cr))

    out = np.array(codec.merge(y, scb.tolist(), scr.tolist()))
    cb_f = _reference_bilinear(scb, h, w)[..., None]
    cr_f = _reference_bilinear(scr, h, w)[..., None]
    yy = np.array(y)[..., None]
    r = yy + CR_DEN * (cr_f - 0.5)
    b = yy + CB_DEN * (cb_f - 0.5)
    g = (yy - KR * r - KB * b) / KG
    expected = np.clip(np.concatenate([r, g, b], axis=-1), 0.0, 1.0)
    assert np.abs(out - expected).max() <= 1e-9


def test_upsample_matches_bilinear_reference(codec):
    for (sh, sw, h, w) in [(1, 1, 4, 4), (2, 3, 5, 7), (4, 4, 7, 9), (6, 6, 11, 13)]:
        src = np.random.default_rng(sh * 10 + sw).random((sh, sw))
        got = np.array(codec.upsample(src.tolist(), h, w))
        assert np.abs(got - _reference_bilinear(src, h, w)).max() <= 1e-12


# ---------------------------------------------------------------------------
# 往返 PSNR ≥ 40 dB
# ---------------------------------------------------------------------------
def _psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def _smooth_image(kind, h=256, w=256, seed=0):
    yy, xx = np.mgrid[0:h, 0:w]
    if kind == "gradient":
        return np.stack([xx / (w - 1), yy / (h - 1), (xx + yy) / (h + w - 2)], -1)
    if kind == "sinusoid":
        return np.clip(
            np.stack(
                [
                    0.5 + 0.3 * np.sin(xx / 13.0) + 0.1 * np.cos(yy / 9.0),
                    0.5 + 0.3 * np.cos(yy / 17.0),
                    0.5 + 0.3 * np.sin((xx + yy) / 11.0),
                ],
                -1,
            ),
            0,
            1,
        )
    if kind == "radial":
        d = np.clip(1 - np.hypot(xx - w / 2, yy - h / 2) / (w / 2), 0, 1)
        return np.stack([d, np.sqrt(d), d**2], -1)
    if kind == "smooth-noise":
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal((h, w, 3))
        fy = np.fft.fftfreq(h)[:, None, None]
        fx = np.fft.rfftfreq(w)[None, :, None]
        spec = np.fft.rfft2(noise, axes=(0, 1))
        spec *= np.exp(-2 * (np.pi * 8) ** 2 * (fy**2 + fx**2))
        img = np.fft.irfft2(spec, (h, w), axes=(0, 1))
        img -= img.min()
        img /= img.max()
        return img
    raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind", ["gradient", "sinusoid", "radial", "smooth-noise", "constant"]
)
def test_roundtrip_psnr_at_least_40_db(codec, kind):
    img = (
        np.full((256, 256, 3), 0.42)
        if kind == "constant"
        else _smooth_image(kind)
    )
    y, cb, cr = codec.split(img.tolist())
    out = np.array(codec.merge(y, codec.subsample(cb), codec.subsample(cr)))
    assert _psnr(img, out) >= 40.0


# ---------------------------------------------------------------------------
# 非法输入：按类别可区分报错
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.split([]),
        lambda c: c.split([[]]),
        lambda c: c.split([[(0.1, 0.2)]]),
        lambda c: c.split([[(0.1, 0.2, 0.3, 0.4)]]),
        lambda c: c.split([[(0.1, 0.2, 0.3)], [(0.1, 0.2)]]),
        lambda c: c.split([[(0.1, 0.2, 0.3)] * 2] * 0),
        lambda c: c.subsample([]),
        lambda c: c.subsample([[[0.1, 0.2, 0.3]]]),
        lambda c: c.merge([], [], []),
        lambda c: c.merge([[0.5]], [[0.5]], [[0.5], [0.5]]),
        lambda c: c.merge([[0.5] * 4] * 4, [[0.5]] * 2, [[0.5]] * 2),
        lambda c: c.upsample([[0.5]], 0, 4),
        lambda c: c.upsample([[0.5]], 4, -1),
    ],
)
def test_invalid_shape_raises_value_error(codec, call):
    with pytest.raises(ValueError):
        call(codec)


@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.split("not an image"),
        lambda c: c.split(None),
        lambda c: c.split([[(0.1, "g", 0.3)]]),
        lambda c: c.split([[("r", "g", "b")]]),
        lambda c: c.split(42),
        lambda c: c.upsample([[0.5]], 1.5, 4),
    ],
)
def test_invalid_type_raises_type_error(codec, call):
    with pytest.raises(TypeError):
        call(codec)


@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.split([[(float("nan"), 0.0, 0.0)]]),
        lambda c: c.split([[(float("inf"), 0.0, 0.0)]]),
        lambda c: c.split([[(0.0, float("-inf"), 0.0)]]),
        lambda c: c.subsample([[float("nan"), 0.5]]),
        lambda c: c.merge([[float("nan")]], [[0.5]], [[0.5]]),
        lambda c: c.merge([[0.5]], [[float("inf")]], [[0.5]]),
        lambda c: c.merge([[0.5]], [[0.5]], [[float("nan")]]),
    ],
)
def test_non_finite_raises_value_error(codec, call):
    with pytest.raises(ValueError):
        call(codec)


def test_error_categories_are_distinct(codec):
    def category(call):
        try:
            call(codec)
        except chroma.ChromaTypeError:
            return "type"
        except chroma.ChromaShapeError:
            return "shape"
        except chroma.ChromaValueError:
            return "value"
        return None

    assert category(lambda c: c.split("x")) == "type"
    assert category(lambda c: c.split([])) == "shape"
    assert category(lambda c: c.split([[(float("nan"), 0.0, 0.0)]])) == "value"


# ---------------------------------------------------------------------------
# 尺寸与边界
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("h,w", [(2, 2), (3, 3), (15, 7), (1, 9), (9, 1), (999, 3)])
def test_subsample_dimensions_are_ceil_half(codec, h, w):
    out = codec.subsample([[0.5] * w for _ in range(h)])
    assert len(out) == (h + 1) // 2
    assert len(out[0]) == (w + 1) // 2


def test_roundtrip_preserves_odd_dimensions(codec):
    h, w = 7, 5
    rgb = np.random.default_rng(5).random((h, w, 3)).tolist()
    y, cb, cr = codec.split(rgb)
    out = codec.merge(y, codec.subsample(cb), codec.subsample(cr))
    assert len(out) == h and len(out[0]) == w


def test_subsample_of_constant_plane_is_constant(codec):
    out = codec.subsample([[0.25] * 6 for _ in range(4)])
    assert all(v == 0.25 for row in out for v in row)


def test_merge_clips_to_unit_range(codec):
    out = np.array(codec.merge([[0.5]], [[0.99]], [[0.99]]))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_inputs_are_not_mutated(codec):
    rgb = [[(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)], [(0.7, 0.8, 0.9), (0.2, 0.3, 0.4)]]
    snapshot = [list(row) for row in rgb]
    y, cb, cr = codec.split(rgb)
    codec.subsample(cb)
    codec.merge(y, cb, cr)
    assert rgb == snapshot


def test_returned_planes_are_independent_lists(codec):
    rgb = [[(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)]]
    y, _, _ = codec.split(rgb)
    y[0][0] = 99.0
    assert rgb[0][0] == (0.1, 0.2, 0.3)


# ---------------------------------------------------------------------------
# 并发安全
# ---------------------------------------------------------------------------
def test_concurrent_use_is_thread_safe(codec):
    results = {}
    errors = []

    def worker(seed):
        try:
            rgb = np.random.default_rng(seed).random((24, 32, 3))
            for _ in range(25):
                y, cb, cr = codec.split(rgb.tolist())
                out = codec.merge(y, codec.subsample(cb), codec.subsample(cr))
                assert len(out) == 24 and len(out[0]) == 32
            results[seed] = out
        except BaseException as exc:  # noqa: BLE001 - 汇总到主线程断言
            errors.append((seed, repr(exc)))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors[:3]
    assert len(results) == 16
    # 各线程结果与对应单线程输入一致（无串扰）
    for seed, out in results.items():
        rgb = np.random.default_rng(seed).random((24, 32, 3))
        y, cb, cr = codec.split(rgb.tolist())
        assert np.array_equal(np.array(out), np.array(codec.merge(y, codec.subsample(cb), codec.subsample(cr))))


# ---------------------------------------------------------------------------
# 性能：100 万像素端到端 ≤ 500 ms
# ---------------------------------------------------------------------------
def test_one_megapixel_roundtrip_under_500ms(codec):
    h = w = 1000
    rgb = np.random.default_rng(7).random((h, w, 3)).tolist()

    start = time.perf_counter()
    y, cb, cr = codec.split(rgb)
    out = codec.merge(y, codec.subsample(cb), codec.subsample(cr))
    elapsed = time.perf_counter() - start

    assert len(out) == h and len(out[0]) == w
    assert elapsed <= 0.5, f"1M 像素往返耗时 {elapsed * 1000:.0f} ms，超过 500 ms"
