"""chroma —— 4:2:0 色度子采样核心 API。

设计要点
--------
* 输入/输出均为 ``[[(r, g, b), ...], ...]`` 形式的嵌套 Python 列表，
  颜色分量为浮点数（通常位于 0..1）。
* 颜色转换采用 BT.601 全范围（JFIF）矩阵::

      Y  =  0.299 R + 0.587 G + 0.114 B
      Cb = (B - Y) / (2 (1 - Kb)) + 0.5
      Cr = (R - Y) / (2 (1 - Kr)) + 0.5

  该矩阵可逆，因此 ``merge(split(rgb))`` 在无量化时逐像素精确
  （误差仅在浮点舍入量级）。
* 亮度 Y 以浮点全精度保存，逐像素等于上式；色度 Cb/Cr 位于 ``[0, 1]``。
* 下采样在 2x2 块上做盒式平均（4:2:0 的标准做法；奇数边以边缘复制补齐，
  输出尺寸为 ``ceil(H/2) x ceil(W/2)``）。上采样为像素中心对齐
  （align_corners=False）的双线性插值，边界复制。二者构成标准
  低通/重建对，``merge`` 对半分辨率色度所做恢复与独立双线性上采样
  逐元素一致。
* 仅依赖 NumPy；不修改输入、无全局可变状态，实例可在多线程间共享，
  所有方法均可并发调用。
"""

from __future__ import annotations

import itertools

import numpy as np

__all__ = [
    "Chroma",
    "ChromaError",
    "ChromaTypeError",
    "ChromaShapeError",
    "ChromaValueError",
    "split",
    "merge",
    "subsample",
    "upsample",
]

# BT.601 / JFIF 系数
_KR = 0.299
_KG = 0.587
_KB = 0.114
_CB_DEN = 2.0 * (1.0 - _KB)  # 1.772
_CR_DEN = 2.0 * (1.0 - _KR)  # 1.402


# ---------------------------------------------------------------------------
# 异常：非法输入按类别可区分
# ---------------------------------------------------------------------------
class ChromaError(Exception):
    """所有 chroma 错误的基类。"""


class ChromaTypeError(ChromaError, TypeError, ValueError):
    """输入类型不受支持（非数值、非序列、参差不齐等）。

    同时继承 ``TypeError`` 与 ``ValueError``：既可用 ``except TypeError``
    精确捕获「类型不对」，也兼容把非法输入统一按 ``ValueError`` 处理的调用方。
    """


class ChromaShapeError(ChromaError, ValueError):
    """输入形状/尺寸不合法（空、维度错误、通道数非 3、平面尺寸不匹配等）。"""


class ChromaValueError(ChromaError, ValueError):
    """输入数值非法（NaN 或无穷）。"""


# ---------------------------------------------------------------------------
# 输入规范化（性能关键路径：避免 np.asarray 对嵌套列表的慢速逐元素处理）
#
# 嵌套 Python 列表 -> float64 数组使用 np.fromiter 在 C 层摊平；结构检查
# 全部基于 C 层 map/all 或 O(行数) 的行宽扫描，因此百万像素路径上没有
# 逐像素的解释器循环。
# ---------------------------------------------------------------------------
_SEQ = (list, tuple, np.ndarray)
_THREE = 3  # 供 map 绑定方法，避免逐元素创建 int


def _flat_stream(rows, depth):
    """返回把嵌套序列摊平成数值迭代器的 C 层迭代器。"""
    if depth == 1:
        return itertools.chain.from_iterable(rows)
    return itertools.chain.from_iterable(itertools.chain.from_iterable(rows))


def _from_nested(rows, depth, count, name):
    """把嵌套序列摊平为 float64 数组；元素类型/数量异常时报可区分错误。

    使用 ``count`` 预分配以提速；随后探测流是否恰好耗尽，避免把超长行
    （例如 4 通道像素）静默截断。
    """
    stream = _flat_stream(rows, depth)
    try:
        flat = np.fromiter(stream, dtype=np.float64, count=count)
    except TypeError as exc:
        # 例如标量出现在需要像素的位置
        raise ChromaTypeError(f"{name} 含有无法转换为数值的元素：{exc}") from None
    except ValueError as exc:
        # 元素不足 / 元素无法转换为数值：统一在下方按总数与实际内容再分类
        if "too short" in str(exc):
            raise ChromaShapeError(
                f"{name} 的元素总数少于声明的形状 {count}"
            ) from None
        raise ChromaTypeError(f"{name} 含有无法转换为数值的元素：{exc}") from None
    _END = object()
    extra = next(stream, _END)
    if extra is not _END:
        raise ChromaShapeError(f"{name} 的元素总数超过声明的形状 {count}")
    if flat.size != count:
        raise ChromaShapeError(
            f"{name} 的元素总数与声明的形状不一致（期望 {count}，实际 {flat.size}）"
        )
    return flat


def _diagnose_3d(rows, n_cols, name):
    """结构尺寸不符时，逐像素定位并抛出分类明确的异常（仅错误路径）。"""
    for r in rows:
        if not isinstance(r, _SEQ):
            raise ChromaTypeError(f"{name} 的行必须是序列")
        if len(r) != n_cols:
            raise ChromaShapeError(f"{name} 的行宽不一致")
        for pix in r:
            if not isinstance(pix, _SEQ):
                raise ChromaTypeError(f"{name} 的像素必须是 (r, g, b) 序列")
            if len(pix) != 3:
                raise ChromaShapeError(
                    f"{name} 的通道数必须为 3，实际出现 {len(pix)}"
                )
    raise ChromaShapeError(f"{name} 的元素总数与声明的形状不一致")


def _diagnose_2d(rows, n_cols, name):
    """行尺寸不符时定位错误（仅错误路径）。"""
    for r in rows:
        if not isinstance(r, _SEQ):
            raise ChromaTypeError(f"{name} 的行必须是数值序列")
        if len(r) != n_cols:
            raise ChromaShapeError(f"{name} 的行宽不一致")
        if r and isinstance(r[0], _SEQ):
            raise ChromaShapeError(f"{name} 必须是二维平面（HxW）")
    raise ChromaShapeError(f"{name} 的元素总数与声明的形状不一致")


def _array_from(obj, ndim, name):
    """把输入规范化为连续 float64 数组，并做类型/形状检查（不检查数值）。"""
    # NumPy 数组走快路径，可零拷贝
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind not in "iuf":
            raise ChromaTypeError(f"{name} 的元素必须是数值，实际 dtype={obj.dtype}")
        if obj.ndim != ndim:
            raise ChromaShapeError(f"{name} 必须是 {ndim} 维，实际 {obj.ndim} 维")
        if obj.size == 0:
            raise ChromaShapeError(f"{name} 不能为空")
        if ndim == 3 and obj.shape[2] != 3:
            raise ChromaShapeError(f"{name} 的通道数必须为 3，实际 {obj.shape[2]}")
        return np.ascontiguousarray(obj, dtype=np.float64)

    if not isinstance(obj, _SEQ):
        raise ChromaTypeError(
            f"{name} 必须是数值嵌套序列，实际为 {type(obj).__name__}"
        )

    rows = list(obj)
    n_rows = len(rows)
    if n_rows == 0:
        raise ChromaShapeError(f"{name} 不能为空")

    first = rows[0]
    if not isinstance(first, _SEQ):
        raise ChromaTypeError(f"{name} 的行必须是序列")
    n_cols = len(first)
    if n_cols == 0:
        raise ChromaShapeError(f"{name} 的行不能为空")

    if ndim == 2:
        if isinstance(first[0], _SEQ):
            raise ChromaShapeError(f"{name} 必须是二维平面（HxW），收到更高维输入")
        count = n_rows * n_cols
        if not all(isinstance(r, _SEQ) and len(r) == n_cols for r in rows):
            _diagnose_2d(rows, n_cols, name)
        flat = _from_nested(rows, 1, count, name)
        return flat.reshape(n_rows, n_cols)

    if ndim == 3:
        pix0 = first[0]
        if not isinstance(pix0, _SEQ):
            raise ChromaShapeError(f"{name} 必须是三维 RGB（HxWx3），收到更低维输入")
        if len(pix0) != 3:
            raise ChromaShapeError(f"{name} 的通道数必须为 3，实际 {len(pix0)}")
        count = n_rows * n_cols * 3
        if not all(isinstance(r, _SEQ) and len(r) == n_cols for r in rows):
            _diagnose_3d(rows, n_cols, name)
        # 逐像素宽度校验：否则不同宽度可能「互相抵消」使元素总数恰好正确，
        # 从而静默产出错误结果。all(map(len==3)) 全在 C 层完成，约 20 ms/Mpx。
        try:
            widths_ok = all(
                map(_THREE.__eq__, map(len, itertools.chain.from_iterable(rows)))
            )
        except TypeError:
            # 像素位置出现标量等不可 len() 的对象
            _diagnose_3d(rows, n_cols, name)
            raise AssertionError("unreachable")
        if not widths_ok:
            _diagnose_3d(rows, n_cols, name)
        try:
            return _from_nested(rows, 2, count, name).reshape(n_rows, n_cols, 3)
        except ChromaShapeError:
            _diagnose_3d(rows, n_cols, name)
            raise

    raise AssertionError("ndim 只能是 2 或 3")


def _check_finite(arr, name):
    if not np.isfinite(arr).all():
        bad = int(arr.size - np.count_nonzero(np.isfinite(arr)))
        raise ChromaValueError(f"{name} 含 {bad} 个非有限值（NaN/Inf）")


# ---------------------------------------------------------------------------
# 纯函数核心：split / down / up / merge
# ---------------------------------------------------------------------------
def _split_planes(rgb):
    """RGB(HxWx3) -> (Y, Cb, Cr) 三个 HxW float64 平面。"""
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    y = _KR * r + _KG * g + _KB * b
    cb = 0.5 + (b - y) / _CB_DEN
    cr = 0.5 + (r - y) / _CR_DEN
    return y, cb, cr


def _merge_planes(y, cb, cr):
    """(Y, Cb, Cr) -> RGB(HxWx3)，按 BT.601 逆矩阵重建并裁到 [0, 1]。"""
    r = y + _CR_DEN * (cr - 0.5)
    b = y + _CB_DEN * (cb - 0.5)
    g = (y - _KR * r - _KB * b) / _KG
    rgb = np.empty(y.shape + (3,), dtype=np.float64)
    rgb[..., 0] = r
    rgb[..., 1] = g
    rgb[..., 2] = b
    np.clip(rgb, 0.0, 1.0, out=rgb)
    return rgb


def _downsample_plane(plane):
    """HxW -> ceil(H/2)xceil(W/2)，2x2 盒式平均；奇数边复制边界像素。"""
    h, w = plane.shape
    hc, wc = (h + 1) // 2, (w + 1) // 2
    ph, pw = 2 * hc, 2 * wc
    if ph != h or pw != w:
        plane = np.pad(plane, ((0, ph - h), (0, pw - w)), mode="edge")
    return plane.reshape(hc, 2, wc, 2).mean(axis=(1, 3))


def _upsample_axis(src, dst_len, axis):
    """沿单轴做像素中心对齐的双线性插值（边界复制）。"""
    src_len = src.shape[axis]
    if dst_len == src_len:
        return src
    pos = (np.arange(dst_len, dtype=np.float64) + 0.5) * (src_len / dst_len) - 0.5
    np.clip(pos, 0.0, src_len - 1, out=pos)
    i0 = np.floor(pos)
    frac = pos - i0
    i0 = i0.astype(np.intp)
    i1 = np.minimum(i0 + 1, src_len - 1)
    shape = [1, 1]
    shape[axis] = dst_len
    frac = frac.reshape(shape)
    a = np.take(src, i0, axis=axis)
    b = np.take(src, i1, axis=axis)
    return a + (b - a) * frac


def _upsample_plane(src, h, w):
    """ceil(H/2)xceil(W/2) -> HxW，可分离双线性插值（先行后列）。"""
    out = _upsample_axis(src, h, 0)
    return _upsample_axis(out, w, 1)


def _nested_plane(plane):
    """HxW 数组 -> 嵌套列表（逐行转换更快；解包 map 结果少一次 list() 调用）。"""
    return [*map(np.ndarray.tolist, plane)]


def _nested_rgb(rgb):
    """HxWx3 数组 -> [[(r, g, b), ...], ...]。

    逐行 ``tolist()`` 比整体三维 ``tolist()`` 更快（行内连续，单次 C 循环）。
    """
    return [*map(np.ndarray.tolist, rgb)]


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------
class Chroma:
    """4:2:0 色度子采样编解码器。

    该类无可变状态，所有方法均为纯函数式，实例可在多线程间共享；
    也可直接调用模块级函数 :func:`split` / :func:`merge` /
    :func:`subsample` / :func:`upsample`。
    """

    __slots__ = ()

    def split(self, rgb):
        """把 RGB 图像拆成 ``(y, cb, cr)`` 三个亮度/色度平面。

        ``rgb`` 形如 ``[[(r, g, b), ...], ...]``（H 行、W 列）。返回值为
        ``(y, cb, cr)`` 三元组，每个平面均为 HxW 的嵌套列表；亮度平面
        的每个元素精确等于 ``0.299r + 0.587g + 0.114b``。
        """
        arr = _array_from(rgb, 3, "rgb")
        _check_finite(arr, "rgb")
        y, cb, cr = _split_planes(arr)
        return (_nested_plane(y), _nested_plane(cb), _nested_plane(cr))

    def merge(self, y, cb, cr):
        """把 ``(y, cb, cr)`` 重建为 RGB 嵌套列表。

        ``cb``/``cr`` 可为与 ``y`` 同尺寸的全分辨率平面，或
        :meth:`subsample` 产生的 ``ceil(H/2) x ceil(W/2)`` 半分辨率平面；
        后者用与 :meth:`upsample` 完全相同的双线性插值恢复，再按 BT.601
        逆矩阵重建。结果分量裁剪到 ``[0, 1]``。其它尺寸视为非法并报
        :class:`ChromaShapeError`。
        """
        ya = _array_from(y, 2, "y")
        cba = _array_from(cb, 2, "cb")
        cra = _array_from(cr, 2, "cr")
        _check_finite(ya, "y")
        _check_finite(cba, "cb")
        _check_finite(cra, "cr")
        h, w = ya.shape
        # 允许两种半分辨率约定：ceil(H/2)（本库 downsample 的输出）
        # 与 floor 版本（与常见库的 4:2:0 约定一致）；再小视为非法。
        half = ((h + 1) // 2, (w + 1) // 2)
        half_floor = (max(1, h // 2), max(1, w // 2))
        if cba.shape != (h, w):
            if cba.shape != half and cba.shape != half_floor:
                raise ChromaShapeError(
                    f"cb 的尺寸 {cba.shape} 与亮度 {ya.shape} 不匹配"
                    f"（应为 {ya.shape} 或半分辨率 {half}）"
                )
            cba = _upsample_plane(cba, h, w)
        if cra.shape != (h, w):
            if cra.shape != half and cra.shape != half_floor:
                raise ChromaShapeError(
                    f"cr 的尺寸 {cra.shape} 与亮度 {ya.shape} 不匹配"
                    f"（应为 {ya.shape} 或半分辨率 {half}）"
                )
            cra = _upsample_plane(cra, h, w)
        return _nested_rgb(_merge_planes(ya, cba, cra))

    def subsample(self, c):
        """对单个色度平面做 4:2:0 下采样。

        输入为 HxW 嵌套列表，输出为 ``ceil(H/2) x ceil(W/2)`` 的嵌套列表，
        每个元素是 2x2 邻域的平均值（奇数边复制边界像素）。
        """
        arr = _array_from(c, 2, "c")
        _check_finite(arr, "c")
        return _nested_plane(_downsample_plane(arr))

    def upsample(self, c, h, w):
        """把半分辨率色度平面按双线性插值放大到 ``(h, w)``。

        采用像素中心对齐（align_corners=False）语义，边界复制。
        """
        arr = _array_from(c, 2, "c")
        _check_finite(arr, "c")
        if isinstance(h, bool) or isinstance(w, bool) or not isinstance(
            h, (int, np.integer)
        ) or not isinstance(w, (int, np.integer)):
            raise ChromaTypeError("h/w 必须为整数")
        h = int(h)
        w = int(w)
        if h <= 0 or w <= 0:
            raise ChromaShapeError("h/w 必须为正整数")
        return _nested_plane(_upsample_plane(arr, h, w))


# 模块级便捷函数（与 Chroma 方法语义一致）
def split(rgb):
    """见 :meth:`Chroma.split`。"""
    return Chroma().split(rgb)


def merge(y, cb, cr):
    """见 :meth:`Chroma.merge`。"""
    return Chroma().merge(y, cb, cr)


def subsample(c):
    """见 :meth:`Chroma.subsample`。"""
    return Chroma().subsample(c)


def upsample(c, h, w):
    """见 :meth:`Chroma.upsample`。"""
    return Chroma().upsample(c, h, w)
