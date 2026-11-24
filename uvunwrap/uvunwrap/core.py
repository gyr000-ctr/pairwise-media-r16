"""uvunwrap 核心 API：盒式（box）/平面（planar）UV 展开。

约束实现要点：
- 展开后 UV 落在 [0, 1]（保持宽高比的均匀缩放 + 平移归一化）；
- 面积失真按 ``distortion()`` 报告（逐三角形 UV/3D 面积比相对全局比例的
  最大偏差，均匀缩放不影响该指标）；
- 盒式展开按面法线主轴分为 ≤6 组（±X/±Y/±Z），接缝组数最少；
- 非法模式 / 非有限输入 / 空网格 / 非法拓扑分别抛出可区分的异常类型
  （均为 ``ValueError`` 子类）；
- 计算全部在局部变量上进行，实例状态仅在最后一次性写入（加锁），并发安全。
"""

from __future__ import annotations

import math
import numbers
import threading

__all__ = [
    "Unwrap",
    "UnwrapError",
    "InvalidModeError",
    "NonFiniteInputError",
    "EmptyMeshError",
    "InvalidTopologyError",
]

_MODES = ("box", "planar")

_AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


class UnwrapError(ValueError):
    """uvunwrap 错误基类（可按 ValueError 捕获）。"""


class InvalidModeError(UnwrapError):
    """非法展开模式。"""


class NonFiniteInputError(UnwrapError):
    """输入包含 NaN/Inf 等非有限数值。"""


class EmptyMeshError(UnwrapError):
    """顶点或面序列为空。"""


class InvalidTopologyError(UnwrapError):
    """面索引非法（非整数、越界、形状错误等）。"""


def _cross(ax, ay, az, bx, by, bz):
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


class Unwrap:
    """盒式/平面 UV 展开器。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._distortion = 0.0

    # ------------------------------------------------------------------ API

    def unwarp(self, verts, faces, mode="box"):
        """返回每顶点 UV（``list[(u, v)]``，各分量落在 [0, 1]）。"""
        if mode not in _MODES:
            raise InvalidModeError(
                "未知展开模式: {!r}（支持: {}）".format(mode, ", ".join(_MODES))
            )
        vs = self._validate_verts(verts)
        fs = self._validate_faces(faces, len(vs))

        # 每个面的未归一化法线（模长 = 2 倍面积）。
        crosses = []
        for i, j, k in fs:
            p0, p1, p2 = vs[i], vs[j], vs[k]
            crosses.append(
                _cross(
                    p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2],
                    p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2],
                )
            )

        if mode == "planar":
            uv_raw = self._project_planar(vs, crosses)
        else:
            uv_raw = self._project_box(vs, fs, crosses)

        uvs = self._normalize(uv_raw)
        dist = self._area_distortion(fs, crosses, uv_raw)

        with self._lock:
            self._distortion = dist
        return uvs

    def distortion(self):
        """最近一次展开的面积失真（0.0 表示无失真）。"""
        with self._lock:
            return self._distortion

    # ------------------------------------------------------------- 校验

    @staticmethod
    def _validate_verts(verts):
        try:
            vs = list(verts)
        except TypeError:
            raise EmptyMeshError("顶点序列不可迭代或为空") from None
        if not vs:
            raise EmptyMeshError("顶点序列为空")
        out = []
        for v in vs:
            try:
                x, y, z = v
            except (TypeError, ValueError):
                raise InvalidTopologyError(
                    "顶点必须是三分量序列: {!r}".format(v)
                ) from None
            coords = []
            for c in (x, y, z):
                if isinstance(c, bool) or not isinstance(c, numbers.Real):
                    raise InvalidTopologyError(
                        "顶点坐标必须是数值: {!r}".format(c)
                    )
                fc = float(c)
                if not math.isfinite(fc):
                    raise NonFiniteInputError(
                        "顶点坐标包含非有限值: {!r}".format(c)
                    )
                coords.append(fc)
            out.append((coords[0], coords[1], coords[2]))
        return out

    @staticmethod
    def _validate_faces(faces, nverts):
        try:
            fs = list(faces)
        except TypeError:
            raise EmptyMeshError("面序列不可迭代或为空") from None
        if not fs:
            raise EmptyMeshError("面序列为空")
        out = []
        for f in fs:
            try:
                i, j, k = f
            except (TypeError, ValueError):
                raise InvalidTopologyError(
                    "面必须是三元索引: {!r}".format(f)
                ) from None
            tri = []
            for idx in (i, j, k):
                if isinstance(idx, bool) or not isinstance(idx, numbers.Integral):
                    raise InvalidTopologyError(
                        "面索引必须是整数: {!r}".format(idx)
                    )
                idx = int(idx)
                if idx < 0 or idx >= nverts:
                    raise InvalidTopologyError(
                        "面索引越界: {}（顶点数 {}）".format(idx, nverts)
                    )
                tri.append(idx)
            out.append((tri[0], tri[1], tri[2]))
        return out

    # ------------------------------------------------------------- 投影

    @staticmethod
    def _project_planar(vs, crosses):
        # 面积加权平均法线作为最佳拟合平面法线。
        nx = ny = nz = 0.0
        for cx, cy, cz in crosses:
            nx += cx
            ny += cy
            nz += cz
        nl = math.sqrt(nx * nx + ny * ny + nz * nz)
        if nl <= 0.0:
            # 退化（如闭合对称网格）：选跨度最小的轴作为法线。
            extents = []
            for a in range(3):
                col = [p[a] for p in vs]
                extents.append(max(col) - min(col))
            normal = _AXES[extents.index(min(extents))]
        else:
            normal = (nx / nl, ny / nl, nz / nl)

        # 由法线构造正交基（选与法线最不平行的坐标轴做叉积，数值稳定）。
        axis = min(range(3), key=lambda a: abs(normal[a]))
        e = _AXES[axis]
        ux, uy, uz = _cross(*normal, *e)
        ul = math.sqrt(ux * ux + uy * uy + uz * uz)
        ux, uy, uz = ux / ul, uy / ul, uz / ul
        wx, wy, wz = _cross(*normal, ux, uy, uz)

        return [
            (p[0] * ux + p[1] * uy + p[2] * uz,
             p[0] * wx + p[1] * wy + p[2] * wz)
            for p in vs
        ]

    @staticmethod
    def _project_box(vs, fs, crosses):
        # 每个面按法线主轴归入 6 组之一：(axis, sign)。
        face_group = []
        face_area2 = []
        for cx, cy, cz in crosses:
            area2 = math.sqrt(cx * cx + cy * cy + cz * cz)
            axis = max(range(3), key=lambda a: abs((cx, cy, cz)[a]))
            sign = 1 if (cx, cy, cz)[axis] >= 0.0 else -1
            face_group.append((axis, sign))
            face_area2.append(area2)

        # 每个顶点归入其最大面积邻面的组（确定性：平局取先出现的面）。
        vgroup = [None] * len(vs)
        vbest = [-1.0] * len(vs)
        for fi, (i, j, k) in enumerate(fs):
            g = face_group[fi]
            w = face_area2[fi]
            for v in (i, j, k):
                if w > vbest[v]:
                    vbest[v] = w
                    vgroup[v] = g

        uv_raw = []
        for v, p in enumerate(vs):
            g = vgroup[v]
            if g is None:
                g = (2, 1)  # 未被任何面引用的孤立顶点：投影到 XY 平面
            axis, sign = g
            if axis == 0:
                u, w = p[1], p[2]
            elif axis == 1:
                u, w = p[0], p[2]
            else:
                u, w = p[0], p[1]
            if sign < 0:
                u = -u
            uv_raw.append((u, w))
        return uv_raw

    # ------------------------------------------------------------- 后处理

    @staticmethod
    def _normalize(uv_raw):
        us = [uv[0] for uv in uv_raw]
        ws = [uv[1] for uv in uv_raw]
        umin, umax = min(us), max(us)
        wmin, wmax = min(ws), max(ws)
        scale = max(umax - umin, wmax - wmin)
        if scale <= 0.0:
            return [(0.0, 0.0)] * len(uv_raw)
        return [((u - umin) / scale, (w - wmin) / scale) for u, w in uv_raw]

    @staticmethod
    def _area_distortion(fs, crosses, uv_raw):
        # 均匀缩放不影响比值，直接用归一化前的 UV 计算。
        a3d = []
        auv = []
        for (i, j, k), (cx, cy, cz) in zip(fs, crosses):
            a3d.append(0.5 * math.sqrt(cx * cx + cy * cy + cz * cz))
            u0, w0 = uv_raw[i]
            u1, w1 = uv_raw[j]
            u2, w2 = uv_raw[k]
            auv.append(0.5 * abs((u1 - u0) * (w2 - w0) - (u2 - u0) * (w1 - w0)))
        total3d = sum(a3d)
        totaluv = sum(auv)
        if total3d <= 0.0:
            return 0.0
        if totaluv <= 0.0:
            return 1.0
        scale = totaluv / total3d
        worst = 0.0
        for a, b in zip(a3d, auv):
            if a > 0.0:
                dev = abs(b / (a * scale) - 1.0)
                if dev > worst:
                    worst = dev
        return float(worst)
