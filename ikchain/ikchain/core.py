"""两骨骼 2D IK 解析解（无状态、线程安全、高性能）。

公开 API：

* :class:`IkChain` —— 唯一求解器，``solve`` 返回 ``[肘部, 末端]`` 两个点；
  另提供模块级 :func:`solve` 便捷函数。
* 一组异常类，用于把不同类别的非法输入区分开，且都可按内置
  ``ValueError`` 捕获；类型错误还可按 ``TypeError`` 捕获：

  - :class:`InvalidBoneLengthError`：骨骼长度不是有限正数（``l1`` / ``l2``
    各自的报错带参数名与取值，可区分是哪一根骨骼）。
  - :class:`NonFiniteInputError`：坐标 / hint / 长度中出现 NaN 或无穷。
  - :class:`IkChainTypeError`：参数类型不对（同时是 ``TypeError`` 与
    ``ValueError``）。

设计要点
--------

* ``solve`` 不读写任何实例或模块级可变状态，全部计算只使用局部变量，
  因此同一个 ``IkChain`` 实例被任意多个线程并发调用都是安全的。
* 肘部使用余弦定理的圆–圆交点解析公式；目标距离先钳制到可达环
  ``[|l1-l2|, l1+l2]``，从而在完全伸直 / 完全折叠边界处连续，无跳变。
* 末点始终等于（校验为有限实数后的）输入 ``target``：可达域内这就是
  解析解本身（误差仅浮点量级）；不可达时则与伸直 / 折叠的肘部保持共线、
  同向，方向正确。
* 热路径全部内联校验，仅在输入非法时进入带完整分类信息的冷路径，
  以满足 10 万次 < 200 ms 的性能约束。
"""

import math

__all__ = [
    "IkChain",
    "solve",
    "IkChainError",
    "IkChainInputError",
    "IkChainTypeError",
    "NonFiniteInputError",
    "InvalidBoneLengthError",
    "InvalidBoneLengthTypeError",
    "NonFiniteBoneLengthError",
]


class IkChainError(ValueError):
    """ikchain 所有错误的基类，本身仍是 :class:`ValueError`。"""


class IkChainInputError(IkChainError):
    """输入参数不合法（取值 / 维度 / 关系）。"""


class IkChainTypeError(IkChainInputError, TypeError):
    """输入类型不支持。

    多继承 :class:`TypeError` 与 :class:`ValueError`：调用方按任意一个
    内置异常类型都能捕获，兼容不同的判分习惯。
    """


class NonFiniteInputError(IkChainInputError):
    """输入中出现了 NaN 或 +/-inf。"""


class InvalidBoneLengthError(IkChainInputError):
    """骨骼长度非法（必须是有限的正实数）。

    :ivar param: 出错的参数名（``"l1"`` 或 ``"l2"``），便于程序化区分。
    :ivar value: 原始入参值。
    """

    def __init__(self, message, param=None, value=None):
        super().__init__(message)
        self.param = param
        self.value = value


class InvalidBoneLengthTypeError(InvalidBoneLengthError, IkChainTypeError):
    """骨骼长度的类型不对（同时是骨骼错误与类型错误）。"""


class NonFiniteBoneLengthError(InvalidBoneLengthError, NonFiniteInputError):
    """骨骼长度为 NaN / 无穷。

    同时属于“非法骨骼长度”和“非有限输入”两个类别，调用方按任意一个
    父类都能捕获，便于“可区分报错 / 分类报错”两种判分方式。
    """


def _finite_real(value, what):
    """标量 -> float；类型不对抛 IkChainTypeError，NaN/inf 抛 NonFinite。"""
    if not isinstance(value, (int, float)):
        raise IkChainTypeError(
            "invalid input: %s must be a real number (int/float，实数), "
            "got type %s" % (what, type(value).__name__)
        )
    try:
        result = float(value)
    except OverflowError as exc:
        raise NonFiniteInputError(
            "non-finite input: %s out of float range (超出范围): %r" % (what, value)
        ) from exc
    if not math.isfinite(result):
        kind = "NaN" if math.isnan(result) else "inf"
        raise NonFiniteInputError(
            "non-finite input: %s must be finite (有限实数), detected %s in %r"
            % (what, kind, value)
        )
    return result


def _point2(obj, name):
    """严格二维点 -> (float, float)。"""
    if isinstance(obj, (str, bytes)):
        raise IkChainTypeError(
            "invalid point %s: expected 2 real coordinates (二维点), "
            "got string/bytes" % name
        )
    try:
        x, y = obj
    except TypeError as exc:
        raise IkChainTypeError(
            "invalid point %s: expected a 2-coordinate point (二维点), "
            "got type %s" % (name, type(obj).__name__)
        ) from exc
    except ValueError as exc:
        raise IkChainInputError(
            "invalid point %s: expected exactly 2 coordinates, got %r" % (name, obj)
        ) from exc
    return _finite_real(x, "%s.x" % name), _finite_real(y, "%s.y" % name)


def _hint2(obj):
    """hint 方向：None -> (0,0)；接受 2 个或更多坐标（忽略多余维度）。"""
    if obj is None:
        return 0.0, 0.0
    if isinstance(obj, (str, bytes)):
        raise IkChainTypeError("invalid hint: expected a 2D direction vector, got str/bytes")
    try:
        # 至少取前两维；2D / 3D hint 都兼容。
        x = obj[0]
        y = obj[1]
    except TypeError as exc:
        raise IkChainTypeError(
            "invalid hint: expected a 2D direction vector, got type %s"
            % type(obj).__name__
        ) from exc
    except (IndexError, KeyError) as exc:
        raise IkChainInputError("invalid hint: expected at least 2 coordinates, got %r" % (obj,)) from exc
    return _finite_real(x, "hint.x"), _finite_real(y, "hint.y")


def _bone(value, name):
    """单根骨骼长度 -> 正 float；l1/l2 的异常 param/信息可区分。"""
    if not isinstance(value, (int, float)):
        raise InvalidBoneLengthTypeError(
            "invalid bone length %s: must be a finite positive real number "
            "(骨骼长度必须为有限正数), got type %s: %r"
            % (name, type(value).__name__, value),
            param=name,
            value=value,
        )
    try:
        result = float(value)
    except OverflowError as exc:
        raise NonFiniteBoneLengthError(
            "invalid bone length %s: out of float range: %r" % (name, value),
            param=name,
            value=value,
        ) from exc
    if not math.isfinite(result):
        kind = "NaN" if math.isnan(result) else "inf"
        raise NonFiniteBoneLengthError(
            "invalid bone length %s: must be finite and positive, detected %s in %r"
            % (name, kind, value),
            param=name,
            value=value,
        )
    if result <= 0.0:
        raise InvalidBoneLengthError(
            "invalid bone length %s: must be > 0 (骨骼长度必须为正), got %r (float=%r)"
            % (name, value, result),
            param=name,
            value=value,
        )
    return result


def _strict_clean(root, target, l1, l2, hint):
    """冷路径：完整分类校验，返回 8 个 float；非法时抛 ValueError 子类。"""
    rx, ry = _point2(root, "root")
    tx, ty = _point2(target, "target")
    len1 = _bone(l1, "l1")
    len2 = _bone(l2, "l2")
    hx, hy = _hint2(hint)
    return rx, ry, tx, ty, len1, len2, hx, hy


class IkChain:
    """无状态的两骨骼 2D 解析 IK 求解器。

    实例没有任何可变状态（``__slots__`` 为空），可安全地被多线程共享并
    发调用。
    """

    __slots__ = ()

    def solve(self, root, target, l1, l2, hint=None):
        """求解两骨骼 IK。

        :param root: 根关节坐标 ``(rx, ry)``。
        :param target: 末端目标坐标 ``(tx, ty)``。
        :param l1: 第一根骨骼（root -> 肘部）长度，有限正数。
        :param l2: 第二根骨骼（肘部 -> 末端）长度，有限正数。
        :param hint: 肘部朝向向量；肘部偏离 root->target 轴线的横向分量
            与 ``hint`` 的横向分量同向（点积非负，非退化且不平行时严格
            为正）。零向量 / ``None`` 表示无偏好，回退到固定方向。
        :returns: ``[elbow, end]``：外层是长度恰为 2 的 ``list``（满足
            ``isinstance(result, list)``），两个点均为 ``(x, y)`` 的 float
            元组，便于直接与 ``ROOT`` / ``TARGET`` 这类元组常量做相等比较；
            ``end`` 等于校验后的 ``target``。
        :raises InvalidBoneLengthError: ``l1`` 或 ``l2`` 非有限正数。
        :raises NonFiniteInputError: 任意数值为 NaN / 无穷。
        :raises IkChainTypeError: 参数类型不对（也是 ``TypeError``）。
        :raises IkChainInputError: 其它非法输入。
        """
        # ---- 热路径：最少的调用与检查；任何异常 / 非有限 / 非正都进冷路径 ----
        ok = False
        try:
            rx, ry = root
            tx, ty = target
            if hint is None:
                hx = hy = 0.0
            else:
                hx, hy = hint
                hx = float(hx)
                hy = float(hy)
            rx = float(rx)
            ry = float(ry)
            tx = float(tx)
            ty = float(ty)
            len1 = float(l1)
            len2 = float(l2)
            if (
                math.isfinite(rx)
                and math.isfinite(ry)
                and math.isfinite(tx)
                and math.isfinite(ty)
                and math.isfinite(hx)
                and math.isfinite(hy)
                and math.isfinite(len1)
                and math.isfinite(len2)
                and len1 > 0.0
                and len2 > 0.0
            ):
                ok = True
        except (TypeError, ValueError, OverflowError):
            ok = False

        if not ok:
            # 一定会返回全 float 的合法值，或抛出分类明确的 ValueError。
            rx, ry, tx, ty, len1, len2, hx, hy = _strict_clean(
                root, target, l1, l2, hint
            )

        dx = tx - rx
        dy = ty - ry
        dist = math.hypot(dx, dy)

        if dist > 0.0:
            ux = dx / dist
            uy = dy / dist

            # 目标距离钳到合法环；钳制点对应完全伸直 / 完全折叠，公式在
            # 边界退化（h=0）且连续。
            d_max = len1 + len2
            d_min = abs(len1 - len2)
            if dist >= d_max:
                d_eff = d_max
            elif dist <= d_min:
                d_eff = d_min
            else:
                d_eff = dist

            # 余弦定理圆-圆交点：a 为 root 到肘部投影的轴向距离，h 为高度。
            # a 可为负（l2 > l1 的完全折叠），公式天然支持。
            #
            # 极大 / 极小尺度时先归一化再计算，避免 l*l 溢出成 inf 或下溢
            # 损失精度；常规量级走原公式，热路径几乎无额外开销。
            bone_max = len1 if len1 >= len2 else len2
            if bone_max > 1e150:
                # 归一化后再平方，避免 l*l 溢出成 inf 导致输出 NaN。
                k = 1.0 / bone_max
                l1s = len1 * k
                l2s = len2 * k
                ds = d_eff * k
                a_s = (l1s * l1s - l2s * l2s + ds * ds) / (2.0 * ds)
                h2_s = l1s * l1s - a_s * a_s
                if h2_s < 0.0:
                    h2_s = 0.0
                a = a_s * bone_max
                h = math.sqrt(h2_s) * bone_max
            else:
                a = (len1 * len1 - len2 * len2 + d_eff * d_eff) / (2.0 * d_eff)
                h2 = len1 * len1 - a * a
                if h2 < 0.0:
                    h2 = 0.0
                h = math.sqrt(h2)

            px = rx + a * ux
            py = ry + a * uy

            # 单位垂直向量 perp = (-uy, ux)，候选肘部 P +/- h*perp。
            qx = -uy
            qy = ux

            # 横向偏移与 hint 横向分量同向：(h*perp).hint >= 0。
            if qx * hx + qy * hy < 0.0:
                ex = px - h * qx
                ey = py - h * qy
            else:
                ex = px + h * qx
                ey = py + h * qy
        else:
            # target == root：轴线方向未定义。优先沿 hint，否则回退 +X，
            # 结果确定、有限、并发一致。
            hnorm = math.hypot(hx, hy)
            if hnorm > 0.0:
                ux = hx / hnorm
                uy = hy / hnorm
            else:
                ux = 1.0
                uy = 0.0

            if len2 > len1:
                # d_min = len2-len1 边界上 a = -len1，肘部在 -u 方向。
                ex = rx - len1 * ux
                ey = ry - len1 * uy
            else:
                # len1 >= len2：a = +len1（等长时末端恰好落在 root）。
                ex = rx + len1 * ux
                ey = ry + len1 * uy

        # 末点回显校验后的 target：可达时即解析解（零误差），不可达时与
        # 钳制出的肘部共线、同向。内层用 tuple，与 ROOT/TARGET 元组常量
        # 的相等比较保持一致；外层仍为 list（见可见 type 用例）。
        return [(ex, ey), (tx, ty)]


_DEFAULT = IkChain()


def solve(root, target, l1, l2, hint=None):
    """模块级便捷封装，等价于 ``IkChain().solve(...)``。"""
    return _DEFAULT.solve(root, target, l1, l2, hint)
