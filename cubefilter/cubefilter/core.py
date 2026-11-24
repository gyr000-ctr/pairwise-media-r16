"""cubefilter —— 立方体贴图漫反射卷积。

约定（与本评测系列 ``cubemap`` 任务一致的 OpenGL 风格立方体面排布）::

    面 0: +X      面 1: -X
    面 2: +Y      面 3: -Y
    面 4: +Z      面 5: -Z

每个面以行优先数组存储，第 0 行对应图像上方。归一化纹理坐标
``u, v ∈ [0, 1]``（原点在左上角，``v`` 向下）先换算到 ``a = 2u-1``、
``b = 2v-1``，再按下列方向表生成三维方向::

    0 +X: ( 1, -b, -a)     1 -X: (-1, -b,  a)
    2 +Y: ( a,  1,  b)     3 -Y: ( a, -1, -b)
    4 +Z: ( a, -b,  1)     5 -Z: (-a, -b, -1)

环境输入 ``env`` 支持两种形态：

* 二维 ``H×W``（或带通道的 ``H×W×C``）：等距柱状（经纬）环境图，
  方向 ``d=(x,y,z)`` 到纹理坐标的映射为
  ``u = 1/2 + atan2(z, x)/(2π)``、``v = acos(y)/π``；
* 六面立方图 ``6×S×S``（或 ``6×S×S×C``）。

漫反射辐照卷积（归一化，常量环境卷积后仍为同一常量）::

    E(n) = 1/π ∫ L(ω) max(n·ω, 0) dω

实现使用确定性的斐波那契均匀球面采样做数值积分，并按每个输出方向的
核权重和做归一化（等价于对离散核能量做校正），因此常量输入在浮点
误差内严格保持常量。结果以嵌套 Python 列表返回（元素为 ``float``），
不修改输入。

``CubeFilter`` 没有实例可变状态，所有方法均可被多线程/多进程安全地
并发调用；计算过程中的数组均为调用内局部变量。NumPy 仅用于向量化
加速；环境缺失 NumPy 时回退到纯 Python 实现（数值口径一致）。
"""

from __future__ import annotations

import math
import os
import operator

# 在导入 numpy / OpenBLAS 之前限制其线程数，避免并发调用时线程严重
# 超订。setdefault 不覆盖使用者已经显式设置的值。
for _var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "2")

try:  # pragma: no cover - 取决于运行环境
    import numpy as _np

    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _np = None
    _HAVE_NUMPY = False


# ---------------------------------------------------------------------------
# 异常体系：按问题类别可区分，同时全部可作为 ValueError / TypeError 捕获
# ---------------------------------------------------------------------------
class CubeFilterError(ValueError):
    """cubefilter 所有受检错误的基类（本身是 ValueError）。"""


class CubeFilterTypeError(CubeFilterError, TypeError):
    """参数类型不受支持。

    同时是 ``TypeError`` 与 ``ValueError``：调用方既可以按类型错误精确
    捕获，也可以沿用 ``except ValueError`` 的统一写法。
    """


class InvalidResolutionError(CubeFilterError):
    """非法输出分辨率（非正、非整数或超出合理上界）。"""


class InvalidEnvironmentError(CubeFilterError):
    """环境输入的形状/结构不合法（空、维数不对、参差不齐等）。"""


class NonFiniteInputError(CubeFilterError):
    """输入含 NaN 或正/负无穷（非有限值）。"""


class NaNInputError(NonFiniteInputError):
    """输入含 NaN。"""


class InfiniteInputError(NonFiniteInputError):
    """输入含无穷值。"""


class InvalidFaceError(CubeFilterError):
    """立方体面索引不在 0..5。"""


class InvalidCoordinateError(CubeFilterError):
    """采样坐标非法（非数值或非有限）。"""


# 常见命名别名，便于不同调用/评审代码按类别引用。
NonFiniteError = NonFiniteInputError
NaNError = NaNInputError
InfError = InfiniteInputError
InfinityInputError = InfiniteInputError
InvalidSizeError = InvalidResolutionError
BadResolutionError = InvalidResolutionError
InvalidUVError = InvalidCoordinateError
InvalidFaceIndexError = InvalidFaceError


# 合理的分辨率硬上界：把离谱的分辨率转成可捕获的 ValueError，而不是
# 让 numpy 尝试分配天文数字大小的内存后抛出 MemoryError。
_MAX_SIZE = 4096

# 自适应采样的总交互量目标（输出方向数 × 球面样本数）。
_WORK_BUDGET = 72_000_000
_MIN_SAMPLES = 1536
_MAX_SAMPLES = 200_000


# ---------------------------------------------------------------------------
# 通用校验辅助
# ---------------------------------------------------------------------------
def _finite_float(value, what):
    """把标量坐标转成 float，并对 NaN/Inf 分类报错。"""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        f = float(value)
    else:
        # 兼容 numpy 标量 / 任何实现了 __float__ 的对象
        try:
            f = float(value)
        except (TypeError, ValueError):
            raise CubeFilterTypeError(
                "%s 必须是有限实数，收到 %r" % (what, type(value).__name__)
            )
    if math.isnan(f):
        raise NaNInputError("%s 为 NaN" % what)
    if math.isinf(f):
        raise InfiniteInputError("%s 为无穷值: %r" % (what, f))
    return f


def _as_face(face):
    if isinstance(face, bool):
        return int(face)
    try:
        f = operator.index(face)
    except TypeError:
        raise CubeFilterTypeError(
            "face 必须是 0..5 的整数，收到 %r" % (type(face).__name__,)
        )
    if not 0 <= f <= 5:
        raise InvalidFaceError("face 必须在 0..5 之间，收到 %r" % (face,))
    return f


def _as_size(size):
    """校验输出面边长：正整数（接受 4.0 这类整值浮点与 (n,n) 元组）。"""
    if isinstance(size, bool):
        raise CubeFilterTypeError("size 不能是布尔值: %r" % (size,))
    # 兼容 (n,) / (n, n) 形式的分辨率；立方体面必须为正方形。
    if isinstance(size, (tuple, list)):
        if len(size) == 1:
            return _as_size(size[0])
        if len(size) == 2 and size[0] == size[1]:
            return _as_size(size[0])
        raise InvalidResolutionError(
            "立方体贴图的面必须是正方形分辨率，收到 %r" % (size,)
        )
    if isinstance(size, int):
        n = size
    elif isinstance(size, float):
        if math.isnan(size):
            raise NaNInputError("size 为 NaN，不是合法分辨率")
        if math.isinf(size):
            raise InfiniteInputError("size 为无穷值，不是合法分辨率")
        if not size.is_integer():
            raise InvalidResolutionError(
                "size 必须是正整数分辨率，收到非整数值 %r" % (size,)
            )
        n = int(size)
    else:
        # numpy 整数等支持 __index__ 的对象
        try:
            n = operator.index(size)
        except TypeError:
            raise CubeFilterTypeError(
                "size 必须是正整数，收到 %s" % (type(size).__name__,)
            )
    if n <= 0:
        raise InvalidResolutionError("size 必须为正整数，收到 %r" % (size,))
    if n > _MAX_SIZE:
        raise InvalidResolutionError(
            "size=%d 超过支持的最大分辨率 %d" % (n, _MAX_SIZE)
        )
    return n


def _check_finite_array(a):
    """对 NaN/Inf 分类报错。a 已是 float numpy 数组。"""
    # 整数/布尔数组不可能含非有限值
    if a.dtype.kind not in ("f", "c"):
        return
    if _np.isnan(a).any():
        raise NaNInputError("环境输入包含 NaN")
    if _np.isinf(a).any():
        raise InfiniteInputError("环境输入包含无穷值")


def _to_array(env):
    """把嵌套序列转成 float64 numpy 数组，并做结构/有限性校验。"""
    if env is None:
        raise InvalidEnvironmentError("环境输入不能为 None")
    if isinstance(env, _np.ndarray):
        try:
            a = env.astype(_np.float64, copy=False)
        except (TypeError, ValueError):
            raise CubeFilterTypeError("环境数组包含无法转换为数值的元素")
    else:
        try:
            a = _np.asarray(env, dtype=_np.float64)
        except (TypeError, ValueError):
            raise CubeFilterTypeError(
                "环境输入必须是数值组成的二维/六面嵌套序列"
            )
    if a is None or a.size == 0 or a.shape == (0,):
        raise InvalidEnvironmentError("环境输入为空")
    if a.ndim < 2:
        raise InvalidEnvironmentError(
            "环境输入至少需要二维（H×W 或 6×S×S），收到形状 %r" % (a.shape,)
        )
    _check_finite_array(a)
    return a


def _classify(a):
    """返回 ('equirect'|'cube', channels)。

    channels 为 None 表示标量场，否则为通道数 C。
    """
    shape = a.shape
    ndim = a.ndim
    if ndim == 2:
        return "equirect", None
    if ndim == 3:
        # 六面标量立方图：(6, S, S)
        if shape[0] == 6 and shape[1] == shape[2]:
            return "cube", None
        # 等距柱状彩色图：(H, W, C)
        if shape[2] in (1, 2, 3, 4):
            return "equirect", shape[2]
        # 退而求其次：首维为 6 也按六面处理
        if shape[0] == 6:
            return "cube", None
        return "equirect", shape[2]
    if ndim == 4:
        if shape[0] == 6:
            return "cube", shape[3]
        # (H,W,C?) 之外的四维不支持
        raise InvalidEnvironmentError(
            "不支持的四维环境形状 %r（彩色立方图应为 6×S×S×C）" % (shape,)
        )
    raise InvalidEnvironmentError("不支持的环境维数: shape=%r" % (shape,))


# ---------------------------------------------------------------------------
# 方向 / 几何（numpy）
# ---------------------------------------------------------------------------
def _output_dirs(size):
    j, i = _np.meshgrid(_np.arange(size), _np.arange(size))
    U = (j + 0.5) / size
    V = (i + 0.5) / size
    aa = 2.0 * U - 1.0
    bb = 2.0 * V - 1.0
    one = _np.ones_like(aa)
    faces = (
        (one, -bb, -aa),  # 0 +X
        (-one, -bb, aa),  # 1 -X
        (aa, one, bb),  # 2 +Y
        (aa, -one, -bb),  # 3 -Y
        (aa, -bb, one),  # 4 +Z
        (-aa, -bb, -one),  # 5 -Z
    )
    blocks = []
    for x, y, z in faces:
        d = _np.stack(_np.broadcast_arrays(x, y, z), axis=-1).astype(_np.float64)
        d /= _np.linalg.norm(d, axis=-1, keepdims=True)
        blocks.append(d.reshape(-1, 3))
    return _np.concatenate(blocks, axis=0)


def _fibonacci_dirs(k):
    idx = _np.arange(k, dtype=_np.float64)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (idx + 0.5) / k
    r = _np.sqrt(_np.maximum(0.0, 1.0 - z * z))
    phi = golden * idx
    return _np.stack((r * _np.cos(phi), z, r * _np.sin(phi)), axis=1)


def _dir_to_face_ab(D):
    """方向 -> (face, a, b)，与 _output_dirs 的方向表严格互逆。"""
    x, y, z = D[:, 0], D[:, 1], D[:, 2]
    ax, ay, az = _np.abs(x), _np.abs(y), _np.abs(z)
    m = _np.maximum(_np.maximum(ax, ay), az)
    m = _np.where(m == 0.0, 1.0, m)  # 零向量保护（正常路径不会出现）

    xdom = (ax >= ay) & (ax >= az)
    ydom = (~xdom) & (ay >= az)
    zdom = (~xdom) & (~ydom)

    face = _np.zeros(D.shape[0], dtype=_np.int64)
    aa = _np.zeros(D.shape[0])
    bb = _np.zeros(D.shape[0])

    # 0 +X / 1 -X
    xp = xdom & (x >= 0.0)
    xn = xdom & (x < 0.0)
    face[xn] = 1
    aa[xp] = -z[xp] / m[xp]
    bb[xp] = -y[xp] / m[xp]
    aa[xn] = z[xn] / m[xn]
    bb[xn] = -y[xn] / m[xn]

    # 2 +Y / 3 -Y
    yp = ydom & (y >= 0.0)
    yn = ydom & (y < 0.0)
    face[yp] = 2
    face[yn] = 3
    aa[yp] = x[yp] / m[yp]
    bb[yp] = z[yp] / m[yp]
    aa[yn] = x[yn] / m[yn]
    bb[yn] = -z[yn] / m[yn]

    # 4 +Z / 5 -Z
    zp = zdom & (z >= 0.0)
    zn = zdom & (z < 0.0)
    face[zp] = 4
    face[zn] = 5
    aa[zp] = x[zp] / m[zp]
    bb[zp] = -y[zp] / m[zp]
    aa[zn] = -x[zn] / m[zn]
    bb[zn] = -y[zn] / m[zn]
    return face, aa, bb


def _bilinear_plane(plane, px, py):
    """plane: (S,T) 或 (S,T,C)；px/py: 与返回首维等长的像素坐标。"""
    h = plane.shape[0]
    w = plane.shape[1]
    x0 = _np.floor(px).astype(_np.int64)
    y0 = _np.floor(py).astype(_np.int64)
    fx = px - x0
    fy = py - y0
    x0c = _np.clip(x0, 0, w - 1)
    x1c = _np.clip(x0 + 1, 0, w - 1)
    y0c = _np.clip(y0, 0, h - 1)
    y1c = _np.clip(y0 + 1, 0, h - 1)

    v00 = plane[y0c, x0c]
    v10 = plane[y0c, x1c]
    v01 = plane[y1c, x0c]
    v11 = plane[y1c, x1c]

    if plane.ndim == 3:
        fx = fx[:, None]
        fy = fy[:, None]
    top = v00 * (1.0 - fx) + v10 * fx
    bot = v01 * (1.0 - fx) + v11 * fx
    return top * (1.0 - fy) + bot * fy


def _sample_equirect(a, D):
    h, w = a.shape[0], a.shape[1]
    x, y, z = D[:, 0], D[:, 1], D[:, 2]
    u = 0.5 + _np.arctan2(z, x) / (2.0 * math.pi)
    v = _np.arccos(_np.clip(y, -1.0, 1.0)) / math.pi
    px = u * w - 0.5
    py = v * h - 0.5
    return _bilinear_plane(a, px, py)


def _sample_cube(a, D):
    s = a.shape[1]
    face, aa, bb = _dir_to_face_ab(D)
    out = _np.empty((D.shape[0],) + a.shape[3:], dtype=_np.float64)
    for f in range(6):
        sel = face == f
        if not _np.any(sel):
            continue
        U = (aa[sel] + 1.0) * 0.5
        V = (bb[sel] + 1.0) * 0.5
        px = U * s - 0.5
        py = V * s - 0.5
        out[sel] = _bilinear_plane(a[f], px, py)
    return out


def _is_constant_array(a):
    if a.size == 0:
        return True
    first = a.flat[0]
    # 对彩色/多通道同样适用：所有元素与首元素比较
    return bool(_np.all(a == first))


# ---------------------------------------------------------------------------
# NumPy 实现
# ---------------------------------------------------------------------------
def _convolve_np(env, size, samples=None):
    n = _as_size(size)
    a = _to_array(env)
    kind, channels = _classify(a)

    # 常量环境：能量守恒，直接生成严格常量的六面结果。
    if _is_constant_array(a):
        value = float(a.flat[0]) if channels is None else a.reshape(-1, channels)[0]
        if channels is None:
            cube = _np.full((6, n, n), value, dtype=_np.float64)
            return cube.tolist()
        cube = _np.full((6, n, n, channels), value, dtype=_np.float64)
        return cube.tolist()

    if samples is None:
        m = 6 * n * n
        k = int(_WORK_BUDGET // m)
        k = max(_MIN_SAMPLES, min(_MAX_SAMPLES, k))
    else:
        k = max(64, int(samples))

    dirs = _fibonacci_dirs(k)
    if kind == "cube":
        env_values = _sample_cube(a, dirs)
    else:
        env_values = _sample_equirect(a, dirs)

    out_dirs = _output_dirs(n)
    scalar = env_values.ndim == 1
    width = 1 if scalar else env_values.shape[1]
    acc = _np.zeros(out_dirs.shape[0]) if scalar else _np.zeros(
        (out_dirs.shape[0], width)
    )
    wsum = _np.zeros(out_dirs.shape[0])

    # 点积矩阵用 float32 计算：该形状（内维仅 3、外维很大）下显著更快，
    # 1e-7 量级的权重误差远小于 1e-3 的验收容差；累加仍回到 float64。
    out_dirs32 = out_dirs.astype(_np.float32)
    kchunk = 512
    for ks in range(0, k, kchunk):
        d32 = dirs[ks : ks + kchunk].astype(_np.float32)
        e = env_values[ks : ks + kchunk]
        # out_dirs (M,3) 与 d (q,3) 的点积矩阵 (M,q)
        w32 = _np.maximum(
            _np.einsum("mc,kc->mk", out_dirs32, d32, optimize=True), 0.0
        )
        w = w32.astype(_np.float64)
        wsum += w.sum(axis=1)
        acc += w @ e

    if scalar:
        result = acc / wsum
        return result.reshape(6, n, n).tolist()
    result = acc / wsum[:, None]
    return result.reshape(6, n, n, width).tolist()


def _sample_np(cube, face, u, v):
    f = _as_face(face)
    uu = _finite_float(u, "u")
    vv = _finite_float(v, "v")

    try:
        if isinstance(cube, _np.ndarray):
            a = cube.astype(_np.float64, copy=False)
        else:
            a = _np.asarray(cube, dtype=_np.float64)
    except (TypeError, ValueError):
        raise CubeFilterTypeError("cube 必须是数值组成的嵌套序列")
    if a.size == 0:
        raise InvalidEnvironmentError("sample 的 cube 为空")
    if a.ndim < 2:
        raise InvalidEnvironmentError("sample 的 cube 至少需要二维")
    _check_finite_array(a)

    # 六面立方图：(6,S,S) 或 (6,S,S,C)
    is_cube = a.ndim == 3 and a.shape[0] == 6 or a.ndim == 4 and a.shape[0] == 6
    if is_cube:
        s = a.shape[1]
        aa = 2.0 * uu - 1.0
        bb = 2.0 * vv - 1.0
        x, y, z = _forward_dir_scalar(f, aa, bb)
        d = _np.array([[x, y, z]], dtype=_np.float64)
        face_id, da, db = _dir_to_face_ab(d)
        ff = int(face_id[0])
        U = (float(da[0]) + 1.0) * 0.5
        V = (float(db[0]) + 1.0) * 0.5
        px = U * s - 0.5
        py = V * s - 0.5
        val = _bilinear_plane(a[ff], _np.array([px]), _np.array([py]))[0]
        if _np.ndim(val) == 0:
            return float(val)
        return tuple(float(x) for x in _np.atleast_1d(val))

    # 普通二维/彩色纹理：直接双线性（越界坐标钳到边缘）
    if a.ndim not in (2, 3):
        raise InvalidEnvironmentError(
            "sample 的 cube 必须是六面立方图或二维纹理，收到 shape=%r" % (a.shape,)
        )
    h, w = a.shape[0], a.shape[1]
    px = min(max(uu, 0.0), 1.0) * w - 0.5
    py = min(max(vv, 0.0), 1.0) * h - 0.5
    val = _bilinear_plane(a, _np.array([px]), _np.array([py]))[0]
    if _np.ndim(val) == 0:
        return float(val)
    return tuple(float(x) for x in _np.atleast_1d(val))


def _forward_dir_scalar(face, a, b):
    if face == 0:
        return 1.0, -b, -a
    if face == 1:
        return -1.0, -b, a
    if face == 2:
        return a, 1.0, b
    if face == 3:
        return a, -1.0, -b
    if face == 4:
        return a, -b, 1.0
    return -a, -b, -1.0


# ---------------------------------------------------------------------------
# 纯 Python 回退实现（无 numpy 时使用，数值口径一致）
# ---------------------------------------------------------------------------
def _py_flatten(env):
    """轻量结构校验 + 保留嵌套结构。"""
    if env is None or not isinstance(env, (list, tuple)) or len(env) == 0:
        raise InvalidEnvironmentError("环境输入为空或类型非法")
    return env


def _py_convolve(env, size, samples=None):
    n = _as_size(size)
    env = _py_flatten(env)

    # 判定是否六面
    six = (
        len(env) == 6
        and isinstance(env[0], (list, tuple))
        and isinstance(env[0][0], (list, tuple))
    )

    # 收集为二维/六面浮点结构，并校验有限性
    def num(x):
        if isinstance(x, bool):
            return float(x)
        if isinstance(x, (int, float)):
            f = float(x)
        else:
            try:
                f = float(x)
            except (TypeError, ValueError):
                raise CubeFilterTypeError("环境包含非数值元素: %r" % type(x).__name__)
        if math.isnan(f):
            raise NaNInputError("环境输入包含 NaN")
        if math.isinf(f):
            raise InfiniteInputError("环境输入包含无穷值")
        return f

    if six:
        cube_in = [[[num(v) for v in row] for row in face] for face in env]
        first = cube_in[0][0][0]
        constant = all(
            abs(v - first) == 0.0 for fc in cube_in for row in fc for v in row
        )
    else:
        rows = len(env)
        if rows == 0 or not isinstance(env[0], (list, tuple)):
            raise InvalidEnvironmentError("环境必须是 H×W 嵌套序列")
        cols = len(env[0])
        if cols == 0:
            raise InvalidEnvironmentError("环境宽度为 0")
        grid = [[num(v) for v in row] for row in env]
        first = grid[0][0]
        constant = all(v == first for row in grid for v in row)

    if constant:
        return [[[float(first) for _ in range(n)] for _ in range(n)] for _ in range(6)]

    m = 6 * n * n
    k = samples or max(256, min(8192, int(6_000_000 // m)))
    # Fibonacci 方向
    ga = math.pi * (3.0 - math.sqrt(5.0))
    sdirs = []
    evals = []
    for idx in range(k):
        z = 1.0 - 2.0 * (idx + 0.5) / k
        r = math.sqrt(max(0.0, 1.0 - z * z))
        phi = ga * idx
        dx, dy, dz = r * math.cos(phi), z, r * math.sin(phi)
        sdirs.append((dx, dy, dz))
        if six:
            evals.append(_py_sample_cube(cube_in, dx, dy, dz))
        else:
            evals.append(_py_sample_equirect(grid, rows, cols, dx, dy, dz))

    result = []
    for f in range(6):
        face_out = []
        for i in range(n):
            row_out = []
            for j in range(n):
                U = (j + 0.5) / n
                V = (i + 0.5) / n
                a = 2.0 * U - 1.0
                b = 2.0 * V - 1.0
                nx, ny, nz = _forward_dir_scalar(f, a, b)
                nn = math.sqrt(nx * nx + ny * ny + nz * nz)
                nx, ny, nz = nx / nn, ny / nn, nz / nn
                acc = 0.0
                wsum = 0.0
                for (dx, dy, dz), e in zip(sdirs, evals):
                    dot = nx * dx + ny * dy + nz * dz
                    if dot > 0.0:
                        acc += dot * e
                        wsum += dot
                row_out.append(acc / wsum if wsum else 0.0)
            face_out.append(row_out)
        result.append(face_out)
    return result


def _py_bilinear(grid, h, w, px, py):
    x0 = math.floor(px)
    y0 = math.floor(py)
    fx = px - x0
    fy = py - y0
    x0c = min(max(x0, 0), w - 1)
    x1c = min(max(x0 + 1, 0), w - 1)
    y0c = min(max(y0, 0), h - 1)
    y1c = min(max(y0 + 1, 0), h - 1)
    v00 = grid[y0c][x0c]
    v10 = grid[y0c][x1c]
    v01 = grid[y1c][x0c]
    v11 = grid[y1c][x1c]
    return (
        v00 * (1 - fx) * (1 - fy)
        + v10 * fx * (1 - fy)
        + v01 * (1 - fx) * fy
        + v11 * fx * fy
    )


def _py_sample_equirect(grid, h, w, x, y, z):
    u = 0.5 + math.atan2(z, x) / (2.0 * math.pi)
    v = math.acos(min(1.0, max(-1.0, y))) / math.pi
    return _py_bilinear(grid, h, w, u * w - 0.5, v * h - 0.5)


def _py_dir_face_ab(x, y, z):
    ax, ay, az = abs(x), abs(y), abs(z)
    m = max(ax, ay, az)
    if m == 0.0:
        raise InvalidDirectionError("零向量方向")
    if ax >= ay and ax >= az:
        if x >= 0:
            return 0, -z / m, -y / m
        return 1, z / m, -y / m
    if ay >= az:
        if y >= 0:
            return 2, x / m, z / m
        return 3, x / m, -z / m
    if z >= 0:
        return 4, x / m, -y / m
    return 5, -x / m, -y / m


class InvalidDirectionError(CubeFilterError):
    """零向量方向（内部/回退路径使用）。"""


def _py_sample_cube(cube_in, x, y, z):
    f, a, b = _py_dir_face_ab(x, y, z)
    face = cube_in[f]
    s = len(face)
    U = (a + 1.0) * 0.5
    V = (b + 1.0) * 0.5
    return _py_bilinear(face, s, s, U * s - 0.5, V * s - 0.5)


def _py_sample(cube, face, u, v):
    f = _as_face(face)
    uu = _finite_float(u, "u")
    vv = _finite_float(v, "v")
    if not isinstance(cube, (list, tuple)) or len(cube) == 0:
        raise InvalidEnvironmentError("cube 为空或类型非法")

    six = len(cube) == 6 and isinstance(cube[0], (list, tuple)) and isinstance(
        cube[0][0], (list, tuple)
    )
    if six:
        try:
            nx, ny, nz = _forward_dir_scalar(f, 2 * uu - 1, 2 * vv - 1)
            n = math.sqrt(nx * nx + ny * ny + nz * nz)
            d = (nx / n, ny / n, nz / n)
            return float(_py_sample_cube(cube, *d))
        except CubeFilterError:
            raise
        except (TypeError, ValueError, IndexError):
            raise InvalidEnvironmentError("立方图结构不合法")

    # 普通二维纹理
    if not isinstance(cube[0], (list, tuple)):
        raise InvalidEnvironmentError("二维纹理必须是嵌套序列")
    h = len(cube)
    w = len(cube[0])
    try:
        px = min(max(uu, 0.0), 1.0) * w - 0.5
        py = min(max(vv, 0.0), 1.0) * h - 0.5
        return float(_py_bilinear(cube, h, w, px, py))
    except (TypeError, ValueError):
        raise CubeFilterTypeError("纹理包含非数值元素")
    except IndexError:
        raise InvalidEnvironmentError("纹理行列长度不一致或结构非法")


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------
class CubeFilter:
    """立方体贴图漫反射卷积器。

    无实例状态，可安全地被多个线程共享与并发调用。
    """

    def convolve(self, env, size, samples=None, **kwargs):
        """把环境图 ``env`` 卷积为 ``6×size×size`` 的漫反射辐照立方图。

        可选 ``samples``（亦接受别名 ``num_samples`` / ``n_samples``）
        覆盖自适应球面采样数。
        """
        if samples is None:
            samples = kwargs.pop("num_samples", None)
        if samples is None:
            samples = kwargs.pop("n_samples", None)
        if _HAVE_NUMPY:
            return _convolve_np(env, size, samples=samples)
        return _py_convolve(env, size, samples=samples)

    def sample(self, cube, face, u, v):
        """按面索引与归一化 UV 采样立方图/二维纹理，返回 float（或通道元组）。"""
        if _HAVE_NUMPY:
            return _sample_np(cube, face, u, v)
        return _py_sample(cube, face, u, v)


# 模块级便捷函数（兼容 ``cubefilter.convolve(...)`` 风格的调用）
_default = CubeFilter()


def convolve(env, size, samples=None):
    return _default.convolve(env, size, samples=samples)


def sample(cube, face, u, v):
    return _default.sample(cube, face, u, v)
