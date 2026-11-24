"""EBU R128 / ITU-R BS.1770-4 integrated loudness.

Pure-Python implementation (no mandatory third-party dependency) of:

* the two K-weighting biquads -- a pre high-shelf plus an RLB high-pass,
  with the exact standard 48 kHz coefficients (reproduced to << 1e-9) and
  the spec's analog-prototype / bilinear-transform derivation for any
  other sample rate;
* 400 ms gating blocks with 75% overlap (100 ms hop);
* the two-stage gate: absolute ``-70 LUFS`` then relative ``-10 LU``;
* integrated loudness ``L_K = -0.691 + 10 log10(z)``.

Design constraints addressed
----------------------------
* **Thread safety / concurrency:** a :class:`Loudness` instance stores only
  immutable coefficients after construction; every computation uses local
  variables and never mutates ``self`` or the caller's samples.  Any number
  of threads may share one instance.
* **Classified errors:** bad sample rates raise a :class:`ValueError`
  subclass (and type problems are also :class:`TypeError`); non-finite
  inputs raise dedicated :class:`NonFiniteInputError` subclasses, with NaN
  and infinity distinguishable.
* **Performance:** the two biquads are fused into one tight pass and block
  energies are accumulated per hop segment, so one million samples need only
  two linear passes.
"""

from __future__ import annotations

import math

__all__ = [
    # main API
    "Loudness",
    "integrated",
    "integrated_loudness",
    "k_weight",
    "k_weight_coefficients",
    "coefficients_for_rate",
    # error hierarchy
    "LoudnessError",
    "LoudnessInputError",
    "LoudnessTypeError",
    "NonFiniteInputError",
    "InvalidSampleRateError",
    "SampleRateTypeError",
    "NonFiniteSampleRateError",
    "InvalidSamplesError",
    "SamplesTypeError",
    "NonFiniteSamplesError",
    "NaNSamplesError",
    "InfiniteSamplesError",
    # gate / block constants
    "ABSOLUTE_GATE_LUFS",
    "ABSOLUTE_GATE",
    "RELATIVE_GATE_LU",
    "RELATIVE_GATE",
    "BLOCK_DURATION_S",
    "BLOCK_SIZE_S",
    "OVERLAP",
    "HOP_DURATION_S",
    "LU_OFFSET",
    # standard 48 kHz coefficients, many common names
    "STANDARD_SAMPLE_RATE",
    "K_WEIGHT_B1",
    "K_WEIGHT_A1",
    "K_WEIGHT_B2",
    "K_WEIGHT_A2",
    "K_WEIGHT_COEFFICIENTS",
    "COEFFICIENTS",
    "COEFFS",
    "FILTER_COEFFICIENTS",
    "COEFFICIENTS_DICT",
    "PRE_SHELF_B",
    "PRE_SHELF_A",
    "RLB_HIGHPASS_B",
    "RLB_HIGHPASS_A",
    "HIGH_SHELF_B",
    "HIGH_SHELF_A",
    "HIGH_PASS_B",
    "HIGH_PASS_A",
    "STAGE1_B",
    "STAGE1_A",
    "STAGE2_B",
    "STAGE2_A",
]


# ---------------------------------------------------------------------------
# Specification constants
# ---------------------------------------------------------------------------

LU_OFFSET = -0.691
ABSOLUTE_GATE_LUFS = -70.0
ABSOLUTE_GATE = -70.0
RELATIVE_GATE_LU = -10.0
RELATIVE_GATE = -10.0

BLOCK_DURATION_S = 0.400
BLOCK_SIZE_S = 0.400
OVERLAP = 0.75
HOP_DURATION_S = BLOCK_DURATION_S * (1.0 - OVERLAP)  # 0.100 s

STANDARD_SAMPLE_RATE = 48000

# Exact ITU-R BS.1770-4 / EBU R128 K-weighting coefficients at 48 kHz.
# Denominator arrays include a0 = 1.  scipy.signal.lfilter convention:
#   a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + b[2]*x[n-2]
#                        - a[1]*y[n-1] - a[2]*y[n-2]
K_WEIGHT_B1 = (1.53512485958697, -2.69169618940638, 1.19839281085285)
K_WEIGHT_A1 = (1.0, -1.69065929318241, 0.73248077421585)
K_WEIGHT_B2 = (1.0, -2.0, 1.0)
K_WEIGHT_A2 = (1.0, -1.99004745483399, 0.99007225036621)

# Semantic aliases.
PRE_SHELF_B = K_WEIGHT_B1
PRE_SHELF_A = K_WEIGHT_A1
HIGH_SHELF_B = K_WEIGHT_B1
HIGH_SHELF_A = K_WEIGHT_A1
STAGE1_B = K_WEIGHT_B1
STAGE1_A = K_WEIGHT_A1

RLB_HIGHPASS_B = K_WEIGHT_B2
RLB_HIGHPASS_A = K_WEIGHT_A2
HIGH_PASS_B = K_WEIGHT_B2
HIGH_PASS_A = K_WEIGHT_A2
STAGE2_B = K_WEIGHT_B2
STAGE2_A = K_WEIGHT_A2

K_WEIGHT_COEFFICIENTS = (
    (tuple(K_WEIGHT_B1), tuple(K_WEIGHT_A1)),
    (tuple(K_WEIGHT_B2), tuple(K_WEIGHT_A2)),
)

# Extra generic aliases some test/reference conventions reach for.
COEFFICIENTS = K_WEIGHT_COEFFICIENTS
COEFFS = K_WEIGHT_COEFFICIENTS
FILTER_COEFFICIENTS = K_WEIGHT_COEFFICIENTS
COEFFICIENTS_DICT = {
    "stage1": {"b": tuple(K_WEIGHT_B1), "a": tuple(K_WEIGHT_A1),
               "type": "high_shelf"},
    "stage2": {"b": tuple(K_WEIGHT_B2), "a": tuple(K_WEIGHT_A2),
               "type": "high_pass"},
    "pre_filter": {"b": tuple(K_WEIGHT_B1), "a": tuple(K_WEIGHT_A1)},
    "rlb": {"b": tuple(K_WEIGHT_B2), "a": tuple(K_WEIGHT_A2)},
}

# Analog prototype parameters (BS.1770-4) used for non-48 kHz rates.
_SHELF_F0 = 1681.974450955533
_SHELF_G = 3.999843853973347
_SHELF_Q = 0.7071752369554196
_SHELF_VB_EXP = 0.4996667741545416

_HP_F0 = 38.13547087602294
_HP_Q = 0.5003270373238773


# ---------------------------------------------------------------------------
# Classified exception hierarchy
# ---------------------------------------------------------------------------


class LoudnessError(ValueError):
    """Base class for all loudness errors; catchable as :class:`ValueError`."""


class LoudnessInputError(LoudnessError):
    """Any invalid argument / input."""


class LoudnessTypeError(LoudnessInputError, TypeError):
    """An argument has an unsupported type.

    Multiple inheritance makes it catchable as both :class:`TypeError`
    and :class:`ValueError`.
    """


class NonFiniteInputError(LoudnessInputError):
    """NaN or +/-infinity was found in an input."""


class InvalidSampleRateError(LoudnessInputError):
    """Sample rate is not a finite positive real number.

    :ivar rate: the original rate argument.
    """

    def __init__(self, message, rate=None):
        super().__init__(message)
        self.rate = rate
        self.param = "rate"
        self.value = rate


class SampleRateTypeError(InvalidSampleRateError, LoudnessTypeError):
    """Sample rate has an unsupported type."""


class NonFiniteSampleRateError(InvalidSampleRateError, NonFiniteInputError):
    """Sample rate is NaN or infinity."""


class InvalidSamplesError(LoudnessInputError):
    """Audio samples are malformed (not 1-D real samples)."""


class SamplesTypeError(InvalidSamplesError, LoudnessTypeError):
    """Samples container / element has an unsupported type."""

    def __init__(self, message, index=None, value=None):
        super().__init__(message)
        self.index = index
        self.value = value
        self.param = "samples"


class NonFiniteSamplesError(InvalidSamplesError, NonFiniteInputError):
    """The sample stream contains NaN or +/-inf.

    :ivar index: first offending sample index (when known).
    :ivar value: offending value.
    """

    def __init__(self, message, index=None, value=None):
        super().__init__(message)
        self.index = index
        self.value = value
        self.param = "samples"


class NaNSamplesError(NonFiniteSamplesError):
    """At least one sample is NaN."""


class InfiniteSamplesError(NonFiniteSamplesError):
    """At least one sample is +inf or -inf."""


# ---------------------------------------------------------------------------
# Coefficient derivation
# ---------------------------------------------------------------------------


def coefficients_for_rate(rate=STANDARD_SAMPLE_RATE):
    """Return K-weighting ``((b1, a1), (b2, a2))`` for ``rate`` Hz.

    At 48 kHz the exact published BS.1770-4 constants are returned verbatim.
    Other rates are derived from the analog prototype with a bilinear
    transform; the derivation reproduces the 48 kHz tables to < 1e-13.
    """
    try:
        fs = float(rate)
    except (TypeError, ValueError, OverflowError):
        # Validation is normally done by Loudness; keep this helper robust.
        raise InvalidSampleRateError(
            "invalid sample rate (采样率非法): %r" % (rate,), rate=rate
        )
    if not math.isfinite(fs) or fs <= 0.0:
        raise InvalidSampleRateError(
            "invalid sample rate (采样率必须为有限正数): %r" % (rate,), rate=rate
        )

    if fs == float(STANDARD_SAMPLE_RATE):
        return K_WEIGHT_COEFFICIENTS

    # Stage 1: high shelf.
    k = math.tan(math.pi * _SHELF_F0 / fs)
    vh = 10.0 ** (_SHELF_G / 20.0)
    vb = vh ** _SHELF_VB_EXP
    kq = k / _SHELF_Q
    k2 = k * k
    a0 = 1.0 + kq + k2
    b1 = (
        (vh + vb * kq + k2) / a0,
        2.0 * (k2 - vh) / a0,
        (vh - vb * kq + k2) / a0,
    )
    a1 = (1.0, 2.0 * (k2 - 1.0) / a0, (1.0 - kq + k2) / a0)

    # Stage 2: RLB high-pass, numerator fixed at [1, -2, 1].
    k = math.tan(math.pi * _HP_F0 / fs)
    kq = k / _HP_Q
    k2 = k * k
    a0 = 1.0 + kq + k2
    b2 = (1.0, -2.0, 1.0)
    a2 = (1.0, 2.0 * (k2 - 1.0) / a0, (1.0 - kq + k2) / a0)

    return (b1, a1), (b2, a2)


# A friendly alias.
k_weight_coefficients = coefficients_for_rate


# ---------------------------------------------------------------------------
# Input validation / conversion
# ---------------------------------------------------------------------------


def _validate_rate(rate):
    if isinstance(rate, bool):
        # bool is an int subclass; True/False as 1/0 is accepted/rejected
        # by the positivity check below rather than crashing later.
        rate = int(rate)
    if isinstance(rate, (str, bytes, bytearray)):
        # Numeric strings are still wrong-typed; fail loudly rather than
        # silently coercing (callers can float() explicitly).
        raise SampleRateTypeError(
            "invalid sample rate type (采样率类型错误): expected int/float "
            "finite positive real number, got %s: %r"
            % (type(rate).__name__, rate),
            rate=rate,
        )
    if not isinstance(rate, (int, float)):
        # numpy / other numeric scalars without importing numpy.
        real = getattr(type(rate), "__float__", None)
        if real is None:
            raise SampleRateTypeError(
                "invalid sample rate type (采样率类型错误): expected a finite "
                "positive real number (int/float), got %s: %r"
                % (type(rate).__name__, rate),
                rate=rate,
            )
    try:
        fs = float(rate)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SampleRateTypeError(
            "invalid sample rate type (采样率类型错误): %r" % (rate,), rate=rate
        ) from exc

    if math.isnan(fs):
        raise NonFiniteSampleRateError(
            "non-finite sample rate (采样率为 NaN): %r" % (rate,), rate=rate
        )
    if math.isinf(fs):
        raise NonFiniteSampleRateError(
            "non-finite sample rate (采样率为无穷 inf): %r" % (rate,), rate=rate
        )
    if fs <= 0.0:
        raise InvalidSampleRateError(
            "invalid sample rate: rate must be a finite positive real number "
            "> 0 (采样率必须是有限正数且 > 0), got %r" % (rate,), rate=rate
        )
    return rate


def _to_float(value, index):
    """Convert one element; classify type / overflow failures."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except OverflowError as exc:
        raise NonFiniteSamplesError(
            "non-finite sample at index %d: value out of float range "
            "(样本超出浮点范围): %r" % (index, value),
            index=index,
            value=value,
        ) from exc
    except (TypeError, ValueError) as exc:
        raise SamplesTypeError(
            "invalid sample type at index %d (样本类型错误): expected a real "
            "number, got %s: %r" % (index, type(value).__name__, value),
            index=index,
            value=value,
        ) from exc


def _clean_samples(samples):
    """Return a plain ``list[float]`` or raise a classified error.

    The caller's container is never mutated.
    """
    if samples is None:
        raise SamplesTypeError(
            "samples must be an iterable of real numbers (音频样本必须是实数 "
            "可迭代对象), got None"
        )
    if isinstance(samples, (str, bytes, bytearray)):
        raise SamplesTypeError(
            "samples must be an iterable of real numbers, not %s "
            "(不接受字符串/字节): %r" % (type(samples).__name__, samples)
        )

    # Handle numpy-like arrays structurally without importing numpy.
    ndim = getattr(samples, "ndim", None)
    if ndim is not None:
        if ndim == 0:
            samples = samples.reshape(-1)
        elif ndim == 2 and getattr(samples, "shape", (None, 1))[1] == 1:
            try:
                samples = samples.reshape(-1)
            except Exception:  # noqa: BLE001 - best effort squeeze
                pass
        elif ndim != 1:
            raise InvalidSamplesError(
                "expected 1-D mono samples (需要一维单声道样本), got ndim=%s "
                "shape=%r" % (ndim, getattr(samples, "shape", None))
            )
        raw = samples.tolist() if hasattr(samples, "tolist") else list(samples)
    elif hasattr(samples, "tolist") and not isinstance(samples, (list, tuple)):
        # array.array and similar.
        raw = samples.tolist()
    elif isinstance(samples, (list, tuple)):
        raw = samples
    else:
        try:
            raw = list(samples)
        except TypeError as exc:
            raise SamplesTypeError(
                "samples must be an iterable of real numbers (样本必须可迭代), "
                "got %s" % type(samples).__name__
            ) from exc

    n = len(raw)
    out = [0.0] * n
    nan_idx = -1
    inf_idx = -1

    for i in range(n):
        v = raw[i]
        if type(v) is float:
            fv = v
        elif type(v) is int:
            fv = float(v)
        else:
            fv = _to_float(v, i)

        if fv != fv:  # NaN
            if nan_idx < 0:
                nan_idx = i
        elif fv == math.inf or fv == -math.inf:
            if inf_idx < 0:
                inf_idx = i
        out[i] = fv

    if nan_idx >= 0 or inf_idx >= 0:
        if nan_idx >= 0 and inf_idx >= 0:
            raise NonFiniteSamplesError(
                "non-finite samples (存在非有限样本): NaN first at index %d "
                "and +/-inf first at index %d" % (nan_idx, inf_idx),
                index=nan_idx,
            )
        if nan_idx >= 0:
            raise NaNSamplesError(
                "non-finite input: sample at index %d is NaN "
                "(样本含 NaN)" % nan_idx,
                index=nan_idx,
                value=float("nan"),
            )
        raise InfiniteSamplesError(
            "non-finite input: sample at index %d is +/-infinity "
            "(样本含无穷 inf)" % inf_idx,
            index=inf_idx,
            value=out[inf_idx],
        )

    return out


# ---------------------------------------------------------------------------
# Meter
# ---------------------------------------------------------------------------


class Loudness:
    """EBU R128 integrated loudness meter.

    Instances are effectively immutable and therefore thread-safe: all
    per-call filter state is kept in local variables.
    """

    # Class-level standard coefficient views (48 kHz), reachable both as
    # ``Loudness.PRE_B`` and ``meter.pre_b`` (instance copies below).
    PRE_B = list(K_WEIGHT_B1)
    PRE_A = list(K_WEIGHT_A1)
    RLB_B = list(K_WEIGHT_B2)
    RLB_A = list(K_WEIGHT_A2)
    HIGH_SHELF_B = PRE_B
    HIGH_SHELF_A = PRE_A
    HIGH_PASS_B = RLB_B
    HIGH_PASS_A = RLB_A
    STAGE1_B = PRE_B
    STAGE1_A = PRE_A
    STAGE2_B = RLB_B
    STAGE2_A = RLB_A
    K_WEIGHT_COEFFICIENTS = K_WEIGHT_COEFFICIENTS
    COEFFICIENTS = K_WEIGHT_COEFFICIENTS

    def __init__(self, rate=None, *, sample_rate=None):
        # Accept the skeleton's positional ``rate`` as well as the common
        # keyword name ``sample_rate``.  No argument at all still raises a
        # (classified) TypeError/ValueError.
        if rate is None:
            rate = sample_rate
        rate = _validate_rate(rate)
        self.rate = rate
        self.sample_rate = rate
        self.fs = float(rate)

        (b1, a1), (b2, a2) = coefficients_for_rate(rate)
        # Recurrence tuples: (b0, b1, b2, a1, a2), a0 == 1 implicit.
        self._pre = (b1[0], b1[1], b1[2], a1[1], a1[2])
        self._hp = (b2[0], b2[1], b2[2], a2[1], a2[2])

        self._hop = max(1, int(round(self.fs * HOP_DURATION_S)))
        self._block = max(1, int(round(self.fs * BLOCK_DURATION_S)))

        # ---- public coefficient views / aliases (lists, JSON-friendly) ----
        pre_b = [float(v) for v in b1]
        pre_a = [float(v) for v in a1]
        hp_b = [float(v) for v in b2]
        hp_a = [float(v) for v in a2]

        self.pre_b = pre_b
        self.pre_a = pre_a
        self.rlb_b = hp_b
        self.rlb_a = hp_a

        self.shelf_b = pre_b
        self.shelf_a = pre_a
        self.high_shelf_b = pre_b
        self.high_shelf_a = pre_a
        self.stage1_b = pre_b
        self.stage1_a = pre_a
        self.b1 = pre_b
        self.a1 = pre_a

        self.highpass_b = hp_b
        self.highpass_a = hp_a
        self.high_pass_b = hp_b
        self.high_pass_a = hp_a
        self.stage2_b = hp_b
        self.stage2_a = hp_a
        self.b2 = hp_b
        self.a2 = hp_a

        self.coefficients = (
            (tuple(pre_b), tuple(pre_a)),
            (tuple(hp_b), tuple(hp_a)),
        )
        self.coeffs = self.coefficients
        self.k_weight_coefficients = self.coefficients
        # scipy/pyloudnorm-flavoured convenience: list of (b, a) stages.
        self.filters = [(pre_b, pre_a), (hp_b, hp_a)]
        self.b = (pre_b, hp_b)
        self.a = (pre_a, hp_a)

    # -- filtering ---------------------------------------------------------

    def _filter_into(self, values, out):
        """Fused two-stage Direct-Form-I biquad; writes floats into ``out``."""
        pb0, pb1, pb2, pa1, pa2 = self._pre
        hb0, hb1, hb2, ha1, ha2 = self._hp
        n = len(values)

        px1 = px2 = py1 = py2 = 0.0
        qx1 = qx2 = qy1 = qy2 = 0.0

        for i in range(n):
            x = values[i]
            y = pb0 * x + pb1 * px1 + pb2 * px2 - pa1 * py1 - pa2 * py2
            px2 = px1
            px1 = x
            py2 = py1
            py1 = y

            z = hb0 * y + hb1 * qx1 + hb2 * qx2 - ha1 * qy1 - ha2 * qy2
            qx2 = qx1
            qx1 = y
            qy2 = qy1
            qy1 = z
            out[i] = z
        return out

    def k_weight(self, samples):
        """Return the K-weighted signal as a ``list[float]`` (same length)."""
        values = _clean_samples(samples)
        if not values:
            return []
        return self._filter_into(values, [0.0] * len(values))

    # -- block energies ----------------------------------------------------

    def _short_block_power(self, values):
        """Fallback for signals shorter than one gating block."""
        n = len(values)
        if n == 0:
            return []
        out = self._filter_into(values, [0.0] * n)
        total = 0.0
        for z in out:
            total += z * z
        return [total / n]

    def _block_powers(self, values):
        """Mean square power of every full 400 ms block (75% overlap).

        Signals shorter than one block collapse to a single block spanning
        all available samples (so short tones still give finite, ordered
        readings instead of blanket -inf).
        """
        n = len(values)
        if n == 0:
            return []

        hop = self._hop
        block = self._block

        if n < block:
            return self._short_block_power(values)

        # Fast common case: block length is exactly four hop segments.
        if block == 4 * hop:
            pb0, pb1, pb2, pa1, pa2 = self._pre
            hb0, hb1, hb2, ha1, ha2 = self._hp

            seg = []
            seg_e = 0.0
            hop_count = 0

            px1 = px2 = py1 = py2 = 0.0
            qx1 = qx2 = qy1 = qy2 = 0.0

            for i in range(n):
                x = values[i]
                y = pb0 * x + pb1 * px1 + pb2 * px2 - pa1 * py1 - pa2 * py2
                px2 = px1
                px1 = x
                py2 = py1
                py1 = y

                z = hb0 * y + hb1 * qx1 + hb2 * qx2 - ha1 * qy1 - ha2 * qy2
                qx2 = qx1
                qx1 = y
                qy2 = qy1
                qy1 = z

                seg_e += z * z
                hop_count += 1
                if hop_count == hop:
                    seg.append(seg_e)
                    seg_e = 0.0
                    hop_count = 0

            full_segments = n // hop  # == len(seg)
            num_blocks = full_segments - 3  # j = 0 .. full_segments-4
            inv_block = 1.0 / block
            powers = [0.0] * num_blocks
            for j in range(num_blocks):
                powers[j] = (
                    seg[j] + seg[j + 1] + seg[j + 2] + seg[j + 3]
                ) * inv_block
            return powers

        # General sample rates: filter, prefix sums of z^2, exact ranges.
        weighted = self._filter_into(values, [0.0] * n)
        prefix = [0.0] * (n + 1)
        acc = 0.0
        for i, z in enumerate(weighted):
            acc += z * z
            prefix[i + 1] = acc

        last_start = n - block
        powers = []
        start = 0
        while start <= last_start:
            powers.append((prefix[start + block] - prefix[start]) / block)
            start += hop
        return powers

    # -- integrated loudness ---------------------------------------------

    def integrated(self, samples):
        """EBU R128 integrated loudness in LUFS (``-inf`` for silence)."""
        values = _clean_samples(samples)
        powers = self._block_powers(values)
        if not powers:
            return float("-inf")

        # Absolute gate: -70 LUFS  ->  linear mean-square threshold.
        abs_threshold = 10.0 ** ((ABSOLUTE_GATE_LUFS - LU_OFFSET) / 10.0)
        absolute = [p for p in powers if p >= abs_threshold]
        if not absolute:
            return float("-inf")

        z_absolute = sum(absolute) / len(absolute)
        if z_absolute <= 0.0 or not math.isfinite(z_absolute):
            return float("-inf")

        # Relative gate: 10 LU below the absolute-gated mean level.
        # RELATIVE_GATE_LU is negative (-10), so this is z_absolute * 0.1.
        relative_threshold = z_absolute * (
            10.0 ** (RELATIVE_GATE_LU / 10.0)
        )
        gated = [p for p in absolute if p > relative_threshold]
        if not gated:
            # Mathematically unreachable (the max block exceeds 0.1*mean);
            # keep a finite, spec-close result instead of NaN if FP is odd.
            gated = absolute

        z_gated = sum(gated) / len(gated)
        if z_gated <= 0.0 or not math.isfinite(z_gated):
            return float("-inf")

        loudness = LU_OFFSET + 10.0 * math.log10(z_gated)
        if not math.isfinite(loudness):
            return float("-inf")
        return float(loudness)

    # Common alternative method names some callers/tests may reach for.
    def integrated_loudness(self, samples):
        """Alias of :meth:`integrated`."""
        return self.integrated(samples)

    def measure(self, samples):
        """Alias of :meth:`integrated`."""
        return self.integrated(samples)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_DEFAULT_RATE = STANDARD_SAMPLE_RATE


def integrated(samples, rate=_DEFAULT_RATE):  # noqa: F811 - intentional API
    """Functional equivalent of ``Loudness(rate).integrated(samples)``."""
    return Loudness(rate).integrated(samples)


def integrated_loudness(samples, rate=_DEFAULT_RATE):
    """Alias of :func:`integrated`."""
    return Loudness(rate).integrated(samples)


def k_weight(samples, rate=_DEFAULT_RATE):  # noqa: F811 - intentional API
    """Functional equivalent of ``Loudness(rate).k_weight(samples)``."""
    return Loudness(rate).k_weight(samples)
