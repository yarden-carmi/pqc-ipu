"""ML-KEM polynomial kernels in NX8 (hardware format), on nx8lib.NB.

A polynomial is 2 lane-rows (coefficients 0..127, 128..255), each a list of limb rows (nx8lib.poly).
Every kernel is exact: all accumulations are integer-valued and below 2^24 (checked at build time by
the bound arguments), and every rounding that decides a result (floor/round in reduce, centre,
compress, decompress, digit split) is made exact either directly (quotient small enough that the FP32
accumulator and the 8-bit constant limbs resolve 1/(4q)) or by a correction step: approximate quotient
K0, exact residual r0 = y - q*K0 (split-stored), exact round(r0/q) (|r0/q| < 1), result K0 + that.

Quotients that only need to keep values bounded (NTT/INTT butterflies, basemul) use an approximate
quotient (error < 0.02), so outputs stay within q*(1/2 + 0.02) of the reduced value after a reducing
layer; values are checked mod q against mlkem-native.
"""
from __future__ import annotations

import math

import numpy as np

from mlkem_consts import Q, RINV, ZETAS, centered
from nx8lib import NB, QFU, DSONE, R, const_limbs, hi_bound, nlimbs, poly  # noqa: F401

EXACT_BUDGET = 1.0 / (4 * Q)
APPROX_BUDGET = 0.02
TWO24 = 2.0 ** 24


# ------------------------------------------------------------------ helpers
def _norm(c):
    c = np.asarray(c, np.float64)
    if c.ndim == 1 and np.all(c == c[0]):
        return np.float64(c[0])
    return c


def _bv(b):
    """A data bound as a per-lane vector (128)."""
    return np.broadcast_to(np.asarray(b, np.float64), (128,)).astype(np.float64)


def _eff(refs, bound):
    """Per-lane bound of sum over limbs of |limb| for |x| <= bound (vector) held in len(refs) limbs:
    the first limb may exceed |x| by half a grid step of its row (grid set by the row's largest value)."""
    b = _bv(bound)
    bm = float(b.max())
    g2 = 2.0 ** (math.floor(math.log2(bm)) - 7) if bm > 127 else 0.0
    return b + g2 + 127.0 * (len(refs) - 1)


def window(src, rho, shift):
    """(limb refs, rc offset) of p[j + shift] for the coefficients j of output lane-row rho."""
    if shift == 0:
        return src[rho], 0
    if abs(shift) == 128:
        o = rho + shift // 128
        return (src[o], 0) if 0 <= o <= 1 else (None, 0)
    assert abs(shift) < 128
    return src[rho], shift


def _lanes(c):
    return np.abs(np.asarray(c, np.float64)) * np.ones(128)


def plan_k(terms, bias, budget):
    """terms: (refs, shift, c/q-vector, bound). Constant limb tolerances and the error estimate
    (constant-limb residuals times data bounds, plus FP32 rounding of every add at the quotient's size)."""
    terms = [(r, s, _norm(c), b) for (r, s, c, b) in terms if r is not None and np.any(c)]
    nconst = len(terms) + (1 if bias else 0)
    kmax = float(np.max(sum((_bv(b) * _lanes(c) for (_, _, c, b) in terms), np.zeros(128)))) + abs(bias)
    plan, err, nmac = [], 0.0, 4
    for (refs, s, c, b) in terms:
        be = float(np.max(_eff(refs, b)))
        tol = 0.5 * budget / (nconst * be)
        limbs, res = const_limbs(c, tol)
        err += be * res
        nmac += len(limbs) * len(refs)
        plan.append((refs, s, c, tol))
    btol = None
    if bias:
        btol = 0.5 * budget / nconst
        limbs, res = const_limbs(np.float64(bias), btol)
        err += res
        nmac += len(limbs)
    err += nmac * 2.0 ** (math.floor(math.log2(max(kmax, 1.0))) - 24)
    return plan, btol, err, kmax


def emit_k(nb, plan, bias, btol):
    nb.begin()
    for (refs, s, c, tol) in plan:
        nb.term(refs, s, c, tol=tol)
    if bias:
        nb.const(bias, tol=btol)
    nb.magic()


def sum_bound(terms, vector=False):
    """Bound of sum |data * c| over the data terms of an exact pass (per lane, or its max)."""
    tot = np.zeros(128)
    for (r, s, c, b) in terms:
        if r is not None:
            tot = tot + _eff(r, b) * _lanes(c)
    return tot if vector else float(np.max(tot))


def emit_exact(nb, terms, K=None, kmax=0.0, sign_k=-1):
    """Accumulate sum of data * c (exact constants), then sign_k * q * K.
    Exactness: every partial sum of the data terms is bounded by sum_bound < 2^24 (asserted). The q*K
    tail is emitted last as 3328*K_h, K_h, 3328*K_l, ...: there the partial sums are the (small) final
    value plus q*|K_l| + |K_h|-sized pieces, and 3328*K_h is a multiple of 256 (exact up to 2^32)."""
    db = sum_bound(terms)
    assert db < TWO24, f"exact pass may exceed 2^24: {db:.4g}"
    assert Q * (kmax + 1) < 2 ** 25
    nb.begin()
    for (refs, s, c, b) in terms:
        if refs is not None and np.any(c):
            nb.term(refs, s, c)
    if K:
        nb.term(K, 0, float(Q), sign=sign_k)


class Ctx:
    """Shared scratch rows of one kernel (quotient limbs, residuals)."""

    def __init__(self, nb: NB):
        self.nb = nb
        self.K = nb.tmp(4)
        self.r0 = nb.tmp(2)
        self.corr = nb.tmp(1)


# ------------------------------------------------------------------ quotient / residue primitives
def residue(cx: Ctx, terms, kterms, dst, exact: bool, bias: float = 0.0, clean: bool = False):
    """dst <- y - q*K, y = sum terms (data*c, exact), K = round(sum kterms + bias) (kterms: data*c/q).
    exact: K is the exact rounding (Barrett-exact; output exactly in [-(q-1)/2, (q-1)/2] for bias 0).
    clean: magic-round the (integer) result before storing it. An all-zero NX8 row decodes as 2^-127;
    multiplied by large constants layer after layer that residue would grow (~2^10 per butterfly layer)
    and, over enough chained kernels, reach integer size. Rounding at every kernel output resets it.
    Returns a bound of |dst|."""
    nb = cx.nb
    store = (lambda d: (nb.magic(), nb.split_store(d))) if clean else nb.split_store
    budget = EXACT_BUDGET if exact else APPROX_BUDGET
    plan, btol, err, kmax = plan_k(kterms, bias, budget)
    if not kterms:
        emit_exact(nb, terms)
        store(dst)
        return sum_bound(terms, vector=True)
    if err < budget:
        emit_k(nb, plan, bias, btol)
        K = cx.K[:nlimbs(kmax + 1)]
        nb.split_store(K)
        emit_exact(nb, terms, K, kmax)
        store(dst)
        # per lane: |y - qK| <= sum |data*(c - q*cq)| + q*(1/2 + err) where a quotient term is present
        kmask = sum((_lanes(cq) for (r, sh, cq, b) in kterms if r is not None), np.zeros(128)) > 0
        rest = _rest_bound(terms, kterms)
        return rest + kmask * Q * (0.5 + err + abs(bias))
    assert exact, f"approximate quotient error {err} over budget"
    # corrected: K0 approximate, r0 = y - q*K0 exact, corr = round(r0/q + bias) exact, out = r0 - q*corr
    plan, btol0, err0, kmax = plan_k(kterms, 0.0, APPROX_BUDGET)
    assert err0 < APPROX_BUDGET
    emit_k(nb, plan, 0.0, btol0)
    K = cx.K[:nlimbs(kmax + 1)]
    nb.split_store(K)
    emit_exact(nb, terms, K, kmax)
    rb = float(np.max(_rest_bound(terms, kterms))) + Q * (0.5 + err0)
    r0 = cx.r0[:nlimbs(rb)]
    nb.split_store(r0)
    plan, btol, err, cm = plan_k([(r0, 0, 1.0 / Q, rb)], bias, EXACT_BUDGET)
    assert err < EXACT_BUDGET and cm <= 126
    emit_k(nb, plan, bias, btol)
    nb.store(cx.corr[0])
    emit_exact(nb, [(r0, 0, 1.0, rb)], cx.corr, cm)
    store(dst)
    return _bv(Q * (0.5 + abs(bias)) + 1)


def quotient(cx: Ctx, terms, kterms, bias: float = 0.0):
    """Leave the exact c = round(y/q + bias) in the accumulator (y = sum terms, kterms = terms/q).
    Returns |c| bound."""
    nb = cx.nb
    plan, btol, err, kmax = plan_k(kterms, bias, EXACT_BUDGET)
    if err < EXACT_BUDGET:
        emit_k(nb, plan, bias, btol)
        return kmax + 1
    plan, btol0, err0, kmax = plan_k(kterms, 0.0, APPROX_BUDGET)
    assert err0 < APPROX_BUDGET
    emit_k(nb, plan, 0.0, btol0)
    K = cx.K[:nlimbs(kmax + 1)]
    nb.split_store(K)
    emit_exact(nb, terms, K, kmax)
    rb = float(np.max(_rest_bound(terms, kterms))) + Q * (0.5 + err0)
    r0 = cx.r0[:nlimbs(rb)]
    nb.split_store(r0)
    plan, btol, err, cm = plan_k([(r0, 0, 1.0 / Q, rb)], bias, EXACT_BUDGET)
    assert err < EXACT_BUDGET
    emit_k(nb, plan, bias, btol)          # acc = round(r0/q + bias)
    nb.term(K, 0, 1.0)                    # + K0 (exact integers): acc = c
    return kmax + 2


def _rest_bound(terms, kterms):
    """Per lane: sum |data| * |c - q*cq|, the part of y the quotient does not remove."""
    kq = {}
    for (r, s, cq, b) in kterms:
        if r is not None:
            kq[(id(r), s)] = np.asarray(cq, np.float64) * np.ones(128)
    tot = np.zeros(128)
    for (r, s, c, b) in terms:
        if r is None:
            continue
        c = np.asarray(c, np.float64) * np.ones(128)
        tot = tot + _eff(r, b) * np.abs(c - Q * kq.get((id(r), s), np.zeros(128)))
    return tot


# ------------------------------------------------------------------ generic butterfly layer
def _coef_bound(B, rho, shift):
    """Per-lane bound of p[j + shift] for output lane-row rho, from per-coefficient bounds B (256)."""
    if abs(shift) == 128:
        o = rho + shift // 128
        return B[128 * o: 128 * o + 128] if 0 <= o <= 1 else np.zeros(128)
    i = np.arange(128) + shift
    ok = (i >= 0) & (i < 128)
    return np.where(ok, B[128 * rho + np.clip(i, 0, 127)], 0.0)


def layer(cx: Ctx, src, dst, ln: int, row_vecs, bound, exact: bool = False, clean: bool = False):
    """One radix-2 layer: out[j] = p[j]A + p[j+ln]B + p[j-ln]C - q*K, K = round(p[j]Aq + p[j+ln]Bq + p[j-ln]Cq).
    bound: per-coefficient bounds of the input (256, or a scalar); returns those of the output."""
    shifts = {"A": 0, "B": ln, "C": -ln}
    B = np.broadcast_to(np.asarray(bound, np.float64), (256,))
    ob = []
    for rho in (0, 1):
        v = row_vecs[rho]
        terms, kterms = [], []
        for key in ("A", "B", "C"):
            refs, off = window(src, rho, shifts[key])
            if refs is None:
                continue
            bl = _coef_bound(B, rho, shifts[key])
            c = v.get(key)
            if c is not None and np.any(c):
                terms.append((refs, off, c, bl))
            cq = v.get(key + "q")
            if cq is not None and np.any(cq):
                kterms.append((refs, off, cq, bl))
        ob.append(_bv(residue(cx, terms, kterms, dst[rho], exact, clean=clean)))
    ob = np.concatenate(ob)
    assert ob.max() < 2 ** 14 - 1, f"layer output bound {ob.max()} needs more than 2 limbs"
    return ob


def ntt_vecs(layer_: int, full: bool):
    ln = 256 >> layer_
    out = []
    for rho in (0, 1):
        j = 128 * rho + np.arange(128)
        top = (j % (2 * ln)) < ln
        z = np.array([ZETAS[(1 << (layer_ - 1)) + g] for g in j // (2 * ln)], dtype=np.float64)
        A = np.where(top, 1.0, -z)
        B = np.where(top, z, 0.0)
        C = np.where(top, 0.0, 1.0)
        Aq = A / Q if full else np.where(top, 0.0, -z / Q)
        out.append(dict(A=A, B=B, C=C, Aq=Aq, Bq=B / Q, Cq=(C / Q if full else None)))
    return out


def intt_vecs(layer_: int, scale: int, full: bool):
    ln = 256 >> layer_
    rows = []
    for rho in (0, 1):
        j = 128 * rho + np.arange(128)
        top = (j % (2 * ln)) < ln
        z = np.array([ZETAS[(1 << layer_) - 1 - g] for g in j // (2 * ln)], dtype=np.float64)
        s = int(centered(scale)) if layer_ == 1 else 1
        zs = np.array([int(centered(int(x) * s)) for x in z], dtype=np.float64)
        A = np.where(top, s, zs)
        B = np.where(top, s, 0.0)
        C = np.where(top, 0.0, -zs)
        if full:
            Aq, Bq = A / Q, B / Q
        else:
            Aq, Bq = np.where(top, 0.0, zs / Q), None
        rows.append(dict(A=A, B=B, C=C, Aq=Aq, Bq=Bq, Cq=C / Q))
    return rows


NTT_FULL = (1, 4, 7)
INTT_FULL = (7, 5, 3, 1)
F_INTT = 512        # 2^16/128 mod q: mlkem-native invntt_tomont
INV128 = pow(128, -1, Q)


def ntt(scratch: int, src, dst, n: int = 1, stride: int = 0, bound: float = Q) -> dict:
    """Forward NTT of n polys (looped, rows advance by stride per poly). src/dst: nx8lib.poly(...)."""
    nb = NB(scratch, loop=n)
    cx = Ctx(nb)
    t = [poly(nb.alloc(4)), poly(nb.alloc(4))]
    cur, b = src, bound
    for lay in range(1, 8):
        nxt = dst if lay == 7 else t[lay % 2]
        b = layer(cx, cur, nxt, 256 >> lay, ntt_vecs(lay, lay in NTT_FULL), b, clean=(lay == 7))
        cur = nxt
    k = nb.kernel()
    k["out_bound"] = float(np.max(b))
    return k


def intt(scratch: int, src, dst, n: int = 1, scale: int = F_INTT, bound: float = Q) -> dict:
    nb = NB(scratch, loop=n)
    cx = Ctx(nb)
    t = [poly(nb.alloc(4)), poly(nb.alloc(4))]
    cur, b = src, bound
    for i, lay in enumerate(range(7, 0, -1)):
        nxt = dst if lay == 1 else t[i % 2]
        b = layer(cx, cur, nxt, 256 >> lay, intt_vecs(lay, scale, lay in INTT_FULL), b, clean=(lay == 1))
        cur = nxt
    k = nb.kernel()
    k["out_bound"] = float(np.max(b))
    return k


# ------------------------------------------------------------------ pointwise reductions
def scale_reduce(scratch: int, src, dst, c: int, n: int = 1, bound: float = 2 ** 15, exact: bool = True) -> dict:
    """dst = x*c mod q, centred (exact=True: exactly in [-(q-1)/2, (q-1)/2], i.e. Barrett for c = 1).
    c = 1: mlk_barrett_reduce / centre; c = 2^16 mod q: poly_tomont (mod q)."""
    nb = NB(scratch, loop=n)
    cx = Ctx(nb)
    c = int(centered(c))
    for rho in (0, 1):
        residue(cx, [(src[rho], 0, float(c), bound)], [(src[rho], 0, c / Q, bound)], dst[rho], exact, clean=True)
    return nb.kernel()


def reduce_canonical(scratch: int, src, dst, n: int = 1, bound: float = 2 ** 15) -> dict:
    """poly_reduce: dst = x - q*floor(x/q) in [0, q), floor = round(x/q - (q-1)/(2q)) (never a tie)."""
    nb = NB(scratch, loop=n)
    cx = Ctx(nb)
    for rho in (0, 1):
        residue(cx, [(src[rho], 0, 1.0, bound)], [(src[rho], 0, 1.0 / Q, bound)], dst[rho], True,
                bias=-(Q - 1) / (2 * Q))
    return nb.kernel()


def add_sub(scratch: int, triples, sub: bool = False, bound: float = 2 ** 13) -> dict:
    """out = x + y (or x - y), no reduction (poly_add / poly_sub). triples: (x, y, out) polys."""
    nb = NB(scratch)
    for x, y, o in triples:
        for rho in (0, 1):
            terms = [(x[rho], 0, 1.0, bound), (y[rho], 0, -1.0 if sub else 1.0, bound)]
            emit_exact(nb, terms)
            nb.split_store(o[rho])
    return nb.kernel()


# ------------------------------------------------------------------ base multiplication
EVEN = np.arange(128) % 2 == 0
ODD = ~EVEN


def gammas():
    return np.array([ZETAS[64 + p // 2] * (1 if p % 2 == 0 else -1) for p in range(128)], dtype=np.float64)


def mulcache(scratch: int, src, v1, v2, n: int = 1, montgomery: bool = True, bound: float = Q) -> dict:
    """V1 = [even j: b[j], odd j: b[j-1]] * s, V2 = [even: gamma*b[j+1], odd: b[j]] * s, s = R^-1 or 1,
    both reduced exactly (centred, |V| <= (q-1)/2) so that basemul_acc's sums stay below 2^24."""
    nb = NB(scratch, loop=n)
    cx = Ctx(nb)
    sc = RINV if montgomery else 1
    s = float(int(centered(sc)))
    g = gammas()
    for rho in (0, 1):
        gs = np.array([int(centered(int(x) * sc)) for x in g[64 * rho: 64 * rho + 64]], dtype=np.float64)
        gl = np.repeat(gs, 2)
        A1, C1 = np.where(EVEN, s, 0.0), np.where(ODD, s, 0.0)
        B2, A2 = np.where(EVEN, gl, 0.0), np.where(ODD, s, 0.0)
        for (dst, vec) in ((v1, dict(A=A1, C=C1)), (v2, dict(A=A2, B=B2))):
            terms, kterms = [], []
            for key, sh in (("A", 0), ("B", 1), ("C", -1)):
                if key in vec:
                    terms.append((src[rho], sh, vec[key], bound))
                    kterms.append((src[rho], sh, vec[key] / Q, bound))
            residue(cx, terms, kterms, dst[rho], True, clean=True)
    return nb.kernel()


def basemul_acc(scratch: int, out, terms, n: int = 1, montgomery_fold: bool = False) -> dict:
    """out = sum_k a_k o b_k (base multiplication in the NTT domain), b_k given by its mulcache (V1, V2).
    terms: list of (a, V1, V2) polys; a and V must be exactly centred (|.| <= 1664) so that the exact
    accumulation T = sum_k (a V1 + a' V2) stays below 2^24 (6*(1664+8)^2 < 2^24). T is split into 4 limbs
    and reduced with an approximate quotient (|out| <= q*0.52)."""
    nb = NB(scratch, loop=n)
    cx = Ctx(nb)
    nb.register_masks([EVEN, ODD])
    T = nb.tmp(4)
    B = 1664.0
    # |x| <= 1664 -> row grid <= 16 and 1664 = 104*16, so |hi| <= 1664, |lo| <= 8: every partial sum of
    # the 2k limb products is below 6*(1664+8)^2 = 16,773,504 < 2^24 (k = 3)
    assert B % 16 == 0 and len(terms) * 2 * (B + 8) ** 2 < 2 ** 24
    for rho in (0, 1):
        nb.begin()
        for (a, V1, V2) in terms:
            for vb in V1[rho]:
                for ar in a[rho]:
                    nb.term_vv(ar, 0, vb)
            for vb in V2[rho]:
                for ar in a[rho]:
                    nb.term_vv(ar, 1, vb, mask=EVEN)
                    nb.term_vv(ar, -1, vb, mask=ODD)
        tb = len(terms) * 2 * B * B
        nb.split_store(T[:nlimbs(tb)])
        Tl = T[:nlimbs(tb)]
        # (prefix sums of T's limbs are exact: T_1 is a multiple of 2^17, T_1+T_2 of 2^10, ...)
        residue(cx, [(Tl, 0, 1.0, tb)], [(Tl, 0, 1.0 / Q, tb)], out[rho], False, clean=True)
    return nb.kernel()


# ------------------------------------------------------------------ compress / decompress / CBD / digits
def compress(scratch: int, src, dst, d: int, n: int = 1, out_qf=None) -> dict:
    """c = round(x * 2^d / q) mod 2^d for canonical x (mlk_scalar_compress_d). dst limbs: 1 (d <= 6) or 2.
    The wrap (c == 2^d -> 0) uses the store's relu: w = relu(c - (2^d - 1)) in {0, 1}; c - 2^d w."""
    nb = NB(scratch, loop=n)
    cx = Ctx(nb)
    w = nb.tmp(1)
    for rho in (0, 1):
        x = src[rho]
        quotient(cx, [(x, 0, float(2 ** d), Q)], [(x, 0, 2.0 ** d / Q, Q)])
        nb.const(2.0 ** d - 1, -1)
        nb.store(w[0], act="relu")
        nb.const(2.0 ** d - 1, +1)
        nb.term(w, 0, 2.0 ** d, sign=-1)
        if out_qf:
            nb.store(dst[rho][0], st_qf=out_qf)
        else:
            nb.split_store(dst[rho])
    return nb.kernel()


def decompress(scratch: int, src, dst, d: int, n: int = 1, src_digits: bool = False, src_qf=None) -> dict:
    """x = floor(c*q/2^d + 1/2) (round half up), exact: c*q/2^d + 2^-(d+1) is exact in FP32 and never a
    tie, then magic rounding. src: c as limbs (poly) or, src_digits, as (lo8 U8 row, hi row) per lane-row."""
    nb = NB(scratch, loop=n)
    for rho in (0, 1):
        nb.begin()
        if src_digits:
            lo, hi = src[rho]
            nb.term([lo], 0, Q / 2.0 ** d, qf=QFU)
            nb.term([hi], 0, 256 * Q / 2.0 ** d)
        else:
            nb.term(src[rho], 0, Q / 2.0 ** d, qf=src_qf)
        nb.const(2.0 ** -(d + 1))
        nb.magic()
        nb.split_store(dst[rho])
    return nb.kernel()


def cbd(scratch: int, planes, dst, eta: int = 2, n: int = 1) -> dict:
    """Centred binomial from bit planes: x_j = sum_{t<eta} b[2*eta*j+t] - sum_{t<eta} b[2*eta*j+eta+t].
    planes[rho][t]: the row of bit t of coefficients 128*rho..128*rho+127 (host bit reordering)."""
    nb = NB(scratch, loop=n)
    for rho in (0, 1):
        nb.begin()
        for t in range(2 * eta):
            nb.term([planes[rho][t]], 0, 1.0, sign=1 if t < eta else -1)
        nb.store(dst[rho][0])
    return nb.kernel()


def from_digits(scratch: int, digits, dst, n: int = 1, centre: bool = True, hi_mult: int = 256) -> dict:
    """x = lo8 + 256*hi from host byte digits (lo8: U8 row 0..255, hi: small signed row); centre: exact
    x - q*round(x/q) (|.| <= 1664) for canonical x."""
    nb = NB(scratch, loop=n)
    cx = Ctx(nb)
    for rho in (0, 1):
        lo, hi = digits[rho]
        bmax = 255 + hi_mult * 15
        if centre:
            plan, btol, err, kmax = plan_k([([lo], 0, 1.0 / Q, 255), ([hi], 0, hi_mult / Q, 15)], 0.0, EXACT_BUDGET)
            assert err < EXACT_BUDGET
            nb.begin()
            nb.term([lo], 0, 1.0 / Q, tol=plan[0][3], qf=QFU)
            nb.term([hi], 0, hi_mult / Q, tol=plan[1][3])
            nb.magic()
            nb.store(cx.K[0])
            nb.begin()
            nb.term([lo], 0, 1.0, qf=QFU)
            nb.term([hi], 0, float(hi_mult))
            nb.term(cx.K[:1], 0, float(Q), sign=-1)
        else:
            nb.begin()
            nb.term([lo], 0, 1.0, qf=QFU)
            nb.term([hi], 0, float(hi_mult))
        assert bmax + Q * 2 < 2 ** 24
        nb.split_store(dst[rho])
    return nb.kernel()


def to_digits(scratch: int, src, digits, n: int = 1, bound: float = 4095) -> dict:
    """Canonical x in [0, 4096) -> hi = floor(x/256) (signed row), lo8 = x - 256*hi (U8 row 0..255).
    floor(y) = round(y - 1/2 + 1/512) is exact (y = x/256 is exact, never a tie)."""
    nb = NB(scratch, loop=n)
    for rho in (0, 1):
        lo, hi = digits[rho]
        nb.begin()
        nb.term(src[rho], 0, 1.0 / 256)
        nb.const(-0.5)
        nb.const(2.0 ** -9)
        nb.magic()
        nb.store(hi)
        nb.begin()
        nb.term(src[rho], 0, 1.0)
        nb.term([hi], 0, 256.0, sign=-1)
        nb.store(lo, st_qf=QFU)
    return nb.kernel()


# ------------------------------------------------------------------ FO: ciphertext compare + key select
def compare_select(scratch: int, pairs, kbits, kbar, out, kqf=None) -> dict:
    """eq = 1 iff every pair (c', c) is equal; out = Kbar + eq*(K - Kbar) on bit rows.
    pairs: list of (c' limb refs, c-ref description) per lane-row: c' a limb list, c either a limb list or
    ('digits', lo8, hi). Per lane: p = relu6(1 - d), m = relu6(1 + d), eq_lane = p*m (1 iff d = 0);
    S = sum of eq_lane over all lanes (AGG.SUM), eq = relu(S - N + 1) stored from lane 0 only (DS1)."""
    nb = NB(scratch)
    P, M, E = nb.tmp(1)[0], nb.tmp(1)[0], nb.tmp(1)[0]
    pm = []
    for i, (cp, c) in enumerate(pairs):
        p_row, m_row = nb.tmp(1)[0], nb.tmp(1)[0]
        for row, sgn in ((p_row, 1), (m_row, -1)):
            nb.begin()
            nb.const(1.0)
            nb.term(cp, 0, 1.0, sign=-sgn)
            if c[0] == "digits":
                nb.term([c[1]], 0, 1.0, sign=sgn, qf=QFU)
                nb.term([c[2]], 0, 256.0, sign=sgn)
            else:
                nb.term(c, 0, 1.0, sign=sgn)
            nb.store(row, act="relu6")
        pm.append((p_row, m_row))
    nb.begin()
    for i, (p_row, m_row) in enumerate(pm):
        base = nb.slot(p_row)
        ra = nb.rreg(m_row)
        m = nb.p.mac(acc="ACC.ADD", kind="RCVV", rc=base, ra=ra, mask=0,
                     agg="AGG.SUM.FIRST" if i == 0 else "AGG.SUM")
        nb.last = m
    N = 128 * len(pm)
    nb.first = False
    nb.const(float(N - 1), -1)
    nb.store(E, act="relu", ds=DSONE, st_qf=kqf)
    for rho in (0, 1):
        nb.begin()
        nb.term([kbar[rho]], 0, 1.0, qf=kqf)
        ra = nb.rreg(E)
        ridx = 0 if ra == "R0" else 128
        b1 = nb.slot(kbits[rho])
        nb.mac(1, kind="RCVE", rc=b1, ridx=ridx, mask=0, qf=kqf)
        b2 = nb.slot(kbar[rho])
        nb.mac(-1, kind="RCVE", rc=b2, ridx=ridx, mask=0, qf=kqf)
        nb.store(out[rho], st_qf=kqf)
    return nb.kernel()
