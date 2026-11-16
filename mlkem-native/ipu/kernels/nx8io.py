"""NX8 kernels for everything ML-KEM does besides polynomial arithmetic: byte/bit/field conversions, sponge
input assembly, absorb/squeeze on bitstream rows, CBD from PRF bits, and SampleNTT with a fixed schedule.

Formats on the IPU:
  bytes      U8 rows, byte k at row k//128, lane k%128 (the run's inputs and outputs);
  bitstream  E4 rows of 0/1, bit j at row j//128, lane j%128 (bits LSB-first in each byte: FIPS 203 BytesToBits);
  fields     poly limbs (nx8lib.poly), coefficient i at lane-row i//128, lane i%128;
  Keccak     15 S8 state rows in difference form (keccak.py).
E4 (E4M3, 15 binades) holds the bits and the power-of-two weight rows 1..2^11 exactly (S8 cannot hold 1 and 2^11
in one row), so the bit-to-field AGG products are exact.

Every kernel's schedule is input-independent: addresses come from CRs/LRs only; values select by arithmetic.
"""
from __future__ import annotations

import math

import numpy as np

from nx8lib import NB, QFE, QFU, R, U8, E4, nlimbs
from mlkem_consts import Q

LOW = np.arange(128) < 64
HIGH = ~LOW
M2 = 1.5 * 2 ** 24        # fl(x + M2) - M2 rounds x to an even integer (|x| < 2^23)


def _srow(state_base, L):
    x, y = L % 5, L // 5
    return state_base + 3 * x + y // 2, y % 2


# ------------------------------------------------------------------ sponge kernels
def zero_rows(scratch, rows, qf_e4=False):
    nb = NB(scratch)
    nb.begin()
    for r in rows:
        nb.nop_mac()
        nb.first = True
        nb.store(R(r), st_qf=QFE if qf_e4 else None)
    return nb.kernel()


def absorb(scratch, state_base, lanes):
    """S <- S^2 - M on the rate lanes (A xor M, difference form). lanes: {L: (bit row ref, half) or 64-bit
    constant array}; constants become preloaded E4 rows. Missing lanes are zero."""
    nb = NB(scratch)
    nb.register_masks([LOW, HIGH])
    src = {}
    for L, v in lanes.items():
        if isinstance(v, tuple):
            src[L] = v
        else:
            bits = np.asarray(v, np.float64)
            if not bits.any():
                continue
            row = np.zeros(128)
            row[:64] = bits
            src[L] = (nb.vrow(row, E4), 0)
    for x in range(5):
        for p in range(3):
            halves = [h for h in (0, 1) if x + 5 * (2 * p + h) in src]
            if not halves:
                continue
            nb.begin()
            nb.term_vs(R(state_base + 3 * x + p), 0)
            for h in halves:
                ref, hs = src[x + 5 * (2 * p + h)]
                nb.term([ref], 64 * hs - 64 * h, 1.0, sign=-1, mask=LOW if h == 0 else HIGH, qf=QFE)
            nb.store(R(state_base + 3 * x + p))
    return nb.kernel()


def extract(scratch, state_base, outs):
    """Squeeze to bitstream rows: outs = [(out ref, {half: state lane L}, keep_other_half)]; out = S^2 (bits)."""
    nb = NB(scratch)
    nb.register_masks([LOW, HIGH])
    for out, halves, keep in outs:
        nb.begin()
        for h, L in halves.items():
            row, hs = _srow(state_base, L)
            nb.term_vs(R(row), 64 * hs - 64 * h, mask=LOW if h == 0 else HIGH)
        if keep:
            other = [h for h in (0, 1) if h not in halves]
            for h in other:
                nb.term([out], 0, 1.0, mask=LOW if h == 0 else HIGH, qf=QFE)
        nb.store(out, st_qf=QFE)
    return nb.kernel()


class SpongePlan:
    """Fixed-length SHA3/SHAKE: message = segments [('rows', [refs]) | ('bytes', b'...')]; data segments are
    whole 128-bit rows, so every 64-bit lane of the padded message is either a data row half or a constant."""

    def __init__(self, segments, rate_bytes, ds, outlen):
        self.rate = rate_bytes
        lanes = []   # per message lane: (ref, half) or 64 bits
        buf = bytearray()

        def flush_bytes():
            while len(buf) >= 8:
                lanes.append(np.unpackbits(np.frombuffer(bytes(buf[:8]), np.uint8), bitorder="little").astype(float))
                del buf[:8]
        for kind, v in segments:
            if kind == "rows":
                assert not buf, "data segments must start on a 64-bit lane"
                for ref in v:
                    lanes += [(ref, 0), (ref, 1)]
            else:
                buf += v
                flush_bytes()
        nbytes = 8 * len(lanes) + len(buf)
        buf.append(ds)
        total = -(-(nbytes + 1) // rate_bytes) * rate_bytes
        while (8 * len(lanes) + len(buf)) < total:
            buf.append(0)
        flush_bytes()
        last = lanes[-1]
        assert not isinstance(last, tuple)
        last = last.copy()
        last[63] = 1.0                       # the final 0x80 bit
        lanes[-1] = last
        rl = rate_bytes // 8
        self.blocks = [dict(enumerate(lanes[i: i + rl])) for i in range(0, len(lanes), rl)]
        self.outlen = outlen
        self.nsq = -(-outlen // rate_bytes)


def sponge_specs(plan, state_base, out_base):
    """The kernel sequence of one hash as specs: ('zero', rows), ('absorb', lanes), ('permute', None),
    ('extract', outs). Equal specs build equal kernels, so callers can share them."""
    specs = [("zero", tuple(state_base + i for i in range(15)))]
    for blk in plan.blocks:
        specs.append(("absorb", blk))
        specs.append(("permute", None))
    rl = plan.rate // 8
    need_lanes = -(-plan.outlen // 8)
    G = 0
    for sb in range(plan.nsq):
        if sb:
            specs.append(("permute", None))
        n = min(rl, need_lanes - G)
        rows = {}
        for L in range(n):
            g = G + L
            rows.setdefault(g // 2, {})[g % 2] = L
        outs = tuple((R(out_base + r), tuple(sorted(h.items())), len(h) == 1 and 0 not in h and sb > 0)
                     for r, h in sorted(rows.items()))
        specs.append(("extract", outs))
        G += n
    return specs


def build_spec(scratch, state_base, spec):
    kind, arg = spec
    if kind == "zero":
        return zero_rows(scratch, arg)
    if kind == "absorb":
        return absorb(scratch, state_base, arg)
    if kind == "extract":
        return extract(scratch, state_base, [(o, dict(h), k) for (o, h, k) in arg])
    raise ValueError(kind)


def spec_key(spec):
    kind, arg = spec
    if kind == "absorb":
        return (kind, tuple((L, v if isinstance(v, tuple) else tuple(np.asarray(v).tolist())) for L, v in sorted(arg.items())))
    return (kind, arg)


# ------------------------------------------------------------------ bits -> fields (AGG)
def bits_to_fields(scratch, bit_base, d, n_out_rows, out_refs, weights, out_kind="limbs"):
    """Field f (f = 0..128*n_out_rows-1) = sum_t weights[t] * bit[d*f + t], one AGG.SUM per field.
    Bit rows are loaded into R_CYCLIC slot (k mod 4) in order, so a field straddling two rows reads one
    contiguous window. out_refs: per output row (of one loop body) the limb refs; out_kind 'limbs' (split
    store), 'u8' (bytes), 's8' (one small row). Looped: each iteration handles n_body output rows."""
    n_body = 1
    while (n_body * d) % 4:
        n_body += 1
    n_iter = -(-n_out_rows // n_body)
    rows_body = n_body * d
    nb = NB(scratch, loop=n_iter)
    w = np.zeros(128)
    w[:len(weights)] = weights
    nb.set_r(0, nb.vrow(w, E4))
    loaded = -1
    for f in range(128 * n_body):
        lastrow = (d * f + d - 1) // 128
        while loaded < lastrow:
            loaded += 1
            nb.load_slot(loaded % 4, R(bit_base + loaded, rows_body))
        nb.agg_vv((d * f) % 512, "R0", f % 128, first=True, qf=QFE)
        if f % 128 == 127:
            refs = out_refs[f // 128]
            if out_kind == "limbs":
                nb.split_store(refs)
                # the split store's reload may have taken a slot: re-place the bit rows still needed
                for r in range((d * (f + 1)) // 128, loaded + 1):
                    nb.load_slot(r % 4, R(bit_base + r, rows_body))
            elif out_kind == "u8":
                nb.store(refs[0], st_qf=QFU)
            else:
                nb.store(refs[0])
    k = nb.kernel()
    k["n_iter"] = n_iter
    return k


# ------------------------------------------------------------------ floor-mod-2: bits of a replicated row
def _bits_from_rep(nb, reps, dst):
    """dst <- floor(y) mod 2, y = sum rep*w (U8 rows, y on a 2^-7 grid, |y| < 2^22), exact:
    -2*floor(y/2) = round_even2(-y + 1 - 2^-9); r = y - 2 floor(y/2); bit = round(r - 1/2 + 2^-9)."""
    def ysum(sign):
        for rep, wref in reps:
            base = nb.slot(rep)
            ra = nb.rreg(wref)
            nb.mac(sign, kind="RCVV", rc=base, ra=ra, mask=0, qf=QFU)
    nb.begin()
    ysum(-1)
    nb.const(1.0 - 2.0 ** -9)
    nb.const(M2, 1)
    nb.const(M2, -1)
    ysum(1)
    nb.const(-0.5 + 2.0 ** -9)
    nb.magic()
    nb.store(dst, st_qf=QFE)


def bytes_to_bits(scratch, byte_base, n_byte_rows, bit_base):
    """Byte rows (U8) -> bitstream rows: for each 16 bytes, MULT.EE broadcasts byte g to lanes 8g..8g+7 (16
    masked EE ops build a replicated row), then bit = floor(byte * 2^-(lane%8)) mod 2. Loop over byte rows."""
    nb = NB(scratch, loop=n_byte_rows)
    gm = [(np.arange(128) // 8) == g for g in range(16)]
    nb.register_masks(gm)
    w8 = nb.vrow(2.0 ** -(np.arange(128) % 8), U8)
    reps = [R(nb.alloc()), R(nb.alloc())]
    for q in range(8):
        nb.set_r(0, R(byte_base, 1))
        nb.begin()
        for g in range(16):
            nb.ee(16 * q + g, 1.0, mask=gm[g], qf=QFU)
        rep = reps[q % 2]
        nb.store(rep, st_qf=QFU)
        nb.rr[1] = None
        _bits_from_rep(nb, [(rep, w8)], R(bit_base + q, 8))
    return nb.kernel()


def fields_to_bits(scratch, digits, d, bit_base):
    """One poly's d-bit fields (given as byte digits: per lane-row (lo8 U8 ref, hi ref) for d > 8, or one small
    S8 ref for d <= 8) -> 2d bitstream rows (FIPS 203 ByteEncode_d's bit order). Per bit row: masked MULT.EE
    broadcasts of the fields covering it build replicated rows, then floor-mod-2 with per-lane 2^-t weights.
    Looped over the row period P (lcm(128, d)/128 rows); the field index advances by 128P/d per iteration."""
    P = math.lcm(128, d) // 128
    nf = 128 * P // d
    nb = NB(scratch, loop=2 * d // P)
    two = d > 8
    masks, wl, wh = {}, {}, {}
    for q in range(P):
        j = 128 * q + np.arange(128)
        t = j % d
        fi = j // d
        for i in sorted(set(fi)):
            masks[(q, i)] = fi == i
        wl[q] = np.where(t < 8, 2.0 ** -np.minimum(t, 7), 0.0)
        wh[q] = np.where(t >= 8, 2.0 ** -np.maximum(t - 8, 0), 0.0)
    nb.register_masks(list(masks.values()))
    reps = [R(nb.alloc()) for _ in range(4)]
    for q in range(P):
        parts = [(0, QFU)] + ([(1, None)] if two else [])
        rr = []
        for k, (part, qf) in enumerate(parts):
            for rho in (0, 1):
                src = digits[rho][part] if two else digits[rho]
                nb.set_r(rho, src)
            nb.begin()
            for (qq, i), m in masks.items():
                if qq == q:
                    nb.ee(i, 1.0, mask=m, qf=qf if two else None, ridx_stride=nf)
            rep = reps[(2 * q + k) % 4]
            nb.store(rep, st_qf=QFU)
            rr.append(rep)
        nb.rr = [None, None]
        ws = [nb.vrow(wl[q], U8)] + ([nb.vrow(wh[q], U8)] if two else [])
        _bits_from_rep(nb, list(zip(rr, ws)), R(bit_base + q, P))
    return nb.kernel()


# ------------------------------------------------------------------ SampleNTT with a fixed schedule
NB_SAMPLE = 5            # SHAKE128 blocks: 280 three-byte iterations, the FIPS 203 Appendix B limit (2^-261)
NCAND = 112 * NB_SAMPLE  # 560 candidates
VALID_LAST = NCAND - 128 * (NCAND // 128)   # 48 candidates in the last row


def sample_prep(scratch, cand, r_rows, scan_rows, T, V, fail_row, first_entry):
    """Candidates c (5 rows x 2 limbs) -> V0 = (z + 512 c) * a per lane (a = accept, z = rejections before),
    and the fail count F += [rejections > 304] (fewer than 256 accepted)."""
    nb = NB(scratch)
    nrow = len(cand)
    masks = [np.arange(128) >= k for k in (1, 2, 4, 8, 16, 32, 64)]
    nb.register_masks(masks)
    valid = (np.arange(128) < VALID_LAST).astype(float)
    tmp = nb.tmp(2)
    # padding candidates (beyond 560) -> 4095 (rejected)
    last = cand[-1]
    nb.begin()
    for ref in last:
        nb.term_vv(ref, 0, nb.vrow(valid), 1)
    nb.term([R(nb.ones)], 0, 4095.0 * (1 - valid))
    nb.split_store(tmp[:2])
    cand = cand[:-1] + [tmp[:2]]
    for k in range(nrow):
        # r = floor((c + 767) / 4096): exact (power-of-two division, bias keeps clear of ties)
        nb.begin()
        nb.term(cand[k], 0, 2.0 ** -12)
        nb.const(767 / 4096 - 0.5 + 2.0 ** -13)
        nb.magic()
        nb.store(r_rows[k])
        # inclusive prefix count of rejections within the row (log scan, U8: up to 128)
        cur = r_rows[k]
        for j, sh in enumerate((1, 2, 4, 8, 16, 32, 64)):
            nb.begin()
            nb.term([cur], 0, 1.0, qf=QFU if j else None)
            nb.term([cur], -sh, 1.0, mask=masks[j], qf=QFU if j else None)
            dst = scan_rows[k][j % 2]
            nb.store(dst, st_qf=QFU)
            cur = dst
        scan_rows[k] = cur
    # T = (exclusive prefix + offset) + 512 c ; V0 = T * accept
    for k in range(nrow):
        nb.begin()
        nb.term([scan_rows[k]], 0, 1.0, qf=QFU)
        nb.term([r_rows[k]], 0, 1.0, sign=-1)
        for kk in range(k):
            nb.set_r(0, scan_rows[kk])
            nb.ee(127, 1.0, qf=QFU)
        nb.term(cand[k], 0, 512.0)
        nb.split_store(T[:3])
        nb.begin()
        nb.term([R(nb.ones)], 0, 1.0)
        nb.term([r_rows[k]], 0, 1.0, sign=-1)
        acc = R(nb.alloc())
        nb.store(acc)
        nb.begin()
        for ref in T[:3]:
            nb.term_vv(ref, 0, acc, 1)
        nb.split_store(V[k])
    # fail count: rejections (incl. the 128*nrow - 560 padding lanes, all rejected) = sum of every row's last
    # inclusive count; fail = [real rejections >= 305] = floor((rej - npad + 207)/512)
    npad = 128 * nrow - NCAND
    nb.begin()
    for kk in range(nrow):
        nb.set_r(0, scan_rows[kk])
        nb.ee(127, 2.0 ** -9, qf=QFU)
    nb.const((207 - npad) / 512 - 0.5 + 2.0 ** -11)
    nb.magic()
    if not first_entry:
        nb.term([fail_row], 0, 1.0)
    nb.store(fail_row)
    return nb.kernel()


def sample_stage(scratch, b, Vin, Vout, Y, W, zero_ref, rows_out):
    """One compaction stage: elements whose bit b of z is set move down by 2^b. With V = z_b + 2^(9-b) c
    (z_b = remaining shift / 2^b), m = V mod 2, Y = m/2, W = (V - 1) m / 2 = V*Y - Y:
    V' = V/2 - W - Y + shift(W, 2^b)  (= (V - m)/2 at its new position). After 9 stages V = c."""
    nb = NB(scratch)
    nrow = len(Vin)
    sh = 1 << b
    for k in range(nrow):
        V = Vin[k]
        nb.begin()
        nb.term(V, 0, -0.5)
        nb.const(0.25)
        nb.magic()
        nb.term(V, 0, 0.5)
        nb.store(Y[k])
        nb.begin()
        for ref in V:
            nb.term_vv(ref, 0, Y[k], 1)
        nb.term([Y[k]], 0, -1.0)
        nb.split_store(W[k])
    for k in range(rows_out):
        V = Vin[k]
        nb.begin()
        nb.term(V, 0, 0.5)
        nb.term(W[k], 0, -1.0)
        nb.term([Y[k]], 0, -1.0)
        if sh < 128:
            for ln in range(len(W[k])):
                a = W[k][ln]
                bnext = W[k + 1][ln] if k + 1 < nrow else zero_ref
                s0 = nb.load_slot(0, a)
                nb.load_slot(1, bnext)
                nb.mac(1, kind="RCVE", rc=s0 + sh, cr="CR1", mask=0)
                nb.slots = [None] * 4
        else:
            kk = k + sh // 128
            for ln in range(len(W[k])):
                src = W[kk][ln] if kk < nrow else zero_ref
                nb.term([src], 0, 1.0)
        nb.split_store(Vout[k])
    return nb.kernel()


def _count_to_fail(nb, flag_rows, scale_pow, fail_row, tmp):
    """F += [sum of all flags >= 1]: AGG.SUM of flag * 2^-k into slot 0, then floor(x + 1 - 2^-k) by magic,
    stored from lane 0 only and broadcast back with MULT.EE."""
    for i, fr in enumerate(flag_rows):
        base = nb.slot(fr)
        m = nb.p.mac(acc="ACC.ADD", kind="RCVE", rc=base, cr=nb.cr(2.0 ** -scale_pow), mask=0,
                     agg="AGG.SUM.FIRST" if i == 0 else "AGG.SUM", agg_dest=0)
        nb.last = m
    nb.first = False
    nb.const(1.0 - 2.0 ** -scale_pow - 0.5 + 2.0 ** -(scale_pow + 2))
    nb.magic()
    from nx8lib import DSONE
    nb.store(tmp, ds=DSONE)
    nb.begin()
    nb.set_r(0, tmp)
    nb.ee(0, 1.0)
    nb.term([fail_row], 0, 1.0)
    nb.store(fail_row)


def range_check(scratch, t_rows, fail_row):
    """ek modulus check: F += [some decoded 12-bit t >= q]; per lane flag = floor((t + 767)/4096)."""
    nb = NB(scratch)
    flags = []
    for refs in t_rows:
        fr = R(nb.alloc())
        nb.begin()
        nb.term(refs, 0, 2.0 ** -12)
        nb.const(767 / 4096 - 0.5 + 2.0 ** -13)
        nb.magic()
        nb.store(fr)
        flags.append(fr)
    nb.begin()
    _count_to_fail(nb, flags, 10, fail_row, R(nb.alloc()))
    return nb.kernel()


def bits_equal_check(scratch, a_rows, b_rows, fail_row):
    """dk hash check: F += [H(ek) != h]; per lane (a - b)^2 = a xor b on bit rows (E4)."""
    nb = NB(scratch)
    flags = []
    for a, b in zip(a_rows, b_rows):
        dr, fr = R(nb.alloc()), R(nb.alloc())
        nb.begin()
        nb.term([a], 0, 1.0, qf=QFE)
        nb.term([b], 0, -1.0, qf=QFE)
        nb.store(dr)
        nb.begin()
        nb.term_vs(dr, 0)
        nb.store(fr)
        flags.append(fr)
    nb.begin()
    _count_to_fail(nb, flags, 9, fail_row, R(nb.alloc()))
    return nb.kernel()


def ok_and_status(scratch, fail_row, out_rows, status_row):
    """FIPS 203 App. B / 7.2 / 7.3: any failure -> every output byte 0 and status 0; else status 1."""
    nb = NB(scratch)
    okr = status_row
    nb.begin()
    nb.const(1.0)
    nb.term([fail_row], 0, 1.0, sign=-1)
    nb.store(okr, act="relu", st_qf=QFU)
    for r in out_rows:
        nb.begin()
        nb.term_vv(r, 0, okr, 1, qf=QFU)
        nb.store(r, st_qf=QFU)
    return nb.kernel()
