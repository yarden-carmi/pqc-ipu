"""NX8 (the hardware's 8-bit element format) kernel library: limb-split integers on an 8-bit datapath.

NX8 facts this relies on (ipu_isa.nx8, emulator snapshot 19ea639):
  * a row is 128 elements of 8 bits plus one per-row scale; with the signed fe=1 type (S8) an element is
    k * 2^(u-6), |k| <= 127, where 2^u is the binade of the row's largest |value|. So every integer with
    |x| <= 127 is exact, and a row holds exactly what the quantiser rounds it to.
  * MULT expands both operands to FP32 and multiplies (an 8-bit x 8-bit product is exact), ACC adds in
    FP32 (exact while every partial sum is an integer below 2^24).
  * The AaQ store quantises R_ACC to one NX8 row and leaves R_ACC unchanged.

Integers wider than 8 bits are held as a list of limb rows whose sum is the value. A *split store*
produces them from an exact accumulator: store hi = Q(acc) (the quantiser's own rounding to the row's
grid), reload it, subtract it from the still-live accumulator (ACC.SUB of hi x 1), store lo = acc - hi;
repeat while lo can exceed 127. |x| < 2^14 needs 2 limbs, |x| < 2^24 needs 4 (nlimbs()). This costs one
MULT and one reload per extra limb; no re-accumulation of the terms is needed.

Constants are split the same way (const_limbs): a scalar becomes one CR per 8-bit limb (each CR constant
has its own scale), a per-lane vector one XMEM row per limb. Exact constants are split until the residual
is 0; approximate ones (1/q, z/q, ...) until the residual is below a tolerance that the caller derives
from a data bound, so that a rounding (magic constant 1.5*2^23) of the accumulated quotient is either
exact (Barrett-exact mode, error < 1/(4q)) or within a stated slack.

Register use: R_CYCLIC slots C0..C3 hold data rows (LRU cache, `slot`), R0/R1 hold vector-constant
limbs or a second data operand (LRU cache, `rreg`), CR5..CR13 hold scalar-constant limbs (overflow goes
to uniform vector rows). CR14.L = DS with one active element, CR14.H = QF of the unsigned 8-bit type
(U8, integers 0..255 exact: used for byte digits), CR15.L/H = full DS / QF of S8.
"""
from __future__ import annotations

import math

import numpy as np

import emu
from emu import nx8_const, nx8_row
from ipu_isa.cr_halves import ds as _ds, qf as _qf
from ipu_isa.nx8 import Nx8Format
from sched import LoopProg

Q = 3329
MAGIC = 1.5 * 2 ** 23
U8 = Nx8Format(sign=0, exp_format=1, num_bits=8)
E4 = Nx8Format(sign=1, exp_format=4, num_bits=8)    # E4M3: bits and power-of-two weight rows (15 binades)
QF_U8 = _qf(U8)
QF_E4 = _qf(E4)
DS1 = _ds(valid_elements=1)
DATA_CRS = ["CR2", "CR3", "CR12"]               # up to three 256-row data blocks per kernel (chosen at build)
DATA_REGIONS = [("CR2", 0), ("CR3", 256)]       # (initial; replaced by the blocks a kernel touches)
SCR_CR = "CR4"
CONST_CRS = [f"CR{i}" for i in range(5, 12)]
QFU = "CR14.H"
DSONE = "CR14.L"
QFE = "CR13.H"


def nlimbs(bound: float) -> int:
    """Limb rows a split store needs for integers |x| <= bound."""
    n, b = 1, float(bound)
    while b > 127:
        u = math.floor(math.log2(b))
        b = 2.0 ** (u - 6) - 1
        n += 1
    return n


def hi_bound(bound: float) -> float:
    """Bound of a split's first limb (the quantiser may round up by half a grid step)."""
    if bound <= 127:
        return bound
    return bound + 2.0 ** (math.floor(math.log2(bound)) - 7)


_climb_cache: dict = {}


def const_limbs(v, tol: float | None = None, nmax: int = 6):
    """Split a scalar or a 128-vector into NX8-exact limbs. tol None: exact; else until max|residual| <= tol.
    Returns (limbs, max residual)."""
    v = np.asarray(v, np.float64)
    key = (v.tobytes(), v.shape, tol)
    if key in _climb_cache:
        return _climb_cache[key]
    rem = v.copy()
    out = []
    while np.any(rem) and (tol is None or np.max(np.abs(rem)) > tol):
        if len(out) == nmax:
            raise ValueError(f"constant needs more than {nmax} limbs (residual {np.max(np.abs(rem))})")
        if rem.ndim == 0:
            _, d = nx8_const(float(rem))
            d = np.float64(d)
        else:
            _, _, d = nx8_row(rem)
            d = np.where(rem == 0, 0.0, d)   # (an all-zero row would decode as 2^-127)
        out.append(d)
        rem = rem - d
    res = (out, float(np.max(np.abs(rem))) if rem.size else 0.0)
    _climb_cache[key] = res
    return res


def R(row: int, stride: int = 0):
    return (int(row), int(stride))


def poly(base: int, nl: int = 2, stride: int = 0):
    """A polynomial: 2 lane-rows x nl limb rows, rows base + nl*rho + l."""
    return [[R(base + nl * rho + l, stride) for l in range(nl)] for rho in (0, 1)]


class NB:
    """Straight-line (or looped) NX8 kernel builder on LoopProg."""

    def __init__(self, scratch: int, loop: int = 1):
        assert scratch >= 512
        self.p = LoopProg(regions=DATA_REGIONS + [(SCR_CR, scratch)], ds="CR15.L", qf="CR15.H", loop=loop)
        self.scratch = scratch
        self.nscr = 0
        self.crs = {SCR_CR: ("int", scratch), "CR13": ("int", (QF_E4 << 16) | emu.DS_FULL),
                    "CR14": ("int", (QF_U8 << 16) | DS1), "CR15": ("int", (emu.QF_S8 << 16) | emu.DS_FULL)}
        self.xmem: dict[int, np.ndarray] = {}
        self.xmem_fmt: dict[int, object] = {}
        self.cr_of: dict[float, str] = {}
        self.free_crs = list(CONST_CRS)
        self.vrows: dict[bytes, int] = {}
        self.slots = [None] * 4
        self.slot_t = [0] * 4
        self.rr = [None, None]
        self.rr_t = [0, 0]
        self.t = 0
        self.mask_rows: dict[int, list] = {}
        self.mask_slot: dict[bytes, tuple[int, int]] = {}
        self.cur_mask = None
        self.first = True
        self.last = None
        self.ones = self.alloc()
        self.xmem[self.ones] = np.ones(128)
        self.register_masks([])

    # ---------------------------------------------------------------- resources
    def alloc(self, n: int = 1) -> int:
        r = self.scratch + self.nscr
        self.nscr += n
        assert self.nscr <= 256, "scratch region full"
        return r

    def cr(self, v: float):
        v = float(v)
        if v == 1.0:
            return "CR1"
        if v in self.cr_of:
            return self.cr_of[v]
        if not self.free_crs:
            return None
        word, dec = nx8_const(v)
        assert dec == v, (v, dec)
        name = self.free_crs.pop(0)
        self.cr_of[v] = name
        self.crs[name] = ("int", word)
        return name

    def vrow(self, v, fmt=None) -> tuple:
        """A constant row (input-independent, preloaded); fmt: S8 (default), U8 or E4."""
        v = np.asarray(v, np.float64)
        fmt = fmt or emu.S8
        k = (v.tobytes(), str(fmt))
        if k not in self.vrows:
            _, _, dec = nx8_row(v, fmt)
            assert np.array_equal(np.where(v == 0, 0, dec), v), "constant row not NX8-exact"
            r = self.alloc()
            self.vrows[k] = r
            self.xmem[r] = v
            self.xmem_fmt[r] = fmt
        return R(self.vrows[k])

    def tmp(self, n: int = 1):
        return [R(self.alloc()) for _ in range(n)]

    def register_masks(self, masks) -> None:
        """Mask rows: slot 0 of every row is all-ones (R_MASK persists between kernels on one IPU)."""
        masks = [np.asarray(m, bool) for m in masks]
        for i in range(0, max(1, len(masks)), 7):
            row = self.alloc()
            grp = [np.ones(128, bool)] + masks[i:i + 7]
            self.mask_rows[row] = grp
            for j, m in enumerate(grp):
                if j:
                    self.mask_slot.setdefault(m.tobytes(), (row, j))
            if self.cur_mask is None:
                self.p.load("MASK", row)
                self.cur_mask = row

    def mask(self, m) -> int:
        if m is None:
            return 0
        row, j = self.mask_slot[np.asarray(m, bool).tobytes()]
        if row != self.cur_mask:
            self.p.load("MASK", row)
            self.cur_mask = row
        return j

    # ---------------------------------------------------------------- register caches
    def slot(self, ref) -> int:
        """R_CYCLIC slot base (element index) holding row ref, loading it if needed."""
        self.t += 1
        for s, r in enumerate(self.slots):
            if r == ref:
                self.slot_t[s] = self.t
                return 128 * s
        s = min(range(4), key=lambda i: self.slot_t[i])
        self.slots[s] = ref
        self.slot_t[s] = self.t
        self.p.load(f"C{s}", ref[0], stride=ref[1])
        return 128 * s

    def rreg(self, ref) -> str:
        self.t += 1
        for s, r in enumerate(self.rr):
            if r == ref:
                self.rr_t[s] = self.t
                return f"R{s}"
        s = min(range(2), key=lambda i: self.rr_t[i])
        self.rr[s] = ref
        self.rr_t[s] = self.t
        self.p.load(f"R{s}", ref[0], stride=ref[1])
        return f"R{s}"

    def invalidate(self, ref) -> None:
        self.slots = [None if r == ref else r for r in self.slots]
        self.rr = [None if r == ref else r for r in self.rr]

    # ---------------------------------------------------------------- MACs
    def begin(self) -> None:
        self.first = True

    def _acc(self, sign: int) -> str:
        a = "ACC.ADD" if sign > 0 else "ACC.SUB"
        if self.first:
            a += ".FIRST"
            self.first = False
        return a

    def mac(self, sign: int = 1, **kw):
        m = self.p.mac(acc=self._acc(sign), **kw)
        self.last = m
        return m

    def term(self, refs, shift: int, c, sign: int = 1, mask=None, tol=None, qf=None):
        """sum over data limbs r in refs of  sign * r[lane + shift] * c  (c scalar or 128-vector)."""
        c = np.asarray(c, np.float64)
        if not np.any(c):
            return
        if c.ndim == 1 and np.all(c == c[0]):
            c = np.float64(c[0])
        limbs, _ = const_limbs(c, tol)
        mi = self.mask(mask)
        for lv in limbs:
            if np.ndim(lv) == 0:
                s = sign * (1 if lv > 0 else -1)
                crn = self.cr(abs(float(lv)))
                if crn is not None:
                    for ref in refs:
                        base = self.slot(ref)
                        self.mac(s, kind="RCVE", rc=base + shift, cr=crn, mask=mi, qf=qf)
                    continue
                assert qf is None, "a constant limb of an unsigned-format MULT must be a CR"
                lv = np.full(128, abs(float(lv)))
                vsign = s
            else:
                vsign = sign
            for ref in refs:
                base = self.slot(ref)
                ra = self.rreg(self.vrow(lv))
                self.mac(vsign, kind="RCVV", rc=base + shift, ra=ra, mask=mi, qf=qf)

    def term_vv(self, ref_a, shift: int, ref_b, sign: int = 1, mask=None, qf=None):
        """sign * a[lane + shift] * b[lane] for two data rows (b in R0/R1)."""
        mi = self.mask(mask)
        base = self.slot(ref_a)
        ra = self.rreg(ref_b)
        self.mac(sign, kind="RCVV", rc=base + shift, ra=ra, mask=mi, qf=qf)

    def term_vs(self, ref, shift: int, sign: int = 1, mask=None, qf=None):
        """sign * a[lane + shift]^2."""
        mi = self.mask(mask)
        base = self.slot(ref)
        self.mac(sign, kind="RCVS", rc=base + shift, mask=mi, qf=qf)

    # explicit register placement (bypasses the LRU caches)
    def load_slot(self, s: int, ref) -> int:
        self.slots[s] = ref
        self.slot_t[s] = 10 ** 9
        self.p.load(f"C{s}", ref[0], stride=ref[1])
        return 128 * s

    def set_r(self, reg: int, ref) -> None:
        if self.rr[reg] != ref:
            self.rr[reg] = ref
            self.rr_t[reg] = 10 ** 9
            self.p.load(f"R{reg}", ref[0], stride=ref[1])

    def ee(self, idx: int, c: float = 1.0, sign: int = 1, mask=None, qf=None, ridx_stride: int = 0):
        """sign * Ra[idx] * c broadcast to the (masked) lanes; Ra = R0 ++ R1."""
        mi = self.mask(mask)
        crn = self.cr(abs(c))
        assert crn is not None
        s = sign * (1 if c > 0 else -1)
        self.mac(s, kind="EE", ridx=idx, cr=crn, mask=mi, qf=qf, ridx_stride=ridx_stride)

    def agg_vv(self, rc: int, ra: str, dest: int, first: bool = True, qf=None, rc_stride: int = 0, agg_stride: int = 0):
        """R_ACC[dest] (=|+=) sum over lanes of R_CYCLIC[rc + lane] * Ra[lane]."""
        m = self.p.mac(acc="ACC.ADD", kind="RCVV", rc=rc, ra=ra, mask=0, qf=qf,
                       agg="AGG.SUM.FIRST" if first else "AGG.SUM", agg_dest=dest,
                       rc_stride=rc_stride, agg_stride=agg_stride)
        self.last = m
        self.first = False
        return m

    def const(self, c, sign: int = 1, tol=None):
        self.term([R(self.ones)], 0, c, sign, tol=tol)

    def magic(self) -> None:
        self.const(MAGIC, 1)
        self.const(MAGIC, -1)

    def nop_mac(self) -> None:
        base = self.slot(R(self.ones))
        self.mac(1, kind="RCVE", rc=base, cr="CR0")

    # ---------------------------------------------------------------- stores
    def store(self, ref, act: str = "identity", st_qf=None, ds=None) -> None:
        self.p.items  # noqa: B018
        if self.last is None or self.last.store is not None:
            self.nop_mac()
        m = self.last
        st_qf = st_qf or "CR15.H"        # S8 unless asked: never inherit the MULT's (U8/E4) read format
        m.store, m.st_stride, m.act, m.st_qf, m.ds = ref[0], ref[1], act, st_qf, ds
        self.invalidate(ref)

    def split_store(self, dst) -> None:
        """Store the exact integer accumulator as len(dst) limb rows (hi first)."""
        self.store(dst[0])
        for i in range(1, len(dst)):
            base = self.slot(dst[i - 1])
            self.mac(-1, kind="RCVE", rc=base, cr="CR1")
            self.store(dst[i])

    # ---------------------------------------------------------------- output
    def _set_regions(self) -> None:
        lo, hi = self.scratch, self.scratch + 256
        rows = set()
        for it in self.p.items:
            r = getattr(it, "row", None) if not hasattr(it, "kind") else it.store
            if r is not None and not (lo <= r < hi):
                rows.add(r)
        blocks = sorted({r // 256 * 256 for r in rows})
        assert len(blocks) <= len(DATA_CRS), f"kernel touches {len(blocks)} data blocks: {blocks}"
        self.p.regions = [(cr, b) for cr, b in zip(DATA_CRS, blocks)] + [(SCR_CR, self.scratch)]
        for cr, b in zip(DATA_CRS, blocks):
            self.crs[cr] = ("int", b)

    def kernel(self) -> dict:
        self._set_regions()
        asm = self.p.emit()
        xmem = dict(self.xmem)
        return dict(asm=asm, crs=self.crs, xmem=xmem, xmem_fmt=dict(self.xmem_fmt), mask_rows=dict(self.mask_rows),
                    body_words=self.p.body_words, prologue_words=self.p.prologue_words, loop=self.p.loop,
                    scratch_rows=self.nscr)


# -------------------------------------------------------------------- host-side I/O (tests, data movement)
def host_split(x, nl: int):
    """The limb rows a split store would produce for integer vector x (used to inject test inputs)."""
    x = np.asarray(x, np.float64)
    out, rem = [], x.copy()
    for i in range(nl):
        if i == nl - 1:
            out.append(rem.copy())
            break
        if not np.any(rem):
            out.append(np.zeros_like(rem))   # (written as an all-zero row: decodes as 2^-127, the quirk)
            continue
        _, _, d = nx8_row(rem)
        out.append(d)
        rem = rem - d
    assert np.all(np.abs(out[-1]) <= 127) and np.array_equal(out[-1], np.rint(out[-1])), "too few limbs"
    return out


def write_poly(sess, P, x) -> None:
    x = np.asarray(x, np.float64).reshape(2, 128)
    for rho in (0, 1):
        for ref, lv in zip(P[rho], host_split(x[rho], len(P[rho]))):
            emu.session_write_nx8(sess, ref[0], lv)


def read_rows(sess, refs, fmt=emu.S8):
    return np.sum([emu.session_read_nx8(sess, r[0], 1, fmt)[0] for r in refs], axis=0)


def read_poly(sess, P, check: bool = True):
    v = np.concatenate([read_rows(sess, P[rho]) for rho in (0, 1)])
    r = np.rint(v)
    if check:
        # exact integers, except the documented all-zero-row quirk (|garbage| < 2^-100)
        assert np.all(np.abs(v - r) < 2.0 ** -60), f"non-integer output {v[np.abs(v - r) >= 2.0 ** -60][:4]}"
    return r.astype(np.int64)


def read_int_row(sess, ref, fmt=emu.S8):
    """One row as integers (NX8 decode, then rint; only the all-zero-row quirk leaves non-integers)."""
    v = emu.session_read_nx8(sess, ref[0] if isinstance(ref, tuple) else ref, 1, fmt)[0]
    r = np.rint(v)
    assert np.all(np.abs(v - r) < 2.0 ** -60)
    return r.astype(np.int64)


def write_u8(sess, row: int, vals) -> None:
    emu.session_write_nx8(sess, row, np.asarray(vals, np.float64), fmt=U8)
