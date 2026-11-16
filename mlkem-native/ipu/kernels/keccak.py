"""Keccak-f[1600] on the IPU, one permutation, bit-sliced: one lane element per state bit.

Layout: 15 rows; row R(x,p) holds Keccak lane (x, 2p) in elements 0..63 and lane (x, 2p+1) in 64..127
(bit z at element 64h + z); the upper half of R(x,2) is padding and stays 0.

The IPU has no XOR/AND/rotate. For bits, XOR is a square:  a ^ b = (a - b)^2,  and MULT.RC.VS squares an
R_CYCLIC window at any element offset, so a rotation (rc_idx shift + two masked windows) and the square
of a difference are free inside one multiply. The state is kept in "difference form": a row S with
A = S^2 (S in {-1,0,1}), so every XOR's square is taken by the next consumer's MULT.RC.VS.
  theta:  T1 = A(x,0) - A(x,1);  T2 = T1^2 - A(x,2);  T3 = T2^2 - swap_halves(T2^2)  => C[x] = T3^2
          Dd  = C[x-1] - rot1(C[x+1])                                               => D[x] = Dd^2
          Delta(x,p) = A(x,p) - D[x]                                                => A ^ D = Delta^2
  rho+pi: B(dest) = rot(Delta^2, r)  (two masked, shifted MULT.RC.VS per lane)
  chi:    S'(x,p) = B - B2 + B1*B2       (S'^2 = B ^ (~B1 & B2))
  iota:   S'(0,0) = S'(0,0)^2 - RC       ((A - c)^2 = A ^ c)
All values are small integers, exact in FP32 and in NX8 (signed 8-bit). One program per round.
"""
from __future__ import annotations

import numpy as np

import emu
from emu import nx8_const
from sched import LoopProg, Prog

ROT = [[0, 36, 3, 41, 18], [1, 44, 10, 45, 2], [62, 6, 43, 15, 61], [28, 55, 25, 21, 56], [27, 20, 39, 8, 14]]  # ROT[x][y]
RC = [0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000, 0x000000000000808B,
      0x0000000080000001, 0x8000000080008081, 0x8000000000008009, 0x000000000000008A, 0x0000000000000088,
      0x0000000080008009, 0x000000008000000A, 0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
      0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
      0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008]


def srow(x, p):
    return 3 * x + p


class KBuilder:
    def __init__(self, scratch: int, mode: str, loop: int = 0, data_base: int = 0):
        self.mode = mode
        if loop:
            self.p = LoopProg(regions=[("CR2", data_base), ("CR3", scratch)], ds="CR15.L", qf="CR15.H", loop=loop)
        else:
            self.p = Prog(regions=[("CR2", data_base), ("CR3", scratch)], ds="CR15.L", qf="CR15.H")
        self.scratch = scratch
        self.ones = scratch
        self.mask_base = scratch + 1           # up to 15 mask rows
        self.tmp_base = scratch + 16           # temporaries
        self.const_base = scratch + 112
        self.xmem = {self.ones: np.ones(128)}
        self.crs = {"CR2": ("int", data_base), "CR3": ("int", scratch), "CR15": ("int", (emu.QF_S8 << 16) | emu.DS_FULL)}
        self.mask_rows: dict[int, list] = {}
        self.mask_slot: dict[bytes, tuple[int, int]] = {}
        self.cur_mask_row = None
        self.slot_rows = [None, None, None]     # rows resident in C0..C2
        self.next_slot = 0
        self.ntmp = 0
        self.nconst = 0
        self.p.load("C3", self.ones)

    # masks: registered up front so mask rows group them
    def register_masks(self, groups):
        for gi, group in enumerate(groups):
            row = self.mask_base + gi
            self.mask_rows[row] = list(group)
            for si, m in enumerate(group):
                self.mask_slot.setdefault(np.asarray(m, bool).tobytes(), (row, si))

    def mask(self, m) -> int:
        row, si = self.mask_slot[np.asarray(m, bool).tobytes()]
        if row != self.cur_mask_row:
            self.p.load("MASK", row)
            self.cur_mask_row = row
        return si

    def tmp(self) -> int:
        r = self.tmp_base + self.ntmp
        self.ntmp += 1
        assert self.ntmp < 96
        return r

    def slot(self, row: int, avoid=()) -> int:
        """R_CYCLIC slot holding `row` (loading it if needed); returns rc base."""
        for s, r in enumerate(self.slot_rows):
            if r == row:
                return 128 * s
        while self.next_slot in avoid:
            self.next_slot = (self.next_slot + 1) % 3
        s = self.next_slot
        self.next_slot = (self.next_slot + 1) % 3
        self.slot_rows[s] = row
        self.p.load(f"C{s}", row)
        return 128 * s

    def stored(self, row):
        """A row was rewritten: forget resident copies."""
        self.slot_rows = [None if r == row else r for r in self.slot_rows]

    def const_row(self, v) -> int:
        r = self.const_base + self.nconst
        self.nconst += 1
        self.xmem[r] = np.asarray(v, np.float64)
        return r

    def cr_one(self):
        return "CR1"

    def kernel(self):
        asm = self.p.emit()
        return dict(asm=asm, crs=self.crs, xmem=self.xmem, mask_rows=self.mask_rows,
                    body_words=getattr(self.p, "body_words", None), loop=getattr(self.p, "loop", 1))


def _masks():
    z = np.arange(128) % 64
    h = np.arange(128) // 64
    allm = np.ones(128, bool)
    low, high = h == 0, h == 1
    rot1a, rot1b = z >= 1, z == 0
    groups = [[allm, low, high, rot1a, rot1b]]
    rmasks = []
    for r in sorted({ROT[x][y] for x in range(5) for y in range(5)} - {0}):
        for dh in (0, 1):
            hm = h == dh
            rmasks += [hm & (z >= r), hm & (z < r)]
    for i in range(0, len(rmasks), 8):
        groups.append(rmasks[i:i + 8])
    return groups, dict(all=allm, low=low, high=high, rot1a=rot1a, rot1b=rot1b)


def keccak_round(scratch: int, state_base: int, rnd: int, mode: str = "fp32", loop: int = 0,
                 data_base: int = 0) -> dict:
    """One round (rnd), or with loop=24 the whole permutation: one program whose body is a round and
    whose iota row (RC bits) is read from 24 consecutive rows with a per-iteration stride of 1."""
    b = KBuilder(scratch, mode, loop=loop, data_base=data_base)
    p = b.p
    groups, M = _masks()
    b.register_masks(groups)
    S = lambda x, pp: state_base + srow(x, pp)  # noqa: E731

    def vs(row, acc, rc_off=0, mask=None, store=None, avoid=()):
        base = b.slot(row, avoid)
        mi = b.mask(M["all"] if mask is None else mask)
        p.mac(kind="RCVS", rc=base + rc_off, mask=mi, acc=acc, store=store)
        if store is not None:
            b.stored(store)

    # ---- theta: C
    T3 = {}
    for x in range(5):
        t1, t2, t3 = b.tmp(), b.tmp(), b.tmp()
        vs(S(x, 0), "ACC.ADD.FIRST")
        vs(S(x, 1), "ACC.SUB", store=t1)
        vs(t1, "ACC.ADD.FIRST")
        vs(S(x, 2), "ACC.SUB", store=t2)
        vs(t2, "ACC.ADD.FIRST")
        vs(t2, "ACC.SUB", rc_off=64, mask=M["low"])
        vs(t2, "ACC.SUB", rc_off=-64, mask=M["high"], store=t3)
        T3[x] = t3
    # ---- theta: D
    Dd = {}
    for x in range(5):
        d = b.tmp()
        vs(T3[(x - 1) % 5], "ACC.ADD.FIRST")
        vs(T3[(x + 1) % 5], "ACC.SUB", rc_off=-1, mask=M["rot1a"])
        vs(T3[(x + 1) % 5], "ACC.SUB", rc_off=63, mask=M["rot1b"], store=d)
        Dd[x] = d
    # ---- theta: Delta = A - D
    Dl = {}
    for x in range(5):
        for pp in range(3):
            r = b.tmp()
            mk = M["low"] if pp == 2 else None
            vs(S(x, pp), "ACC.ADD.FIRST", mask=mk)
            vs(Dd[x], "ACC.SUB", mask=mk, store=r)
            Dl[(x, pp)] = r
    # ---- rho + pi into B rows
    contrib: dict[tuple[int, int], list] = {}
    for x in range(5):
        for y in range(5):
            xd, yd = y, (2 * x + 3 * y) % 5
            contrib.setdefault((xd, yd // 2), []).append((x, y, yd % 2))
    B = {}
    for (xd, pd), items in sorted(contrib.items()):
        rB = b.tmp()
        ops = []
        for (x, y, hd) in items:
            hs, r = y % 2, ROT[x][y]
            src = Dl[(x, y // 2)]
            base_off = 64 * hs - 64 * hd - r
            z = np.arange(128) % 64
            hm = (np.arange(128) // 64) == hd
            if r == 0:
                ops.append((src, base_off, hm))
            else:
                ops.append((src, base_off, hm & (z >= r)))
                ops.append((src, base_off + 64, hm & (z < r)))
        # order by mask row to limit LD.MASK
        ops.sort(key=lambda o: b.mask_slot.get(np.asarray(o[2], bool).tobytes(), (0, 0))[0])
        for i, (src, off, mk) in enumerate(ops):
            if not mk.any():
                continue
            if mk.tobytes() not in b.mask_slot:   # half masks (r=0)
                mk_use = M["low"] if mk[0] else M["high"]
            else:
                mk_use = mk
            vs(src, "ACC.ADD.FIRST" if i == 0 else "ACC.ADD", rc_off=off, mask=mk_use,
               store=rB if i == len(ops) - 1 else None)
        B[(xd, pd)] = rB
    # ---- chi (+ iota on (0,0))
    for x in range(5):
        for pp in range(3):
            out = S(x, pp) if (x, pp) != (0, 0) else b.tmp()
            b0, b1, b2 = B[(x, pp)], B[((x + 1) % 5, pp)], B[((x + 2) % 5, pp)]
            mi = b.mask(M["all"])
            p.load("R0", b2)
            base0 = b.slot(b0)
            p.mac(kind="RCVE", rc=base0, cr="CR1", acc="ACC.ADD.FIRST", mask=mi)
            p.mac(kind="VE", ridx=0, cr="CR1", acc="ACC.SUB", mask=mi)
            base1 = b.slot(b1, avoid=(base0 // 128,))
            p.mac(kind="RCVV", rc=base1, ra="R0", acc="ACC.ADD", mask=mi, store=out)
            b.stored(out)
            if (x, pp) == (0, 0):
                if loop:
                    rc0 = None
                    for r in range(loop):
                        rr = b.const_row(np.array([(RC[r] >> z) & 1 for z in range(64)] + [0] * 64, np.float64))
                        rc0 = rr if rc0 is None else rc0
                    vs(out, "ACC.ADD.FIRST")
                    p.load("R1", rc0, stride=1)
                else:
                    rc_bits = np.array([(RC[rnd] >> z) & 1 for z in range(64)] + [0] * 64, np.float64)
                    vs(out, "ACC.ADD.FIRST")
                    p.load("R1", b.const_row(rc_bits))
                p.mac(kind="RCVV", rc=384, ra="R1", acc="ACC.SUB", mask=mi, store=S(0, 0))
                b.stored(S(0, 0))
    return b.kernel()


# ---------------------------------------------------------------- host-side layout helpers
def state_to_rows(st):
    """25 uint64 lanes (st[x+5y]) -> 15 rows of bits (difference form: A = S^2, S = bits)."""
    rows = np.zeros((15, 128))
    for x in range(5):
        for y in range(5):
            bits = [(int(st[x + 5 * y]) >> z) & 1 for z in range(64)]
            rows[srow(x, y // 2), 64 * (y % 2): 64 * (y % 2) + 64] = bits
    return rows


def rows_to_state(rows):
    a = np.rint(np.asarray(rows) ** 2).astype(np.int64)
    st = []
    for y in range(5):
        for x in range(5):
            bits = a[srow(x, y // 2), 64 * (y % 2): 64 * (y % 2) + 64]
            st.append(sum(int(bit) << z for z, bit in enumerate(bits)))
    out = [0] * 25
    for y in range(5):
        for x in range(5):
            out[x + 5 * y] = st[5 * y + x]
    return out
