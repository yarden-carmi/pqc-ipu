"""A small straight-line program builder and scheduler for the IPU.

The kernels here are written as two in-order streams:
  * MAC ops: one MULT plus the ACC op that consumes its MULT_RES (always the same word, timing rule 3),
    optionally with an ACTIVATE.QUANTIZE identity store of the forwarded R_ACC (same word);
  * loads: LD.MULT (R0/R1), LD.MULT.CYCLIC (slots C0..C3) and LD.MASK.
The scheduler places every MAC in program order, one per word, and every load into the single LOAD slot
at the earliest word its dependencies allow (load->MULT latency 1 word: MULT reads registers at cycle
start; store->load latency 1 word: LOAD runs before AAQ in a word; a load may share the word of the last
MULT reading the old register value). It then assigns loop registers (LRs) for every address/index
operand, emitting INC/DEC in the 3 LR sub-slots (LOAD/MULT/AAQ read LRs after the same word's LR
writes, timing rule 1).

Addresses are absolute XMEM rows; each memory op is relative to one of the region base CRs given to
Prog (offsets must stay < 256 so a single INC/DEC moves an LR between any two offsets).
"""
from __future__ import annotations

from dataclasses import dataclass, field

SLOT_BASE = {"C0": 0, "C1": 128, "C2": 256, "C3": 384}
LR_FIXED = {0: 0}          # LR0 is always 0 (mask_shift operand, cyclic index 0)
POOL = list(range(1, 16))  # LRs the allocator may use


@dataclass
class Load:
    dst: str        # R0 R1 C0..C3 MASK
    row: int
    word: int = -1
    stride: int = 0  # LoopProg: row advances by this much per loop iteration


@dataclass
class Mac:
    kind: str       # RCVV RCVE VE EE
    acc: str        # ACC.ADD / ACC.ADD.FIRST / ACC.SUB / ACC.SUB.FIRST / ACC.MAX ...
    rc: int = 0     # R_CYCLIC element index (RC* kinds)
    ra: str = "R0"  # RCVV operand
    cr: str | None = None   # CR constant (RCVE, VE, EE)
    ridx: int | None = None  # Ra element index (RCVE with LR src, VE base, EE element)
    mask: int = 0
    store: int | None = None
    act: str = "identity"
    reads: tuple[str, ...] | None = None  # override of registers read
    word: int = -1
    note: str = ""
    qf: str | None = None     # LoopProg: QF half for this MULT (default Prog.qf)
    st_qf: str | None = None  # LoopProg: QF half for this MAC's store (default: the MULT's)
    ds: str | None = None     # LoopProg: DS half for the store (default Prog.ds)
    st_stride: int = 0        # LoopProg: store row advances by this much per loop iteration
    agg: str | None = None    # LoopProg: AGG.SUM/AGG.SUM.FIRST instead of an ACC op
    agg_dest: int = 0         # LoopProg: AGG destination slot (0..127)
    rc_stride: int = 0        # LoopProg: per-iteration advance of the R_CYCLIC window index
    ridx_stride: int = 0      # LoopProg: per-iteration advance of the R0/R1 element index
    agg_stride: int = 0       # LoopProg: per-iteration advance of the AGG destination slot

    def regs(self) -> tuple[str, ...]:
        if self.reads is not None:
            return self.reads
        r: list[str] = ["MASK"]
        if self.kind.startswith("RC"):
            s = self.rc % 512
            for k in range(4):
                # window [s, s+128) mod 512 intersects slot k? (a strided window may reach any slot)
                if self.rc_stride or any(((s + i) % 512) // 128 == k for i in (0, 127)):
                    r.append(f"C{k}")
        if self.kind == "RCVV":
            r.append(self.ra)
        if self.kind in ("VE", "EE") or (self.kind == "RCVE" and self.cr is None):
            if self.kind == "VE" or self.ridx_stride:
                r += ["R0", "R1"]
            else:
                r.append("R0" if (self.ridx % 256) < 128 else "R1")
        return tuple(r)


@dataclass
class Prog:
    regions: list[tuple[str, int]]   # (CR name, base row)
    ds: str = "CR15.L"
    qf: str = "CR15.H"
    items: list = field(default_factory=list)
    nwords: int = 0
    start_word: int = 2

    def load(self, dst: str, row: int, stride: int = 0) -> None:
        self.items.append(Load(dst, row, stride=stride))

    def mac(self, **kw) -> Mac:
        m = Mac(**kw)
        self.items.append(m)
        return m

    # ------------------------------------------------------------------
    def _schedule(self) -> None:
        last_load_of: dict[str, Load] = {}          # register -> last load (program order)
        readers_since: dict[str, list[int]] = {}    # register -> words of MACs reading the current value
        store_word: dict[int, int] = {}             # row -> word of last store
        loads_of_row: dict[int, list[int]] = {}     # row -> words of loads that read it (since last store)
        load_busy: set[int] = set()
        mac_word = -1
        start = self.start_word  # Prog: words 0-1 left for LR setup
        for it in self.items:
            if isinstance(it, Load):
                w = start
                prev = last_load_of.get(it.dst)
                if prev is not None:
                    w = max(w, prev.word + 1)
                w = max([w] + readers_since.get(it.dst, []))
                if it.row in store_word:
                    w = max(w, store_word[it.row] + 1)
                while w in load_busy:
                    w += 1
                it.word = w
                load_busy.add(w)
                last_load_of[it.dst] = it
                readers_since[it.dst] = []
                loads_of_row.setdefault(it.row, []).append(w)
            else:
                w = max(start, mac_word + 1)
                for r in it.regs():
                    ld = last_load_of.get(r)
                    if ld is not None:
                        w = max(w, ld.word + 1)
                if it.store is not None:
                    w = max([w] + loads_of_row.get(it.store, []))
                it.word = w
                mac_word = w
                for r in it.regs():
                    readers_since.setdefault(r, []).append(w)
                if it.store is not None:
                    store_word[it.store] = w
                    loads_of_row[it.store] = []
        self.nwords = max([i.word for i in self.items] + [0]) + 1

    def _region(self, row: int) -> tuple[str, int]:
        best = None
        for cr, base in self.regions:
            if base <= row and (best is None or base > best[1]):
                best = (cr, base)
        assert best is not None and row - best[1] < 256, f"row {row} has no region within 255 rows"
        return best[0], row - best[1]

    def emit(self, prologue: str = "", epilogue: str = "") -> str:
        self._schedule()
        W = self.nwords
        words: list[dict[str, str]] = [dict() for _ in range(W + 1)]
        lr_ops: list[list[str]] = [[] for _ in range(W + 1)]
        # LR needs: (word, value) -> chosen LR
        needs: list[tuple[int, int, object, str]] = []  # (word, value, item, operand-key)
        for it in self.items:
            if isinstance(it, Load):
                cr, off = self._region(it.row)
                needs.append((it.word, off, it, "off"))
                if it.dst.startswith("C"):
                    needs.append((it.word, SLOT_BASE[it.dst], it, "idx"))
            else:
                if it.kind.startswith("RC"):
                    needs.append((it.word, it.rc % 512, it, "rc"))
                if it.ridx is not None:
                    needs.append((it.word, it.ridx % 256, it, "ridx"))
                if it.store is not None:
                    cr, off = self._region(it.store)
                    needs.append((it.word, off, it, "st"))
        needs.sort(key=lambda n: n[0])
        val = {lr: 0 for lr in POOL}
        last_use = {lr: -1 for lr in POOL}
        free_slots = [3] * (W + 1)
        chosen: dict[tuple[int, str], int] = {}
        modular = {"rc": 512, "ridx": 256}
        for (w, v, it, key) in needs:
            mod = modular.get(key)
            def holds(lr: int) -> bool:
                return (val[lr] - v) % mod == 0 if mod else val[lr] == v
            cand = [lr for lr in POOL if holds(lr)]
            if cand:
                lr = cand[0]
            else:
                # pick an LR free after some word < = w with an LR sub-slot available
                best = None
                for lr in POOL:
                    if last_use[lr] >= w:
                        continue
                    d = v - val[lr]
                    if mod:
                        d = ((d + mod // 2) % mod) - mod // 2
                    nsteps = max(1, -(-abs(d) // 255))
                    ws = [x for x in range(w, last_use[lr], -1) if free_slots[x] > 0][:nsteps]
                    if len(ws) < nsteps:
                        continue
                    score = (w - last_use[lr]) - 1000 * (nsteps - 1)  # prefer one step, then idle-longest
                    if best is None or score > best[0]:
                        best = (score, lr, ws, d)
                if best is None:
                    raise RuntimeError(f"no LR available at word {w} for value {v}")
                _, lr, ws, d = best
                rem = d
                for at in ws:
                    step = max(-255, min(255, rem))
                    if step:
                        lr_ops[at].append(f"INC LR{lr} {step}" if step > 0 else f"DEC LR{lr} {-step}")
                        free_slots[at] -= 1
                    rem -= step
                assert rem == 0
                val[lr] = val[lr] + d
            last_use[lr] = max(last_use[lr], w)
            chosen[(id(it), key)] = lr
        # render
        out = [prologue] if prologue else []
        out.append("    LD.MASK LR0 CR0 ;;" if False else "")
        for w in range(W):
            slots: list[str] = list(lr_ops[w])
            for it in self.items:
                if it.word != w:
                    continue
                if isinstance(it, Load):
                    cr, off = self._region(it.row)
                    lo = chosen[(id(it), "off")]
                    if it.dst in ("R0", "R1"):
                        slots.append(f"LD.MULT {it.dst} LR{lo} {cr}")
                    elif it.dst == "MASK":
                        slots.append(f"LD.MASK LR{lo} {cr}")
                    else:
                        slots.append(f"LD.MULT.CYCLIC LR{lo} {cr} LR{chosen[(id(it), 'idx')]}")
                else:
                    tail = f"{it.mask} LR0 {self.ds} {self.qf}"
                    if it.kind == "RCVV":
                        m = f"MULT.RC.VV LR{chosen[(id(it), 'rc')]} {it.ra} {tail}"
                    elif it.kind == "RCVE":
                        src = it.cr if it.cr is not None else f"LR{chosen[(id(it), 'ridx')]}"
                        m = f"MULT.RC.VE LR{chosen[(id(it), 'rc')]} {src} {tail}"
                    elif it.kind == "RCVS":
                        m = f"MULT.RC.VS LR{chosen[(id(it), 'rc')]} {tail}"
                    elif it.kind == "VE":
                        m = f"MULT.VE LR{chosen[(id(it), 'ridx')]} {it.cr} {tail}"
                    elif it.kind == "EE":
                        m = f"MULT.EE LR{chosen[(id(it), 'ridx')]} {it.cr} {tail}"
                    else:
                        raise ValueError(it.kind)
                    slots.append(m)
                    slots.append(it.acc)
                    if it.store is not None:
                        cr, off = self._region(it.store)
                        slots.append(f"ACTIVATE.QUANTIZE {it.act} {self.ds} {self.qf} {cr} LR{chosen[(id(it), 'st')]}")
            out.append("    " + (" ; ".join(slots) if slots else "NOP") + " ;;")
        if epilogue:
            out.append(epilogue)
        out.append("    HALT ;;")
        return "\n".join(o for o in out if o)


# ---------------------------------------------------------------------------------------------------
@dataclass
class LoopProg(Prog):
    """Prog with per-MAC QF/DS halves, AGG ops and an optional hardware loop around the whole body.

    loop > 1: the body runs `loop` times (BNE on a down-counter in LR15). Rows of loads/stores with a
    `stride` advance by it each iteration (their LRs are advanced by the end-of-body fix-up), all other
    LRs are restored to their start values. LR start values are set in a short prologue. Every word
    (prologue, body, HALT) is a retired cycle; a taken branch costs nothing extra in this ISA (COND is a
    slot of the same word, no delay slot).
    loop == 1: straight-line, LRs set up in-body (as Prog), body starts at word 0.
    """
    loop: int = 1
    start_word: int = 0
    counter_lr: int = 15

    def _render_mac(self, it, lr):
        ds = it.ds or self.ds
        qf = it.qf or self.qf
        tail = f"{it.mask} LR0 {ds} {qf}"
        if it.kind == "RCVV":
            m = f"MULT.RC.VV LR{lr(it, 'rc')} {it.ra} {tail}"
        elif it.kind == "RCVE":
            src = it.cr if it.cr is not None else f"LR{lr(it, 'ridx')}"
            m = f"MULT.RC.VE LR{lr(it, 'rc')} {src} {tail}"
        elif it.kind == "RCVS":
            m = f"MULT.RC.VS LR{lr(it, 'rc')} {tail}"
        elif it.kind == "VE":
            m = f"MULT.VE LR{lr(it, 'ridx')} {it.cr} {tail}"
        elif it.kind == "EE":
            m = f"MULT.EE LR{lr(it, 'ridx')} {it.cr} {tail}"
        else:
            raise ValueError(it.kind)
        out = [m, (f"{it.agg} LR{lr(it, 'agg')} {self.ds}" if it.agg else it.acc)]
        if it.store is not None:
            sq = it.st_qf or qf
            cr, off = self._region(it.store)
            out.append(f"ACTIVATE.QUANTIZE {it.act} {it.ds or self.ds} {sq} {cr} LR{lr(it, 'st')}")
        return out

    def emit(self, prologue: str = "", epilogue: str = "") -> str:
        self._schedule()
        err, best = None, None
        for reserve, pin in ((3, True), (2, True), (3, False), (2, False), (1, True), (1, False), (4, False)):
            try:
                asm = self._emit(prologue, epilogue, reserve, pin)
            except (RuntimeError, AssertionError, StopIteration) as e:
                err = e
                continue
            # keep the shortest program: body words dominate looped kernels, then the prologue
            key = (self.body_words * self.loop + self.prologue_words, self.body_words)
            if best is None or key < best[0]:
                best = (key, asm, self.body_words, self.prologue_words)
        if best is None:
            raise RuntimeError(f"LR allocation failed: {err}")
        _, asm, self.body_words, self.prologue_words = best
        return asm

    def _emit(self, prologue, epilogue, reserve, pin) -> str:
        W = self.nwords
        looping = self.loop > 1
        pool = [i for i in range(1, 16) if not (looping and i == self.counter_lr)]
        needs = []   # (word, value, item, key, stride, mod)
        for it in self.items:
            if isinstance(it, Load):
                cr, off = self._region(it.row)
                needs.append((it.word, off, it, "off", it.stride, None))
                if it.dst.startswith("C"):
                    needs.append((it.word, SLOT_BASE[it.dst], it, "idx", 0, None))
            else:
                if it.kind.startswith("RC"):
                    needs.append((it.word, it.rc % 512, it, "rc", it.rc_stride, 512))
                if it.ridx is not None:
                    needs.append((it.word, it.ridx % 256, it, "ridx", it.ridx_stride, 256))
                if it.agg:
                    needs.append((it.word, it.agg_dest % 128, it, "agg", it.agg_stride, 128))
                if it.store is not None:
                    cr, off = self._region(it.store)
                    needs.append((it.word, off, it, "st", it.st_stride, None))
        needs.sort(key=lambda n: n[0])

        def steps(d):
            out, rem = [], d
            while rem:
                s_ = max(-255, min(255, rem))
                out.append(s_)
                rem -= s_
            return out

        def op(lr, s_):
            return f"INC LR{lr} {s_}" if s_ > 0 else f"DEC LR{lr} {-s_}"
        # partition the LR pool by stride class (a strided LR advances every iteration; it cannot also
        # serve unstrided operands): up to 3 LRs per strided class, the rest for stride 0
        distinct: dict[int, set] = {}
        for (_, v, _, _, stride, _) in needs:
            distinct.setdefault(stride, set()).add(v)
        part: dict[int, list] = {}
        rest = list(pool)
        for sd in sorted(k for k in distinct if k != 0):
            nres = min(reserve, len(distinct[sd]))
            part[sd], rest = rest[:nres], rest[nres:]
        part[0] = rest
        assert len(rest) >= 3, "too many strided operand classes for 15 LRs"
        val, init, cls = {}, {}, {}
        pinned: dict[int, int] = {}
        last_use = {lr: -1 for lr in pool}
        pro_ops: list[str] = []
        # R_CYCLIC slot indices 128/256/384: pinned LRs set in the prologue (never changed)
        for v in (sorted({n[1] for n in needs if n[3] == "idx" and n[1] != 0}) if pin else []):
            f = part[0].pop()
            pinned[v] = f
            init[f], val[f], cls[f] = v, v, 0
            pro_ops.extend(op(f, s_) for s_ in steps(v))
        free_slots = [3] * (W + 64)
        lr_ops: list[list[str]] = [[] for _ in range(W + 64)]
        chosen = {}

        def steps(d):
            out, rem = [], d
            while rem:
                s = max(-255, min(255, rem))
                out.append(s)
                rem -= s
            return out

        def op(lr, s):
            return f"INC LR{lr} {s}" if s > 0 else f"DEC LR{lr} {-s}"

        for (w, v, it, key, stride, mod) in needs:
            def holds(lr):
                return (val[lr] - v) % mod == 0 if mod else val[lr] == v
            if stride == 0 and (v % mod == 0 if mod else v == 0):
                chosen[(id(it), key)] = 0          # LR0 is never written: it always holds 0
                continue
            if key == "idx" and pin:               # slot bases 128/256/384: one pinned LR each
                lr = pinned[v]
                last_use[lr] = max(last_use[lr], w)
                chosen[(id(it), key)] = lr
                continue
            cpool = [lr for lr in part.get(stride, pool) if lr not in pinned.values()]
            cand = [lr for lr in cpool if lr in init and cls[lr] == stride and holds(lr)]
            if cand:
                lr = cand[0]
            else:
                fresh = [lr for lr in cpool if lr not in init]
                lr = None
                if fresh:
                    f = fresh[0]
                    if looping:
                        pro_ops.extend(op(f, s) for s in steps(v))
                        lr = f
                    else:
                        st = steps(v)
                        ws = [x for x in range(w, -1, -1) if free_slots[x] > 0][:len(st)]
                        if len(ws) == len(st):
                            for at, s in zip(ws, st):
                                lr_ops[at].append(op(f, s))
                                free_slots[at] -= 1
                            lr = f
                    if lr is not None:
                        init[lr], val[lr], cls[lr] = v, v, stride
                if lr is None:
                    best = None
                    for c in cpool:
                        if c not in init or cls[c] != stride or last_use[c] >= w:
                            continue
                        d = v - val[c]
                        if mod:
                            d = ((d + mod // 2) % mod) - mod // 2
                        st = steps(d)
                        ws = [x for x in range(w, last_use[c], -1) if free_slots[x] > 0][:len(st)]
                        if len(ws) < len(st):
                            continue
                        score = (w - last_use[c]) - 1000 * (len(st) - 1)
                        if best is None or score > best[0]:
                            best = (score, c, ws, st, d)
                    if best is None:
                        raise RuntimeError(f"no LR available at word {w} for value {v} (stride {stride})")
                    _, lr, ws, st, d = best
                    for at, s in zip(ws, st):
                        lr_ops[at].append(op(lr, s))
                        free_slots[at] -= 1
                    val[lr] += d
            last_use[lr] = max(last_use[lr], w)
            chosen[(id(it), key)] = lr
        nb = W
        if looping:
            # restore every LR to its start value (+ stride) after its last use in the body
            # (one write per LR per word: two sub-slots writing one LR is fault F-3)
            for lr in init:
                st = steps(init[lr] + cls[lr] - val[lr])
                after = last_use[lr]
                for s in st:
                    at = next((x for x in range(after + 1, nb) if free_slots[x] > 0), None)
                    if at is None:
                        at = nb
                        nb += 1
                    lr_ops[at].append(op(lr, s))
                    free_slots[at] -= 1
                    after = at
            if free_slots[nb - 1] == 0:
                nb += 1
            lr_ops[nb - 1].append(f"DEC LR{self.counter_lr} 1")
            pro_ops = [op(self.counter_lr, s) for s in steps(self.loop)] + pro_ops
        if True:
            # prologue words: never two writes of one LR in a word
            words_p: list[list[str]] = []
            for o in pro_ops:
                reg = o.split()[1]
                for wp in words_p:
                    if len(wp) < 3 and all(x.split()[1] != reg for x in wp):
                        wp.append(o)
                        break
                else:
                    words_p.append([o])
            pro_ops = [x for wp in words_p for x in wp + ["|"]]
        lr = lambda it, key: chosen[(id(it), key)]  # noqa: E731
        out = [prologue] if prologue else []
        if True:
            cur = []
            for o in pro_ops:
                if o == "|":
                    out.append("    " + " ; ".join(cur) + " ;;")
                    cur = []
                else:
                    cur.append(o)
        else:
            for i in range(0, len(pro_ops), 3):
                out.append("    " + " ; ".join(pro_ops[i:i + 3]) + " ;;")
        if looping:
            out.append("L0:")
        by_word: dict[int, list] = {}
        for it in self.items:
            by_word.setdefault(it.word, []).append(it)
        for w in range(nb):
            slots = list(lr_ops[w])
            for it in by_word.get(w, []):
                if isinstance(it, Load):
                    cr, off = self._region(it.row)
                    lo = lr(it, "off")
                    if it.dst in ("R0", "R1"):
                        slots.append(f"LD.MULT {it.dst} LR{lo} {cr}")
                    elif it.dst == "MASK":
                        slots.append(f"LD.MASK LR{lo} {cr}")
                    else:
                        slots.append(f"LD.MULT.CYCLIC LR{lo} {cr} LR{lr(it, 'idx')}")
                else:
                    slots.extend(self._render_mac(it, lr))
            if looping and w == nb - 1:
                slots.append(f"BNE LR{self.counter_lr} CR1 L0")
            out.append("    " + (" ; ".join(slots) if slots else "NOP") + " ;;")
        if epilogue:
            out.append(epilogue)
        out.append("    HALT ;;")
        self.body_words = nb
        self.prologue_words = pro_ops.count("|")
        return "\n".join(o for o in out if o)
