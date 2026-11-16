"""Thin driver around the IPU emulator in <repository root>/emu (git submodule, read-only).

assemble(asm) -> words; Run(asm, fmt) to set CRs/XMEM, run to HALT and read results and stats.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]          # repository root: emu/ is shared by every implementation
for _p in sorted((REPO / "emu" / "src").glob("*/src")):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ipu_as.session import assemble as _assemble  # noqa: E402
from ipu_emu.reference import ReferenceModel  # noqa: E402
from ipu_isa.cr_halves import ds as _ds  # noqa: E402
from ipu_isa.formats import ElementFormat  # noqa: E402

DS_FULL = _ds()  # 128 valid elements, one group, mask pad 0
LANES = 128


def assemble(asm: str) -> list[int]:
    res = _assemble(asm)
    return list(res.words) if hasattr(res, "words") else list(res)


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


class Run:
    """One emulator instance in FP32 or NX8 mode."""

    def __init__(self, asm: str, fmt: str = "fp32") -> None:
        self.fmt = ElementFormat.FP32 if fmt == "fp32" else ElementFormat.NX8
        self.row_bytes = 512 if fmt == "fp32" else 128
        self.m = ReferenceModel(self.fmt)
        self.words = assemble(asm)
        self.m.load_program(self.words)

    # registers -------------------------------------------------------------
    def cr(self, idx: int, value: int) -> None:
        self.m.write_register("cr", value & 0xFFFFFFFF, index=idx)

    def cr_float(self, idx: int, value: float) -> None:
        self.cr(idx, f32_bits(value))

    def lr(self, idx: int, value: int) -> None:
        self.m.write_register("lr", value & 0xFFFFFFFF, index=idx)

    # XMEM ------------------------------------------------------------------
    def write_rows_f32(self, row: int, data: np.ndarray) -> None:
        a = np.asarray(data, dtype="<f4").reshape(-1, LANES)
        self.m.write_xmem(row * self.row_bytes, a.tobytes())

    def read_rows_f32(self, row: int, n: int) -> np.ndarray:
        b = self.m.read_xmem(row * self.row_bytes, n * self.row_bytes)
        return np.frombuffer(b, dtype="<f4").reshape(n, LANES).copy()

    def write_rows_bytes(self, row: int, data: bytes) -> None:
        self.m.write_xmem(row * self.row_bytes, data)

    def read_rows_bytes(self, row: int, n: int) -> bytes:
        return self.m.read_xmem(row * self.row_bytes, n * self.row_bytes)

    def write_meta(self, row: int, word: int) -> None:
        self.m.write_xmem_metadata(row * self.row_bytes, word)

    def read_meta(self, row: int) -> int:
        return self.m.read_xmem_metadata(row * self.row_bytes)

    def write_masks(self, row: int, masks: list[np.ndarray]) -> None:
        """Up to 8 masks of 128 bools into one XMEM row (first 128 bytes), LSB-first per byte."""
        buf = bytearray(self.row_bytes)
        for k, m in enumerate(masks):
            bits = np.packbits(np.asarray(m, dtype=np.uint8), bitorder="little")
            buf[16 * k: 16 * k + 16] = bits.tobytes()
        self.m.write_xmem(row * self.row_bytes, bytes(buf))

    # run -------------------------------------------------------------------
    def run(self, max_cycles: int = 2_000_000):
        r = self.m.run(max_cycles)
        st = self.m.status
        if st.fault is not None:
            raise RuntimeError(f"fault {st.fault} at pc {st.pc}")
        self.result = r
        self.stats = self.m.stats()
        return self.stats


class Session:
    """One IPU (one ReferenceModel) running a sequence of kernels on a shared XMEM.

    A kernel is a dict with 'asm', 'crs' (CR name -> float32 or ('int', value)), 'xmem' (row -> f32 row
    of constants), 'masks' ((row, [mask arrays])). The host writes constants and CRs before each
    kernel (not counted); cycles are the words each kernel's program retires.
    """

    def __init__(self, fmt: str = "fp32") -> None:
        self.fmt = ElementFormat.FP32 if fmt == "fp32" else ElementFormat.NX8
        self.row_bytes = 512 if fmt == "fp32" else 128
        self.m = ReferenceModel(self.fmt)
        self.log: list[tuple[str, object]] = []
        self._consts_written: set[int] = set()

    def write_rows_f32(self, row: int, data) -> None:
        a = np.asarray(data, dtype="<f4").reshape(-1, LANES)
        self.m.write_xmem(row * self.row_bytes, a.tobytes())

    def read_rows_f32(self, row: int, n: int) -> np.ndarray:
        b = self.m.read_xmem(row * self.row_bytes, n * self.row_bytes)
        return np.frombuffer(b, dtype="<f4").reshape(n, LANES).copy()

    def run_kernel(self, k: dict, name: str = "", max_cycles: int = 5_000_000):
        if not k.get("_written_to") is self:
            fmts = k.get("xmem_fmt", {})
            for row, v in k.get("xmem", {}).items():
                if self.fmt == ElementFormat.NX8:
                    session_write_nx8(self, row, v, fmt=fmts.get(row, S8))
                else:
                    self.write_rows_f32(row, v)
            mrows = dict(k.get("mask_rows", {}))
            if "masks" in k:
                mrows[k["masks"][0]] = k["masks"][1]
            for row, masks in mrows.items():
                buf = bytearray(self.row_bytes)
                for i, mk in enumerate(masks):
                    buf[16 * i: 16 * i + 16] = np.packbits(np.asarray(mk, np.uint8), bitorder="little").tobytes()
                self.m.write_xmem(row * self.row_bytes, bytes(buf))
            k["_written_to"] = self
        if "words" not in k:
            k["words"] = assemble(k["asm"])
        self.m.load_program(k["words"])
        for i in range(16):
            self.m.write_register("lr", 0, index=i)
        for cr, v in k.get("crs", {}).items():
            idx = int(cr[2:])
            if isinstance(v, tuple):
                self.m.write_register("cr", v[1] & 0xFFFFFFFF, index=idx)
            else:
                self.m.write_register("cr", f32_bits(float(v)), index=idx)
        self.m.run(max_cycles)
        st = self.m.status
        if st.fault is not None:
            raise RuntimeError(f"{name}: fault {st.fault} at pc {st.pc}")
        s = self.m.stats()
        self.log.append((name, s))
        return s


# ---------------------------------------------------------------------------- NX8 helpers
from ipu_isa.cr_halves import constant as _constant, qf as _qf_half  # noqa: E402
from ipu_isa.nx8 import Nx8Format, decode_element, encode_row  # noqa: E402

S8 = Nx8Format(sign=1, exp_format=1, num_bits=8)   # signed, fe=1: integers -127..127 on the row's grid
QF_S8 = _qf_half(S8)


def nx8_row(values, fmt=S8):
    """Quantize 128 values to one NX8 row: (bytes, metadata word, decoded values)."""
    v = [float(x) for x in np.asarray(values, np.float64).reshape(-1)]
    e = encode_row(v, fmt)
    meta = e.scale | (e.dynamic_exponent << 8)
    dec = np.array([decode_element(c, fmt, e.scale, e.dynamic_exponent) for c in e.elements])
    return e.elements, meta, dec


def nx8_const(v: float, fmt=S8) -> tuple[int, float]:
    """A MULT-constant CR word holding v (rounded to 8 bits); returns (word, decoded value)."""
    b, meta, dec = nx8_row([v] + [0.0] * 127, fmt)
    return _constant(b[0], meta & 0xFF, fmt.exp_format, fmt.sign), float(dec[0])


def session_write_nx8(sess, row: int, values, fmt=S8, exact=True):
    rows = np.asarray(values, np.float64).reshape(-1, LANES)
    for i, r in enumerate(rows):
        b, meta, dec = nx8_row(r, fmt)
        if exact and np.any(r) and not np.array_equal(dec, r):  # an all-zero row decodes as 2^-127 (OQ: scale 0)
            raise ValueError(f"row {row + i} not exactly representable in NX8: {r[dec != r][:4]}")
        sess.m.write_xmem((row + i) * 128, b)
        sess.m.write_xmem_metadata((row + i) * 128, meta)


def session_read_nx8(sess, row: int, n: int, fmt=S8):
    out = []
    for i in range(n):
        b = sess.m.read_xmem((row + i) * 128, 128)
        meta = sess.m.read_xmem_metadata((row + i) * 128)
        out.append([decode_element(c, fmt, meta & 0xFF, (meta >> 8) & 1) for c in b])
    return np.array(out)


# ---------------------------------------------------------------------------- the host rule, enforced
class StrictSession(Session):
    """A Session that enforces the host rule of a run:
      setup: the host may write constants (every kernel's constant rows, preloaded) and the inputs;
      run:   between launches the host may only load a program and its CRs/LRs; ANY host access to XMEM
             (read or write, data or metadata) raises;
      done:  the host reads the outputs.
    Every host XMEM access is logged with its phase, so a test can assert that none happened during the run.
    """

    def __init__(self, fmt: str = "nx8") -> None:
        super().__init__(fmt)
        self.phase = "setup"
        self.host_log: list[tuple[str, str, int]] = []
        self.launches = 0
        m = self.m
        for name in ("write_xmem", "read_xmem", "write_xmem_metadata", "read_xmem_metadata"):
            orig = getattr(m, name)

            def wrapped(addr, *a, _orig=orig, _name=name, **kw):
                self.host_log.append((self.phase, _name, addr))
                if self.phase == "run":
                    raise RuntimeError(f"host {_name} at byte {addr} between kernel launches")
                return _orig(addr, *a, **kw)
            setattr(m, name, wrapped)

    def preload(self, kernels) -> None:
        """Write every kernel's (input-independent) constant rows before the first launch."""
        assert self.phase == "setup"
        for k in kernels:
            if k.get("_written_to") is self:
                continue
            fmts = k.get("xmem_fmt", {})
            for row, v in k.get("xmem", {}).items():
                session_write_nx8(self, row, v, fmt=fmts.get(row, S8))
            for row, masks in dict(k.get("mask_rows", {})).items():
                buf = bytearray(self.row_bytes)
                for i, mk in enumerate(masks):
                    buf[16 * i: 16 * i + 16] = np.packbits(np.asarray(mk, np.uint8), bitorder="little").tobytes()
                self.m.write_xmem(row * self.row_bytes, bytes(buf))
            k["_written_to"] = self
            if "words" not in k:
                k["words"] = assemble(k["asm"])

    def start(self) -> None:
        self.phase = "run"

    def finish(self) -> None:
        self.phase = "done"

    def run_kernel(self, k: dict, name: str = "", max_cycles: int = 5_000_000):
        if self.phase == "run":
            assert k.get("_written_to") is self, f"{name}: constants not preloaded"
        self.launches += 1
        return super().run_kernel(k, name, max_cycles)

    def host_accesses_during_run(self) -> int:
        return sum(1 for (ph, _, _) in self.host_log if ph == "run")
