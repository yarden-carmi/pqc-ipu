"""ML-KEM-768 keygen / encaps / decaps entirely on one emulated IPU in NX8, under the host rule:
  before the first launch the host writes constants (every kernel's input-independent rows) and the run's
  input bytes into XMEM; between launches it only loads the next program and its CRs/LRs; after the last
  launch it reads the output bytes. emu.StrictSession enforces this (any host XMEM access between launches
  raises), and every intermediate value is produced by IPU kernels.

Data path (all on the IPU): input bytes -> bit rows (bytes_to_bits) -> hashes absorb whole bit rows plus
constant rows (domain bytes, padding) -> squeezed bit rows -> fields (AGG) / CBD (AGG) / SampleNTT (fixed
5-block schedule, arithmetic compaction) -> polynomial kernels (nx8poly) -> fields -> bit rows (fields_to_bits)
-> bytes (pack). Encaps checks ek (all t < q), decaps checks H(ek) == h, as mlkem-native does; any failed check
or SampleNTT limit makes every output byte 0 and the status byte 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "kernels"))
import emu  # noqa: E402
import nx8lib as L  # noqa: E402
import nx8poly as NP  # noqa: E402
import nx8io as IO  # noqa: E402
import keccak as KC  # noqa: E402
from nx8lib import R, QFE  # noqa: E402
from mlkem_consts import Q  # noqa: E402

K = 3
INV128 = pow(128, -1, Q)
W12 = [2.0 ** t for t in range(12)]


class Rows:
    def __init__(self, start, end):
        self.n, self.end = start, end

    def take(self, n):
        r = self.n
        self.n += n
        assert self.n <= self.end, ("rows exhausted", self.n, self.end)
        return r


def polys(base, count, nl=2):
    return [L.poly(base + 2 * nl * i, nl) for i in range(count)]


def st(p, stride):
    return [[(r[0], stride) for r in p[rho]] for rho in (0, 1)]


def dig(base, count):
    return [[(R(base + 4 * i + 2 * rho), R(base + 4 * i + 2 * rho + 1)) for rho in (0, 1)] for i in range(count)]


class IpuMlKem768:
    def __init__(self, session):
        self.S = session
        b0, b1, b2, b3 = Rows(0, 256), Rows(256, 512), Rows(512, 768), Rows(768, 1024)
        # ---------------- block 0: polynomials
        T = b0.take
        self.A = polys(T(36), 9)
        self.t = polys(T(12), 3)
        self.s = polys(T(12), 3)
        self.e = polys(T(12), 3)
        self.small = polys(T(14), 7, nl=1)
        self.r = polys(T(12), 3)
        self.V = [(L.poly(b, 2), L.poly(b + 4, 2)) for b in [T(8) for _ in range(K)]]
        self.uv = polys(T(16), 4)
        self.mu = polys(T(4), 1)[0]
        self.cu = polys(T(12), 3)
        self.cv = polys(T(2), 1, nl=1)[0]
        self.cuin = polys(T(12), 3)          # received u fields (decaps)
        self.cvin = polys(T(2), 1, nl=1)[0]
        self.up = polys(T(12), 3)
        self.vp = polys(T(4), 1)[0]
        self.w = polys(T(4), 1)[0]
        self.mout = polys(T(2), 1, nl=1)[0]  # m' bits (E4)
        self.tdig = T(12)
        self.sdig = T(12)
        self.udig = T(12)
        self.traw = polys(T(12), 3)          # decoded 12-bit fields before centring
        # ---------------- block 1: Keccak state, hash outputs, sampling temporaries
        T = b1.take
        self.SB = T(16)
        self.gout, self.hout, self.jout, self.kout = T(8), T(8), T(8), T(8)
        self.dbits, self.mbits = T(8), T(8)
        self.xof = T(60)
        self.cand = [[R(b) for b in (r, r + 1)] for r in [T(2) for _ in range(5)]]
        self.rr = [R(T(1)) for _ in range(5)]
        self.scan = [[R(T(1)), R(T(1))] for _ in range(5)]
        self.Tt = [R(T(1)) for _ in range(3)]
        self.Va = [[R(T(1)) for _ in range(3)] for _ in range(5)]
        self.Vb = [[R(T(1)) for _ in range(3)] for _ in range(5)]
        self.Y = [R(T(1)) for _ in range(5)]
        self.Wr = [[R(T(1)) for _ in range(3)] for _ in range(5)]
        self.zero = R(T(1))
        self.fail = R(T(1))
        self.hchk = T(8)
        # ---------------- block 2: dk/ek bit rows, PRF outputs
        T = b2.take
        self.dkb = T(160)       # s 0..71 | t 72..143 | rho 144..145 | h 146..147 | z 148..155
        self.prf = T(56)
        # ---------------- block 3: ct bit rows, input and output bytes
        T = b3.take
        self.ctb = T(72)        # u 0..59 | v 60..67
        self.in_d, self.in_z, self.in_m = T(1), T(1), T(1)
        self.in_ek, self.in_dk, self.in_ct = T(10), T(20), T(9)
        self.out_ek, self.out_dk, self.out_ct, self.out_ss, self.out_status = T(10), T(19), T(9), T(1), T(1)
        self._scr = 1024
        self.kern = {}
        self.cat = {}
        self._spec_cache = {}
        self.perm = self._k("permute", KC.keccak_round(self._alloc(256), self.SB, 0, "nx8", loop=24,
                                                        data_base=self.SB // 256 * 256), "keccak")
        self.const_rows = {self.zero[0]: np.zeros(128)}
        self._build()

    # ------------------------------------------------------------ kernel bookkeeping
    def _alloc(self, n=256):
        r = self._scr
        self._scr += n
        assert self._scr <= 16384, "NX8 XMEM rows exhausted"
        return r

    def _k(self, name, kern, cat):
        if "scratch_rows" in kern:
            self._scr += kern["scratch_rows"]       # the kernel was built at the scratch pointer (_nx)
            assert self._scr <= 16384, "NX8 XMEM rows exhausted"
        kern["words"] = emu.assemble(kern["asm"])
        assert len(kern["words"]) <= 1024, (name, len(kern["words"]))
        kern["name"], kern["cat"] = name, cat
        self.kern.setdefault(name, kern)
        return kern

    def _nx(self):
        """Scratch base for the next NB kernel (advanced past it by _k)."""
        return self._scr

    def _mk(self, name, cat, fn, *a, **kw):
        kern = fn(self._nx(), *a, **kw)
        return self._k(name, kern, cat)

    def _mk_keccak(self, name, fn, *a, **kw):
        base = self._alloc(256)
        kern = fn(base, *a, **kw)
        kern["words"] = emu.assemble(kern["asm"])
        kern["name"], kern["cat"] = name, "keccak"
        self.kern[name] = kern
        return kern

    def sponge(self, name, segments, rate, ds, outlen, out_base):
        plan = IO.SpongePlan(segments, rate, ds, outlen)
        seq = []
        for spec in IO.sponge_specs(plan, self.SB, out_base):
            if spec[0] == "permute":
                seq.append(self.perm)
                continue
            key = IO.spec_key(spec)
            if key not in self._spec_cache:
                kern = IO.build_spec(self._nx(), self.SB, spec)
                self._spec_cache[key] = self._k(f"{name}.{spec[0]}{len(self._spec_cache)}", kern, "keccak")
            seq.append(self._spec_cache[key])
        return seq

    # ------------------------------------------------------------ program construction (input-independent)
    def _build(self):
        mk = self._mk
        A, t, s, e, r, uv = self.A, self.t, self.s, self.e, self.r, self.uv
        V1, V2 = st(self.V[0][0], 8), st(self.V[0][1], 8)
        dkb, ctb = self.dkb, self.ctb
        P = {}
        # shared pieces ------------------------------------------------------
        bytes_ = lambda nm, src, n, dst: mk(nm, "encode", IO.bytes_to_bits, src, n, dst)  # noqa: E731
        P["unpack_d"] = [bytes_("unpack_d", self.in_d, 1, self.dbits)]
        P["unpack_z"] = [bytes_("unpack_z", self.in_z, 1, dkb + 148)]
        P["unpack_m"] = [bytes_("unpack_m", self.in_m, 1, self.mbits)]
        P["unpack_ek"] = [bytes_("unpack_ek", self.in_ek, 10, dkb + 72)]
        P["unpack_dk"] = [bytes_("unpack_dk", self.in_dk, 20, dkb)]
        P["unpack_ct"] = [bytes_("unpack_ct", self.in_ct, 9, ctb)]
        rows = lambda base, n: [R(base + i) for i in range(n)]  # noqa: E731
        # SampleNTT for the 9 matrix entries: XOF(rho || j || i), parse, prep, 9 stages, centre into A[3i+j]
        samp = []
        for i in range(K):
            for j in range(K):
                samp += self.sponge(f"xof{i}{j}", [("rows", rows(dkb + 144, 2)), ("bytes", bytes([j, i]))],
                                    168, 0x1F, 840, self.xof)
                if "parse" not in self.kern:
                    mk("parse", "sampling", IO.bits_to_fields, self.xof, 12, 5, [[R(self.cand[0][0][0], 2), R(self.cand[0][1][0], 2)]], W12)
                    mk("prep_first", "sampling", IO.sample_prep, self.cand, self.rr, [list(x) for x in self.scan], self.Tt, self.Va, self.fail, True)
                    mk("prep", "sampling", IO.sample_prep, self.cand, self.rr, [list(x) for x in self.scan], self.Tt, self.Va, self.fail, False)
                    vin = self.Va
                    for b in range(9):
                        nrow = 5 if b < 8 else 4
                        rows_out = 5 if b < 7 else (4 if b == 7 else 2)
                        vout = self.Vb if b % 2 == 0 else self.Va
                        mk(f"stage{b}", "sampling", IO.sample_stage, b, vin[:nrow], vout, self.Y, self.Wr, self.zero, rows_out)
                        vin = vout
                    self.cfinal = [vin[0], vin[1]]
                ent = 3 * i + j
                samp += [self.kern["parse"], self.kern["prep_first" if ent == 0 else "prep"]]
                samp += [self.kern[f"stage{b}"] for b in range(9)]
                samp.append(mk(f"centre_A{ent}", "sampling", NP.scale_reduce, self.cfinal, A[ent], 1, bound=4095))
        P["sample_A"] = samp
        # PRF + CBD (eta = 2) for n polys
        def prf_cbd(tag, seed_rows, n):
            seq = []
            for N in range(n):
                seq += self.sponge(f"prf{tag}{N}", [("rows", seed_rows), ("bytes", bytes([N]))], 136, 0x1F, 128,
                                   self.prf + 8 * N)
            seq.append(mk(f"cbd{tag}", "sampling", IO.bits_to_fields, self.prf, 4, 2 * n,
                          [[R(self.small[0][0][0][0], 1)]], [1.0, 1.0, -1.0, -1.0], out_kind="s8"))
            return seq
        # decode 12-bit fields from bit rows, exact centring (and the ek range check)
        def decode12(tag, bit_base, dst, check=False):
            seq = [mk(f"decode{tag}", "encode", IO.bits_to_fields, bit_base, 12, 6,
                      [[R(self.traw[0][0][0][0], 2), R(self.traw[0][0][1][0], 2)]], W12)]
            seq.append(mk(f"centre{tag}", "encode", NP.scale_reduce, st(self.traw[0], 4), st(dst[0], 4), 1, n=3, bound=4095))
            return seq
        # encode a canonical poly vector (2 limbs) to 12-bit fields' bit rows
        def encode12(tag, src, digbase, bit_base):
            d_ = dig(digbase, K)
            seq = [mk(f"todig{tag}", "encode", NP.to_digits, st(src[0], 4),
                      [[(d_[0][rho][0][0], 4), (d_[0][rho][1][0], 4)] for rho in (0, 1)], n=3)]
            for i in range(K):
                seq.append(mk(f"f2b12{tag}{i}", "encode", IO.fields_to_bits, d_[i], 12, bit_base + 24 * i))
            return seq
        def pack(tag, bit_base, nrows, out_base):
            return [mk(f"pack{tag}", "encode", IO.bits_to_fields, bit_base, 8, nrows, [[R(out_base, 1)]],
                       [2.0 ** t_ for t_ in range(8)], out_kind="u8")]
        # ---------------- keygen
        kg = P["unpack_d"] + P["unpack_z"]
        kg += self.sponge("G_kg", [("rows", rows(self.dbits, 2)), ("bytes", bytes([K]))], 72, 0x06, 64, dkb + 144)
        kg += prf_cbd("kg", rows(dkb + 146, 2), 2 * K)
        kg += P["sample_A"]
        kg += [mk("ntt_se", "poly", NP.ntt, st(self.small[0], 2), st(s[0], 4), n=6, bound=2),
               mk("mulcache_s", "poly", NP.mulcache, st(s[0], 4), V1, V2, n=3, montgomery=False, bound=Q),
               mk("basemul_As", "poly", NP.basemul_acc, st(t[0], 4), [(st(A[j], 12), self.V[j][0], self.V[j][1]) for j in range(K)], n=3),
               mk("add_te", "poly", NP.add_sub, [(t[i], e[i], t[i]) for i in range(K)], bound=Q),
               mk("reduce_t", "poly", NP.reduce_canonical, st(t[0], 4), st(t[0], 4), n=3, bound=2 * Q),
               mk("reduce_s", "poly", NP.reduce_canonical, st(s[0], 4), st(s[0], 4), n=3, bound=Q)]
        kg += encode12("t", t, self.tdig, dkb + 72) + encode12("s", s, self.sdig, dkb)
        kg += self.sponge("H_kg", [("rows", rows(dkb + 72, 74))], 136, 0x06, 32, dkb + 146)
        kg += pack("ek", dkb + 72, 10, self.out_ek) + pack("dk", dkb, 19, self.out_dk)
        kg.append(mk("ok_kg", "encode", IO.ok_and_status, self.fail, [R(self.out_ek + i) for i in range(10)]
                     + [R(self.out_dk + i) for i in range(19)], R(self.out_status)))
        self.seq_keygen = kg
        # ---------------- K-PKE encrypt (shared by encaps and the decaps re-encryption); r seed rows given
        def encrypt(tag, coins_rows, mbits_rows, encode_out):
            seq = decode12(f"t{tag}", dkb + 72, t)
            seq += P["sample_A"]                 # (its first entry resets the failure count F)
            if tag == "en":                      # FIPS 203 7.2 modulus check on ek: F += [some t >= q]
                seq.append(mk("ek_range_check", "encode", IO.range_check,
                              [self.traw[i][rho] for i in range(K) for rho in (0, 1)], self.fail))
            seq += prf_cbd(tag, coins_rows, 2 * K + 1)
            seq += [mk(f"ntt_r{tag}", "poly", NP.ntt, st(self.small[0], 2), st(r[0], 4), n=3, bound=2),
                    mk(f"mulcache_r{tag}", "poly", NP.mulcache, st(r[0], 4), V1, V2, n=3, montgomery=False, bound=Q),
                    mk(f"basemul_u{tag}", "poly", NP.basemul_acc, st(uv[0], 4), [(st(A[3 * j], 4), self.V[j][0], self.V[j][1]) for j in range(K)], n=3),
                    mk(f"basemul_v{tag}", "poly", NP.basemul_acc, uv[3], [(t[j], self.V[j][0], self.V[j][1]) for j in range(K)]),
                    mk(f"intt_uv{tag}", "poly", NP.intt, st(uv[0], 4), st(uv[0], 4), n=4, scale=INV128, bound=Q),
                    mk(f"decompress_mu{tag}", "poly", NP.decompress, mbits_rows, self.mu, 1, src_qf=QFE),
                    mk(f"add_enc{tag}", "poly", NP.add_sub, [(uv[i], self.small[3 + i], uv[i]) for i in range(K)]
                       + [(uv[3], self.small[6], uv[3]), (uv[3], self.mu, uv[3])], bound=2 * Q),
                    mk(f"reduce_uv{tag}", "poly", NP.reduce_canonical, st(uv[0], 4), st(uv[0], 4), n=4, bound=3 * Q),
                    mk(f"compress_u{tag}", "poly", NP.compress, st(uv[0], 4), st(self.cu[0], 4), 10, n=3),
                    mk(f"compress_v{tag}", "poly", NP.compress, uv[3], self.cv, 4)]
            if encode_out:
                d_ = dig(self.udig, K)
                seq.append(mk(f"todig_u{tag}", "encode", NP.to_digits, st(self.cu[0], 4),
                              [[(d_[0][rho][0][0], 4), (d_[0][rho][1][0], 4)] for rho in (0, 1)], n=3))
                for i in range(K):
                    seq.append(mk(f"f2b10{tag}{i}", "encode", IO.fields_to_bits, d_[i], 10, ctb + 20 * i))
                seq.append(mk(f"f2b4{tag}", "encode", IO.fields_to_bits, [self.cv[0][0], self.cv[1][0]], 4, ctb + 60))
            return seq
        # ---------------- encaps
        en = P["unpack_ek"] + P["unpack_m"]
        en += self.sponge("H_en", [("rows", rows(dkb + 72, 74))], 136, 0x06, 32, self.hout)
        en += self.sponge("G_en", [("rows", rows(self.mbits, 2) + rows(self.hout, 2))], 72, 0x06, 64, self.gout)
        en += encrypt("en", rows(self.gout + 2, 2), [[R(self.mbits)], [R(self.mbits + 1)]], True)
        en += pack("ct", ctb, 9, self.out_ct) + pack("ss_en", self.gout, 1, self.out_ss)
        en.append(mk("ok_en", "encode", IO.ok_and_status, self.fail, [R(self.out_ct + i) for i in range(9)]
                     + [R(self.out_ss)], R(self.out_status)))
        self.seq_encaps = en
        # ---------------- decaps
        de = P["unpack_dk"] + P["unpack_ct"]
        de += decode12("s", dkb, s)
        de += [mk("decode_u", "encode", IO.bits_to_fields, ctb, 10, 6,
                  [[R(self.cuin[0][0][0][0], 4), R(self.cuin[0][0][1][0], 4)], [R(self.cuin[0][1][0][0], 4), R(self.cuin[0][1][1][0], 4)]],
                  [2.0 ** t_ for t_ in range(10)]),
               mk("decode_v", "encode", IO.bits_to_fields, ctb + 60, 4, 2, [[R(self.cvin[0][0][0], 1)]],
                  [1.0, 2.0, 4.0, 8.0], out_kind="s8"),
               mk("decompress_u", "poly", NP.decompress, st(self.cuin[0], 4), st(self.up[0], 4), 10, n=3),
               mk("decompress_v", "poly", NP.decompress, self.cvin, self.vp, 4),
               mk("ntt_u", "poly", NP.ntt, st(self.up[0], 4), st(self.up[0], 4), n=3, bound=Q),
               mk("mulcache_u", "poly", NP.mulcache, st(self.up[0], 4), V1, V2, n=3, montgomery=False, bound=Q),
               mk("basemul_w", "poly", NP.basemul_acc, self.w, [(s[j], self.V[j][0], self.V[j][1]) for j in range(K)]),
               mk("intt_w", "poly", NP.intt, self.w, self.w, scale=INV128, bound=Q),
               mk("sub_w", "poly", NP.add_sub, [(self.vp, self.w, self.w)], sub=True, bound=Q),
               mk("reduce_w", "poly", NP.reduce_canonical, self.w, self.w, bound=2 * Q),
               mk("compress_m", "poly", NP.compress, self.w, self.mout, 1, out_qf=L.QFE)]
        de += self.sponge("G_de", [("rows", [self.mout[0][0], self.mout[1][0]] + rows(dkb + 146, 2))], 72, 0x06, 64, self.gout)
        de += self.sponge("J_de", [("rows", rows(dkb + 148, 2) + rows(ctb, 68))], 136, 0x1F, 32, self.jout)
        de += encrypt("de", rows(self.gout + 2, 2), [[self.mout[0][0]], [self.mout[1][0]]], False)
        de += self.sponge("H_de", [("rows", rows(dkb + 72, 74))], 136, 0x06, 32, self.hchk)
        de.append(mk("dk_hash_check", "encode", IO.bits_equal_check, rows(self.hchk, 2), rows(dkb + 146, 2), self.fail))
        pairs = [(self.cu[i][rho], self.cuin[i][rho]) for i in range(K) for rho in (0, 1)]
        pairs += [(self.cv[rho], self.cvin[rho]) for rho in (0, 1)]
        de.append(mk("fo_select", "fo", NP.compare_select, pairs, [R(self.gout), R(self.gout + 1)],
                     [R(self.jout), R(self.jout + 1)], [R(self.kout), R(self.kout + 1)], kqf=L.QFE))
        de += pack("ss_de", self.kout, 1, self.out_ss)
        de.append(mk("ok_de", "encode", IO.ok_and_status, self.fail, [R(self.out_ss)], R(self.out_status)))
        self.seq_decaps = de

    # ------------------------------------------------------------ running under the host rule
    def setup(self):
        """Before the first launch: every kernel's constants and the constant data rows (input-independent)."""
        self.S.preload(list(self.kern.values()))
        for row, v in self.const_rows.items():
            emu.session_write_nx8(self.S, row, v)

    def _run(self, seq):
        cyc = {}
        self.per_kernel = {}
        for kern in seq:
            stt = self.S.run_kernel(kern, kern["name"])
            cyc[kern["cat"]] = cyc.get(kern["cat"], 0) + stt.cycles
            pk_ = self.per_kernel.setdefault(kern["name"], [0, 0, kern["cat"], len(kern["words"])])
            pk_[0] += stt.cycles
            pk_[1] += 1
        return cyc

    def _io(self, inputs, outputs, seq):
        """One operation: inputs written before the first launch, outputs read after the last, nothing between."""
        assert self.S.phase == "setup"
        for base, data in inputs:
            data = np.frombuffer(bytes(data), np.uint8)
            nrow = -(-len(data) // 128)
            pad = np.zeros(128 * nrow, np.uint8)
            pad[:len(data)] = data
            for i in range(nrow):
                L.write_u8(self.S, base + i, pad[128 * i: 128 * i + 128])
        self.S.start()
        launches0 = self.S.launches
        cyc = self._run(seq)
        self.S.finish()
        outs = []
        for base, n in outputs:
            nrow = -(-n // 128)
            v = np.concatenate([L.read_int_row(self.S, base + i, L.U8) for i in range(nrow)])[:n]
            outs.append(bytes(v.astype(np.uint8)))
        status = int(L.read_int_row(self.S, self.out_status, L.U8)[0])
        self.S.phase = "setup"          # the next operation may write its inputs
        return outs, status, cyc, self.S.launches - launches0

    def keygen(self, d, z):
        (ek, dk), status, cyc, n = self._io([(self.in_d, d), (self.in_z, z)], [(self.out_ek, 1184), (self.out_dk, 2400)],
                                            self.seq_keygen)
        return ek, dk, status, cyc, n

    def encaps(self, ek, m):
        (ct, ss), status, cyc, n = self._io([(self.in_ek, ek), (self.in_m, m)], [(self.out_ct, 1088), (self.out_ss, 32)],
                                            self.seq_encaps)
        return ss, ct, status, cyc, n

    def decaps(self, dk, ct):
        (ss,), status, cyc, n = self._io([(self.in_dk, dk), (self.in_ct, ct)], [(self.out_ss, 32)], self.seq_decaps)
        return ss, status, cyc, n
