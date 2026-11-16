"""The NX8 I/O, sponge and sampling kernels against references (FIPS 203 bit orders, hashlib, SampleNTT).
Byte/bit/field conversions and CBD are checked against mlkem-native's vectors where those exist."""
from common import *
import hashlib
import json
import nx8lib as L
import nx8io as IO
import nx8poly as NP
import keccak as KC

S = emu.Session("nx8")
SCR = iter(range(2048, 16384, 256))
nxt = lambda: next(SCR)  # noqa: E731
rng = np.random.default_rng(42)
results, stats = {}, {}


def report(name, ok, extra=""):
    results[name] = bool(ok)
    print(f"{name:40s} bit-exact={bool(ok)} {extra}")


def run(name, k, per=1):
    st = S.run_kernel(k, name)
    stats[name] = dict(cycles=st.cycles / per, words=len(k["words"]), mult_busy=st.mult_active_cycles / st.cycles,
                       lane_occ=st.lane_occupancy, xmem_rd=st.xmem_reads / per, xmem_wr=st.xmem_writes / per)
    return st


def bits_of(b):
    return np.unpackbits(np.frombuffer(bytes(b), np.uint8), bitorder="little").astype(float)


def write_bits(base, bits):
    bits = np.concatenate([bits, np.zeros((-len(bits)) % 128)])
    for i in range(len(bits) // 128):
        emu.session_write_nx8(S, base + i, bits[128 * i: 128 * i + 128], fmt=L.E4)


def read_bits(base, n):
    return np.concatenate([L.read_int_row(S, base + i, L.E4) for i in range(-(-n // 128))])[:n]


def byte_decode(b, d):
    bits = np.unpackbits(np.frombuffer(b, np.uint8), bitorder="little").reshape(-1, d)
    return (bits.astype(np.int64) << np.arange(d)).sum(axis=1)


# ---------------------------------------------------------------- bytes -> bits
data = rng.integers(0, 256, 384).astype(np.uint8)
for i in range(3):
    L.write_u8(S, 100 + i, data[128 * i: 128 * i + 128])
k = IO.bytes_to_bits(nxt(), 100, 3, 200)
run("bytes_to_bits", k, 3 * 128)
report("bytes -> bits (BytesToBits)", np.array_equal(read_bits(200, 3072), bits_of(data)), "cycles/byte %.2f" % stats["bytes_to_bits"]["cycles"])

# ---------------------------------------------------------------- bits -> fields: d = 12, 10, 4, 8 (pack), CBD
for d, kind in ((12, "limbs"), (10, "limbs"), (4, "s8"), (8, "u8")):
    nbytes = 32 * d * 2          # two polys' worth
    raw = rng.integers(0, 256, nbytes).astype(np.uint8).tobytes()
    write_bits(300, bits_of(raw))
    nout = 4
    nl = 2 if kind == "limbs" else 1
    n_body = 1
    while (n_body * d) % 4:
        n_body += 1
    outs = [[L.R(500 + nl * o + l, nl * n_body) for l in range(nl)] for o in range(n_body)]
    w = [2.0 ** t for t in range(d)]
    kk = IO.bits_to_fields(nxt(), 300, d, nout, outs, w, out_kind=kind)
    run(f"bits_to_fields d={d}", kk, 512)
    if kind == "u8":
        got = np.concatenate([L.read_int_row(S, 500 + o, L.U8) for o in range(nout)])
        want = np.frombuffer(raw, np.uint8).astype(np.int64)
    else:
        got = np.concatenate([np.sum([L.read_int_row(S, 500 + nl * o + l) for l in range(nl)], axis=0) for o in range(nout)])
        want = byte_decode(raw, d)
    report(f"bits -> {'bytes (pack)' if d == 8 else f'fields d={d} (ByteDecode)'}", np.array_equal(got, want),
           "cycles/field %.2f" % stats[f"bits_to_fields d={d}"]["cycles"])
for eta in (2, 3):
    buf = vec(f"cbd{eta}_in", np.uint8, (16, 64 * eta)); out = vec(f"cbd{eta}_out", shape=(16, 256))
    write_bits(300, bits_of(buf[:2].tobytes()))
    d = 2 * eta
    n_body = 1
    while (n_body * d) % 4:
        n_body += 1
    outs = [[L.R(500 + o, n_body)] for o in range(n_body)]
    kk = IO.bits_to_fields(nxt(), 300, d, 4, outs, [1.0] * eta + [-1.0] * eta, out_kind="s8")
    run(f"cbd{eta} from bits", kk, 2)
    got = np.concatenate([L.read_int_row(S, 500 + o) for o in range(4)])
    report(f"CBD eta={eta} from PRF bits", np.array_equal(got, out[:2].reshape(-1)), "cycles/poly %.0f" % stats[f"cbd{eta} from bits"]["cycles"])

# ---------------------------------------------------------------- fields -> bits (ByteEncode) for d = 12, 10, 4, 1-limb digits
for d in (12, 10, 4):
    x = rng.integers(0, Q if d == 12 else 2 ** d, 256)
    if d > 8:
        for rho in (0, 1):
            v = x[128 * rho: 128 * rho + 128]
            L.write_u8(S, 600 + 2 * rho, v & 255)
            emu.session_write_nx8(S, 601 + 2 * rho, (v >> 8).astype(float))
        dig = [(L.R(600 + 2 * rho), L.R(601 + 2 * rho)) for rho in (0, 1)]
    else:
        for rho in (0, 1):
            emu.session_write_nx8(S, 600 + rho, x[128 * rho: 128 * rho + 128].astype(float))
        dig = [L.R(600 + rho) for rho in (0, 1)]
    kk = IO.fields_to_bits(nxt(), dig, d, 700)
    run(f"fields_to_bits d={d}", kk)
    want = ((x[:, None] >> np.arange(d)) & 1).reshape(-1)
    report(f"fields d={d} -> bits (ByteEncode)", np.array_equal(read_bits(700, 256 * d), want),
           "cycles/poly %.0f" % stats[f"fields_to_bits d={d}"]["cycles"])

# ---------------------------------------------------------------- sponges on bitstream rows
SB = 900                 # state rows
perm = KC.keccak_round(nxt(), SB, 0, "nx8", loop=24, data_base=SB // 256 * 256)


def sponge(segments, rate, ds, outlen, out_base):
    plan = IO.SpongePlan(segments, rate, ds, outlen)
    cyc = 0
    for spec in IO.sponge_specs(plan, SB, out_base):
        kk = perm if spec[0] == "permute" else IO.build_spec(nxt(), SB, spec)
        cyc += S.run_kernel(kk, spec[0]).cycles
    return cyc


msg = rng.integers(0, 256, 256 + 8 * 128).astype(np.uint8)
write_bits(1000, bits_of(msg))                     # 2 + 8 rows... (rows 1000..)
okh = True
for nrows, tail, rate, ds, fn, outlen in ((2, b"\x03", 72, 0x06, lambda m: hashlib.sha3_512(m).digest(), 64),
                                          (10, b"", 136, 0x06, lambda m: hashlib.sha3_256(m).digest(), 32),
                                          (2, bytes([7, 9]), 168, 0x1F, lambda m: hashlib.shake_128(m).digest(840), 840),
                                          (2, bytes([5]), 136, 0x1F, lambda m: hashlib.shake_256(m).digest(128), 128)):
    segs = [("rows", [L.R(1000 + i) for i in range(nrows)])] + ([("bytes", tail)] if tail else [])
    m = msg[:16 * nrows].tobytes() + tail
    cyc = sponge(segs, rate, ds, outlen, 1100)
    good = np.array_equal(read_bits(1100, 8 * outlen), bits_of(fn(m)))
    okh &= good
    print(f"   sponge rate {rate} ds {ds:#x} msg {len(m)} B out {outlen} B: {good} cycles {cyc}")
report("SHA3-256/512, SHAKE128/256 on bit rows", okh)

# ---------------------------------------------------------------- SampleNTT with a fixed schedule (5 blocks, compaction)
def sample_ref(stream):
    out = []
    for p in range(0, len(stream), 3):
        d1 = stream[p] | ((stream[p + 1] & 15) << 8)
        d2 = (stream[p + 1] >> 4) | (stream[p + 2] << 4)
        for dd in (d1, d2):
            if dd < Q and len(out) < 256:
                out.append(dd)
    return np.array(out), len(out) == 256


def build_sampler(xof_base):
    ks = []
    cand = [[L.R(1300 + 2 * o + l) for l in range(2)] for o in range(5)]
    outs = [[L.R(1300 + l, 2)] for l in range(2)]
    outs = [[L.R(1300, 2), L.R(1301, 2)]]
    ks.append(("parse", IO.bits_to_fields(nxt(), xof_base, 12, 5, outs, [2.0 ** t for t in range(12)])))
    rr = [L.R(1310 + k) for k in range(5)]
    scan = [[L.R(1315 + 2 * k), L.R(1316 + 2 * k)] for k in range(5)]
    T = [L.R(1325 + i) for i in range(3)]
    Va = [[L.R(1330 + 3 * k + i) for i in range(3)] for k in range(5)]
    Vb = [[L.R(1345 + 3 * k + i) for i in range(3)] for k in range(5)]
    fail = L.R(1360)
    ks.append(("prep", IO.sample_prep(nxt(), cand, rr, scan, T, Va, fail, True)))
    Y = [L.R(1361 + k) for k in range(5)]
    Wr = [[L.R(1366 + 3 * k + i) for i in range(3)] for k in range(5)]
    zero = L.R(1381)
    vin = Va
    for b in range(9):
        nrow = 5 if b < 8 else 4
        rows_out = 5 if b < 7 else (4 if b == 7 else 2)
        vout = Vb if b % 2 == 0 else Va
        ks.append((f"stage{b}", IO.sample_stage(nxt(), b, vin[:nrow], vout, Y, Wr, zero, rows_out)))
        vin = vout
    return ks, vin, fail


emu.session_write_nx8(S, 1381, np.zeros(128))
ok = True
cyc_s = {}
ks, vfin, fail = build_sampler(1200)
def stream_of(vals):                         # 560 12-bit candidates -> 840 bytes (FIPS 203 Alg. 7 parse order)
    out = bytearray()
    for a, b in zip(vals[0::2], vals[1::2]):
        out += bytes([a & 255, (a >> 8) | ((b & 15) << 4), b >> 4])
    return bytes(out)


crafted = [stream_of([4095] * 304 + list(range(256))),                       # largest shift (z = 304)
           stream_of([7, 4000] * 152 + [5] * 256),                           # alternating, exactly 256 accepted
           stream_of([3328] * 560),                                          # every candidate accepted (q - 1)
           stream_of([3329] * 305 + [1] * 255),                              # 255 accepted: limit reached
           b"\xff" * 840]                                                    # every candidate rejected
NRAND = 30
for trial in range(NRAND + len(crafted)):
    if trial < NRAND:
        seed = rng.integers(0, 256, 34).astype(np.uint8).tobytes()
        stream = hashlib.shake_128(seed).digest(840)
    else:
        stream = crafted[trial - NRAND]
    write_bits(1200, bits_of(stream))
    write_bits(1200 + 53, np.zeros(7 * 128))
    for name, kk in ks:
        c = S.run_kernel(kk, name).cycles
        cyc_s[name] = c
    got = np.concatenate([np.sum([L.read_int_row(S, r[0]) for r in vfin[rho]], axis=0) for rho in (0, 1)])
    want, full = sample_ref(stream)
    f = L.read_int_row(S, fail)[0]
    if full:
        good = np.array_equal(got, want) and f == 0
    else:
        good = f == 1
    ok &= good
    if trial >= NRAND or not good:
        print(f"   sample trial {trial}: {'ok' if good else 'MISMATCH'} (full={full}, fail flag={f})")
tot = sum(cyc_s.values())
stats["SampleNTT (parse+prep+9 stages)"] = dict(cycles=tot, parts=cyc_s)
report("SampleNTT, fixed 5-block schedule", ok, f"({NRAND} random seeds + {len(crafted)} crafted) cycles {tot} {cyc_s}")

print("ALL NX8 IO KERNELS BIT-EXACT:", all(results.values()))
json.dump(dict(results=results, stats=stats), open(ROOT / "results" / "nx8_io.json", "w"), indent=1, default=str)
