"""Every ML-KEM polynomial kernel in NX8 (the hardware format) against mlkem-native reference vectors.

Inputs are injected as NX8 limb rows (host_split: what a split store would produce); outputs are read
back as the sum of their limb rows. Values in the NTT domain are compared mod q;
everything with a defined representative (reduce, Barrett, add/sub, compress, decompress, CBD, digits)
is compared exactly.
"""
from common import *
import nx8lib as L
import nx8poly as NP
from mlkem_consts import ntt_ref, intt_ref, RINV

S = emu.Session("nx8")
SCR = iter(range(1024, 10 ** 6, 256))
results = {}
stats = {}


def run(name, k, per):
    st = S.run_kernel(k, name)
    stats[name] = dict(cycles=st.cycles / per, macs=st.mult_active_cycles / per, rd=st.xmem_reads / per,
                       wr=st.xmem_writes / per, occ=st.lane_occupancy, eff=st.effective_mac_utilization,
                       words=len(k["words"]), loop=k.get("loop", 1), mult_busy=st.mult_active_cycles / st.cycles)
    return st


def report(name, ok, extra=""):
    results[name] = bool(ok)
    print(f"{name:34s} bit-exact={bool(ok)}  {extra}")


def P(i, nl=2):
    return L.poly(nl * 2 * i, nl)


# ---------------------------------------------------------------- NTT / INTT (16 vectors + adversarial), looped
rng = np.random.default_rng(7)
xin = vec("ntt_in", shape=(16, 256)); xo = vec("ntt_out", shape=(16, 256))
extra = [rng.choice([-(Q - 1), Q - 1], 256) for _ in range(6)] + [np.full(256, Q - 1), np.full(256, -(Q - 1))]
allx = list(xin) + extra
n = len(allx)
for i, x in enumerate(allx):
    L.write_poly(S, P(i), x)
k = NP.ntt(next(SCR), L.poly(0, 2, 4), L.poly(200, 2, 4), n=n)
run("ntt", k, n)
got = [L.read_poly(S, L.poly(200 + 4 * i)) for i in range(n)]
ok = all(np.array_equal(modq(g), modq(xo[i]) if i < 16 else ntt_ref(allx[i])) for i, g in enumerate(got))
report("NTT (16 vectors + 8 extreme)", ok, f"max|out| {max(np.abs(g).max() for g in got)}  bound {k['out_bound']:.0f}")

xin = vec("intt_in", shape=(16, 256)); xo = vec("intt_out", shape=(16, 256))
allx = list(xin) + extra
for i, x in enumerate(allx):
    L.write_poly(S, P(i), x)
k = NP.intt(next(SCR), L.poly(0, 2, 4), L.poly(200, 2, 4), n=n)
run("intt", k, n)
got = [L.read_poly(S, L.poly(200 + 4 * i)) for i in range(n)]
ok = all(np.array_equal(modq(g), modq(xo[i]) if i < 16 else intt_ref(allx[i])) for i, g in enumerate(got))
report("inverse NTT (to Montgomery)", ok, f"max|out| {max(np.abs(g).max() for g in got)}")

# ---------------------------------------------------------------- poly_reduce (canonical), bitwise; int16 inputs: 3 limbs
xin = vec("red_in", shape=(16, 256)); xo = vec("red_out", shape=(16, 256))
for i in range(16):
    L.write_poly(S, P(i, 3), xin[i])
k = NP.reduce_canonical(next(SCR), L.poly(0, 3, 6), L.poly(200, 2, 4), n=16)
run("reduce", k, 16)
got = np.array([L.read_poly(S, L.poly(200 + 4 * i)) for i in range(16)])
report("poly_reduce -> [0,q)", np.array_equal(got, xo))

# ---------------------------------------------------------------- Barrett (centred) over every int16, bitwise
bin_ = vec("barrett_in").reshape(256, 256); bout = vec("barrett_out").reshape(256, 256)
kb = NP.scale_reduce(next(SCR), L.poly(0, 3, 6), L.poly(300, 2, 4), 1, n=32)
ok = True
for b in range(8):
    for i in range(32):
        L.write_poly(S, P(i, 3), bin_[32 * b + i])
    run("barrett", kb, 32)
    got = np.array([L.read_poly(S, L.poly(300 + 4 * i)) for i in range(32)])
    ok &= np.array_equal(got, bout[32 * b: 32 * b + 32])
report("Barrett reduce (all 65536 int16)", ok)

# ---------------------------------------------------------------- tomont (after an exact centring), mod q
xin = vec("red_in", shape=(16, 256)); xo = vec("tomont_out", shape=(16, 256))
for i in range(16):
    L.write_poly(S, P(i, 3), xin[i])
S.run_kernel(NP.scale_reduce(next(SCR), L.poly(0, 3, 6), L.poly(200, 2, 4), 1, n=16), "centre")
k = NP.scale_reduce(next(SCR), L.poly(200, 2, 4), L.poly(300, 2, 4), (1 << 16) % Q, n=16, bound=1664, exact=False)
run("tomont", k, 16)
got = np.array([L.read_poly(S, L.poly(300 + 4 * i)) for i in range(16)])
report("tomont (x 2^16 mod q)", np.array_equal(modq(got), modq(xo)), f"max|out| {np.abs(got).max()}")

# ---------------------------------------------------------------- add / sub, bitwise
a = vec("add_a", shape=(16, 256)); b = vec("add_b", shape=(16, 256))
for i in range(8):
    L.write_poly(S, P(i), a[i]); L.write_poly(S, P(8 + i), b[i])
k = NP.add_sub(next(SCR), [(P(i), P(8 + i), P(16 + i)) for i in range(8)])
run("add", k, 8)
got = np.array([L.read_poly(S, P(16 + i)) for i in range(8)])
k = NP.add_sub(next(SCR), [(P(i), P(8 + i), P(24 + i)) for i in range(8)], sub=True)
run("sub", k, 8)
got2 = np.array([L.read_poly(S, P(24 + i)) for i in range(8)])
report("poly_add / poly_sub", np.array_equal(got, vec("add_out", shape=(16, 256))[:8])
       and np.array_equal(got2, vec("sub_out", shape=(16, 256))[:8]))

# ---------------------------------------------------------------- mulcache + basemul_acc (k = 3), Montgomery as mlkem-native
A = vec("bm_a", shape=(16, 3, 256)); B = vec("bm_b", shape=(16, 3, 256)); O = vec("bm_out", shape=(16, 256))
MC = vec("mc_out", shape=(16, 3, 128))
T = 4
for t in range(T):
    for kk in range(3):
        L.write_poly(S, P(3 * t + kk), A[t, kk])           # rows 0..47
        L.write_poly(S, P(12 + 3 * t + kk), B[t, kk])      # rows 48..95
ka = NP.scale_reduce(next(SCR), L.poly(0, 2, 4), L.poly(0, 2, 4), 1, n=3 * T, bound=Q)   # centre a (exact)
S.run_kernel(ka, "centre a")
# mulcache of every b: V1 at rows 96+8j, V2 at 100+8j
kmc = NP.mulcache(next(SCR), L.poly(48, 2, 4), L.poly(96, 2, 8), L.poly(100, 2, 8), n=3 * T, montgomery=True)
run("mulcache", kmc, 3 * T)
ok = True
for j in range(3 * T):
    v2 = L.read_poly(S, L.poly(100 + 8 * j))
    ok &= np.array_equal(modq(v2[0::2] * (1 << 16)), modq(MC[j // 3, j % 3]))
    ok &= np.abs(v2).max() <= (Q - 1) // 2
report("mulcache (vs mlk_poly_mulcache)", ok, "V2[even]*R == cache (mod q), |V| <= (q-1)/2")
outs = L.poly(200, 2, 4)
kbm = NP.basemul_acc(next(SCR), outs, [(L.poly(4 * 3 * 0 + 4 * kk, 2, 12), L.poly(96 + 8 * kk, 2, 24),
                                        L.poly(100 + 8 * kk, 2, 24)) for kk in range(3)], n=T)
run("basemul_acc", kbm, T)
got = np.array([L.read_poly(S, L.poly(200 + 4 * t)) for t in range(T)])
report("basemul_acc_montgomery (k=3)", np.array_equal(modq(got), modq(O[:T])), f"max|out| {np.abs(got).max()}")
# adversarial: |a| = |V| = 1664 everywhere, signs aligned (largest partial sums)
for kk in range(3):
    L.write_poly(S, L.poly(4 * kk), np.full(256, 1664))
    for r in (96 + 8 * kk, 100 + 8 * kk):
        L.write_poly(S, L.poly(r), np.full(256, 1664))
kbm1 = NP.basemul_acc(next(SCR), L.poly(400), [(L.poly(4 * kk), L.poly(96 + 8 * kk), L.poly(100 + 8 * kk)) for kk in range(3)])
S.run_kernel(kbm1, "basemul extreme")
got = L.read_poly(S, L.poly(400))
g = NP.gammas()
want = []
for p in range(256):
    gp = g[p // 2] if p % 2 == 0 else 1
    want.append(3 * (1664 * 1664 + 1664 * 1664 * (1 if p % 2 == 0 else 1)))
report("basemul_acc extreme |a|=|V|=1664", np.array_equal(modq(got), modq(np.array(want))))

# ---------------------------------------------------------------- compress d = 1, 4, 10, 5, 11 (all 3329 inputs incl.)
cin = vec("cmp_in")
for i in range(16):
    L.write_poly(S, P(i), cin[256 * i: 256 * i + 256])
for d, name in ((1, "cmp_d1"), (4, "cmp_d4"), (10, "cmp_d10")):
    nl = 1 if d <= 6 else 2
    k = NP.compress(next(SCR), L.poly(0, 2, 4), L.poly(200, nl, 2 * nl), d, n=16)
    run(f"compress d={d}", k, 16)
    got = np.concatenate([L.read_poly(S, L.poly(200 + 2 * nl * i, nl)) for i in range(16)])
    report(f"compress d={d}", np.array_equal(got, vec(name, np.uint16).astype(np.int64)))
cin = vec("cmp5_in")
for i in range(16):
    L.write_poly(S, P(i), cin[256 * i: 256 * i + 256])
for d, name in ((5, "cmp_d5"), (11, "cmp_d11")):
    nl = 1 if d <= 6 else 2
    k = NP.compress(next(SCR), L.poly(0, 2, 4), L.poly(200, nl, 2 * nl), d, n=16)
    run(f"compress d={d}", k, 16)
    got = np.concatenate([L.read_poly(S, L.poly(200 + 2 * nl * i, nl)) for i in range(16)])
    report(f"compress d={d} (ML-KEM-1024)", np.array_equal(got, vec(name, np.uint16).astype(np.int64)))

# ---------------------------------------------------------------- decompress d = 1 (frommsg), 4, 10, 5, 11
msgs = vec("msg_in", np.uint8, (16, 32)); fm = vec("frommsg_out", shape=(16, 256))
for i in range(16):
    bits = np.unpackbits(msgs[i], bitorder="little")
    L.write_poly(S, L.poly(2 * i, 1), bits)
k = NP.decompress(next(SCR), L.poly(0, 1, 2), L.poly(200, 2, 4), 1, n=16)
run("decompress d=1", k, 16)
got = np.array([L.read_poly(S, L.poly(200 + 4 * i)) for i in range(16)])
report("decompress d=1 (poly_frommsg)", np.array_equal(got, fm))
for d, name in ((4, "dec_d4"), (5, "dec_d5"), (10, "dec_d10"), (11, "dec_d11")):
    c = np.arange(16 * 256) % (2 ** d)
    if d <= 6:
        for i in range(16):
            L.write_poly(S, L.poly(2 * i, 1), c[256 * i: 256 * i + 256])
        k = NP.decompress(next(SCR), L.poly(0, 1, 2), L.poly(200, 2, 4), d, n=16)
    else:   # c from host byte digits (lo8 unsigned row, hi row): data movement only
        for i in range(16):
            for rho in (0, 1):
                cc = c[256 * i + 128 * rho: 256 * i + 128 * rho + 128]
                L.write_u8(S, 4 * i + 2 * rho, cc & 255)
                emu.session_write_nx8(S, 4 * i + 2 * rho + 1, (cc >> 8).astype(np.float64))
        dig = [[L.R(2 * rho), L.R(2 * rho + 1)] for rho in (0, 1)]
        dig = [[L.R(2 * rho, 4), L.R(2 * rho + 1, 4)] for rho in (0, 1)]
        k = NP.decompress(next(SCR), dig, L.poly(200, 2, 4), d, n=16, src_digits=True)
    run(f"decompress d={d}", k, 16)
    got = np.concatenate([L.read_poly(S, L.poly(200 + 4 * i)) for i in range(16)])
    report(f"decompress d={d}" + (" (ML-KEM-1024)" if d in (5, 11) else ""), np.array_equal(got, vec(name)))

# ---------------------------------------------------------------- CBD eta = 2, 3 from bit planes (host reorders bits)
for eta in (2, 3):
    buf = vec(f"cbd{eta}_in", np.uint8, (16, 64 * eta)); out = vec(f"cbd{eta}_out", shape=(16, 256))
    for i in range(16):
        bits = np.unpackbits(buf[i], bitorder="little").reshape(256, 2 * eta)
        for rho in (0, 1):
            for t in range(2 * eta):
                emu.session_write_nx8(S, 12 * i + 2 * eta * rho + t, bits[128 * rho: 128 * rho + 128, t].astype(np.float64))
    planes = [[L.R(12 * 0 + 2 * eta * rho + t, 12) for t in range(2 * eta)] for rho in (0, 1)]
    k = NP.cbd(next(SCR), planes, L.poly(200, 1, 2), eta, n=16)
    run(f"cbd eta={eta}", k, 16)
    got = np.array([L.read_poly(S, L.poly(200 + 2 * i, 1)) for i in range(16)])
    report(f"CBD eta={eta}" + (" (ML-KEM-512)" if eta == 3 else ""), np.array_equal(got, out))

# ---------------------------------------------------------------- byte digits in/out (the host's (de)serialisation interface)
xs = rng.integers(0, Q, (16, 256))
xs[0, :] = np.arange(256) * 13 % Q; xs[1, :128] = Q - 1; xs[1, 128:] = 0
for i in range(16):
    for rho in (0, 1):
        v = xs[i, 128 * rho: 128 * rho + 128]
        L.write_u8(S, 4 * i + 2 * rho, v & 255)
        emu.session_write_nx8(S, 4 * i + 2 * rho + 1, (v >> 8).astype(np.float64))
dig = [[L.R(2 * rho, 4), L.R(2 * rho + 1, 4)] for rho in (0, 1)]
k = NP.from_digits(next(SCR), dig, L.poly(200, 2, 4), n=16, centre=True)
run("from_digits (centre)", k, 16)
got = np.array([L.read_poly(S, L.poly(200 + 4 * i)) for i in range(16)])
cen = np.where(xs > Q // 2, xs - Q, xs)
report("from byte digits + exact centre", np.array_equal(got, cen))
k = NP.from_digits(next(SCR), dig, L.poly(200, 2, 4), n=16, centre=False)
S.run_kernel(k, "from_digits")
k = NP.to_digits(next(SCR), L.poly(200, 2, 4), [[L.R(400 + 2 * rho, 4), L.R(401 + 2 * rho, 4)] for rho in (0, 1)], n=16)
run("to_digits", k, 16)
ok = True
for i in range(16):
    for rho in (0, 1):
        lo = L.read_int_row(S, 400 + 4 * i + 2 * rho, L.U8)
        hi = L.read_int_row(S, 401 + 4 * i + 2 * rho)
        ok &= np.array_equal(lo + 256 * hi, xs[i, 128 * rho: 128 * rho + 128]) and np.all(lo < 256) and np.all(lo >= 0)
report("to byte digits (lo8 U8 + hi)", ok)

# ---------------------------------------------------------------- the all-zero-row quirk: zero inputs everywhere
# (an all-zero NX8 row decodes as 2^-127, not 0; every kernel must still produce exact results)
Z = np.zeros(256, np.int64)
ok = True
for i in range(4):
    L.write_poly(S, P(i), Z)
for name, kz, outp, want in (
        ("ntt", NP.ntt(next(SCR), P(0), P(10)), P(10), Z),
        ("intt", NP.intt(next(SCR), P(0), P(11)), P(11), Z),
        ("reduce", NP.reduce_canonical(next(SCR), P(0), P(12)), P(12), Z),
        ("barrett", NP.scale_reduce(next(SCR), P(0), P(13), 1), P(13), Z),
        ("compress10", NP.compress(next(SCR), P(0), P(14), 10), P(14), Z),
        ("compress1", NP.compress(next(SCR), P(0), L.poly(60, 1), 1), L.poly(60, 1), Z),
        ("decompress1", NP.decompress(next(SCR), L.poly(0, 1), P(16), 1), P(16), Z),
        ("mulcache", NP.mulcache(next(SCR), P(0), P(17), P(18)), P(18), Z),
        ("basemul", NP.basemul_acc(next(SCR), P(19), [(P(0), P(1), P(2))]), P(19), Z),
        ("add", NP.add_sub(next(SCR), [(P(0), P(1), P(20))]), P(20), Z),
        ("to_digits", NP.to_digits(next(SCR), P(0), [[L.R(84), L.R(85)], [L.R(86), L.R(87)]]), None, Z)):
    S.run_kernel(kz, name)
    if outp is not None:
        got = L.read_poly(S, outp)
        ok &= np.array_equal(got, want)
for r in (84, 85, 86, 87):
    ok &= np.array_equal(L.read_int_row(S, r, L.U8 if r % 2 == 0 else emu.S8), np.zeros(128))
# a chain on the all-zero polynomial (as decapsulating c1 = 0 would run): NTT -> mulcache -> basemul -> INTT
S.run_kernel(NP.ntt(next(SCR), P(0), P(22)), "z ntt")
S.run_kernel(NP.mulcache(next(SCR), P(22), P(23), P(24), montgomery=False), "z mulcache")
L.write_poly(S, P(25), rng.integers(-1664, 1665, 256))
S.run_kernel(NP.basemul_acc(next(SCR), P(26), [(P(25), P(23), P(24))] * 3), "z basemul")
S.run_kernel(NP.intt(next(SCR), P(26), P(26), scale=NP.INV128), "z intt")
S.run_kernel(NP.reduce_canonical(next(SCR), P(26), P(27), bound=Q), "z reduce")
S.run_kernel(NP.compress(next(SCR), P(27), L.poly(120, 1), 1), "z compress")
zc = np.concatenate([L.read_rows(S, P(26)[rho]) for rho in (0, 1)])
ok &= np.array_equal(L.read_poly(S, L.poly(120, 1)), Z) and np.abs(zc).max() < 2.0 ** -100
print(f"   zero chain NTT->mulcache->basemul->INTT: max |residue| {np.abs(zc).max():.3g}")
# mixed: one zero lane-row next to a full-range one
x = np.concatenate([np.zeros(128, np.int64), rng.integers(-Q + 1, Q, 128)])
L.write_poly(S, P(0), x)
S.run_kernel(NP.ntt(next(SCR), P(0), P(21)), "ntt half-zero")
ok &= np.array_equal(modq(L.read_poly(S, P(21))), ntt_ref(x))
report("all-zero rows (2^-127 quirk)", ok, "zero inputs through every kernel type")

print("ALL NX8 POLY KERNELS BIT-EXACT:", all(results.values()))
import json
(ROOT / "results").mkdir(exist_ok=True)
json.dump(dict(results=results, stats=stats), open(ROOT / "results" / "nx8_polyops.json", "w"), indent=1)
