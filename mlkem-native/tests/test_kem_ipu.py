"""ML-KEM-768 entirely on the IPU in NX8 under the host rule (StrictSession): bit-exact vs mlkem-native.

  main          the main KEM vector + 3 crafted ciphertexts + invalid ek / dk (mlkem-native's error returns)
  seeds A B     fresh random seeds A..B-1 (keygen, encaps, decaps)
Asserts that the host touched XMEM only before the first launch (inputs, constants) and after the last
(outputs) of every operation. Writes results/kem_ipu_<tag>.json.
"""
from common import *
import json
sys.path.insert(0, str(ROOT / "ipu"))
import mlkem768_ipu as M

mode = sys.argv[1] if len(sys.argv) > 1 else "main"
S = emu.StrictSession("nx8")
kem = M.IpuMlKem768(S)
kem.setup()
res, ops = {}, {}


def check(name, ok):
    res[name] = bool(ok)
    print(f"{name:42s} {'OK' if ok else 'FAIL'}")


def record(name, cyc, n, per=None):
    ops[name] = dict(cycles=cyc, total=sum(cyc.values()), launches=n, host_accesses_during_run=S.host_accesses_during_run(),
                     per_kernel=per)


if mode == "main":
    coins = vec("kem_coins", np.uint8).tobytes(); ecoins = vec("kem_ecoins", np.uint8).tobytes()
    pk = vec("kem_pk", np.uint8).tobytes(); sk = vec("kem_sk", np.uint8).tobytes()
    ct = vec("kem_ct", np.uint8).tobytes(); ss = vec("kem_ss", np.uint8).tobytes()
    ek, dk, st_, cyc, n = kem.keygen(coins[:32], coins[32:]); record("keygen", cyc, n, dict(kem.per_kernel))
    check("keygen: ek, dk bit-exact, status 1", ek == pk and dk == sk and st_ == 1)
    K1, c, st_, cyc, n = kem.encaps(pk, ecoins); record("encaps", cyc, n, dict(kem.per_kernel))
    check("encaps: ct, ss bit-exact, status 1", c == ct and K1 == ss and st_ == 1)
    K2, st_, cyc, n = kem.decaps(sk, ct); record("decaps", cyc, n, dict(kem.per_kernel))
    check("decaps: ss bit-exact, status 1", K2 == ss and st_ == 1)
    bad = vec("kem_bad_ct", np.uint8, (3, len(ct))); bss = vec("kem_bad_ss", np.uint8, (3, 32))
    for v, nm in enumerate(("flipped bit", "c1 = 0", "all-zero ct")):
        Kx, st_, cyc, n = kem.decaps(sk, bad[v].tobytes())
        check(f"decaps implicit rejection ({nm})", Kx == bss[v].tobytes() and st_ == 1)
    rc = vec("bad_rc", np.int32)
    Kx, cx, st_, cyc, n = kem.encaps(vec("bad_pk", np.uint8).tobytes(), vec("seed_ecoins", np.uint8, (6, 32))[0].tobytes())
    check(f"encaps rejects ek with t >= q (C: {rc[0]})", rc[0] == -4 and st_ == 0 and cx == bytes(1088) and Kx == bytes(32))
    Kx, st_, cyc, n = kem.decaps(vec("bad_sk", np.uint8).tobytes(), vec("seed_ct", np.uint8, (6, 1088))[0].tobytes())
    check(f"decaps rejects dk with H(ek) != h (C: {rc[1]})", rc[1] == -5 and st_ == 0 and Kx == bytes(32))
else:
    a, b = int(sys.argv[2]), int(sys.argv[3])
    sc = vec("seed_coins", np.uint8, (6, 64)); se = vec("seed_ecoins", np.uint8, (6, 32))
    spk = vec("seed_pk", np.uint8, (6, 1184)); ssk = vec("seed_sk", np.uint8, (6, 2400))
    sct = vec("seed_ct", np.uint8, (6, 1088)); sss = vec("seed_ss", np.uint8, (6, 32))
    for v in range(a, b):
        ek, dk, s1, cyc, n = kem.keygen(sc[v, :32].tobytes(), sc[v, 32:].tobytes()); record(f"seed{v} keygen", cyc, n)
        K1, c, s2, cyc, n = kem.encaps(spk[v].tobytes(), se[v].tobytes()); record(f"seed{v} encaps", cyc, n)
        K2, s3, cyc, n = kem.decaps(ssk[v].tobytes(), sct[v].tobytes()); record(f"seed{v} decaps", cyc, n)
        check(f"seed {v}: ek dk ct ss(enc) ss(dec) bit-exact", ek == spk[v].tobytes() and dk == ssk[v].tobytes()
              and c == sct[v].tobytes() and K1 == sss[v].tobytes() and K2 == sss[v].tobytes() and s1 == s2 == s3 == 1)
host = S.host_accesses_during_run()
check(f"host XMEM accesses between launches: {host}", host == 0)
before = sum(1 for (ph, _, _) in S.host_log if ph == "setup"); after = sum(1 for (ph, _, _) in S.host_log if ph == "done")
print(f"host XMEM accesses: before first launch {before} (constants + inputs), between launches {host}, after last {after} (outputs)")
for k, v in ops.items():
    if v["per_kernel"] is not None or mode == "main":
        print(f"{k}: total {v['total']} cycles {v['cycles']} launches {v['launches']}")
print("ALL OK:", all(res.values()))
tag = mode if mode == "main" else f"seeds{sys.argv[2]}_{sys.argv[3]}"
json.dump(dict(results=res, ops=ops, host_log_counts=dict(before=before, during=host, after=after),
               kernels=len(kem.kern), scratch_end=kem._scr,
               words={n_: len(k_["words"]) for n_, k_ in kem.kern.items()}),
          open(ROOT / "results" / f"kem_ipu_{tag}.json", "w"), indent=1)
