"""End to end from the committed assembly: ML-KEM-768 run with the machine words of asm/<kernel>.asm.

Every operation gets a fresh StrictSession. Each kernel's machine words are replaced by assemble(asm/<name>.asm),
so what runs is the committed files, not the builders' output; the launch order and every kernel's CR values are
checked against asm/launches.json. Every emulator call is watched from outside StrictSession: between the first
and the last launch the host may only load a program, write registers, run and read the status; any XMEM access
fails the test. Outputs must be bit-exact against mlkem-native (vectors/): the main vector, the 3 crafted
ciphertexts (implicit rejection) and the 6 random seeds. The launch trace (programs and register writes) must be
identical for every input of an operation, so the host takes no decision from the data.

  python test_asm_e2e.py [first_seed last_seed]     (default: seeds 0..5)
Writes results/asm_e2e.json.
"""
from common import *
import hashlib
import json
sys.path.insert(0, str(ROOT / "ipu"))
import mlkem768_ipu as M

ASM = ROOT / "asm"
LJ = json.loads((ASM / "launches.json").read_text())
WORDS = {f.stem: emu.assemble(f.read_text()) for f in sorted(ASM.glob("*.asm"))}
ALLOWED = {"load_program", "write_register", "run", "stats", "status"}
XMEM = {"write_xmem", "read_xmem", "write_xmem_metadata", "read_xmem_metadata"}
res, runs = {}, {}


def check(name, ok):
    res[name] = bool(ok)
    print(f"{name:58s} {'OK' if ok else 'FAIL'}", flush=True)


def cr_value(v):
    return f"0x{v[1] & 0xFFFFFFFF:08x}" if isinstance(v, tuple) else float(v)


def fresh(op):
    """A new session and KEM whose kernels run the committed assembly; returns (kem, watch)."""
    S = emu.StrictSession("nx8")
    watch = dict(S=S, trace=[], xmem=[], other=[])
    m = S.m
    for name in dir(m):
        if name.startswith("_") or not callable(getattr(type(m), name, None)):
            continue
        orig = getattr(m, name)

        def w(*a, _o=orig, _n=name, **k):
            if S.phase == "run":
                if _n in XMEM:
                    watch["xmem"].append(_n)
                elif _n == "load_program":
                    watch["trace"].append(("P", hashlib.sha1(repr(list(a[0])).encode()).hexdigest()))
                elif _n == "write_register":
                    watch["trace"].append(("R", repr(a), repr(sorted(k.items()))))
                elif _n not in ALLOWED:
                    watch["other"].append(_n)
            return _o(*a, **k)
        setattr(m, name, w)
    kem = M.IpuMlKem768(S)
    seq = getattr(kem, f"seq_{'decaps' if op == 'reject' else op}")
    assert [k["name"] for k in seq] == LJ["launch_order"]["decaps" if op == "reject" else op], op
    for k in {id(k): k for k in seq}.values():
        assert {c: cr_value(v) for c, v in k.get("crs", {}).items()} == LJ["kernels"][k["name"]]["crs"], k["name"]
        k["words"] = list(WORDS[k["name"]])
    kem.setup()
    return kem, watch


def done(case, op, ok, watch, cyc, n):
    S = watch["S"]
    xm = len(watch["xmem"]) + S.host_accesses_during_run()
    runs[f"{case} {op}"] = dict(op=op, trace=hashlib.sha1(repr(watch["trace"]).encode()).hexdigest(),
                                cycles=sum(cyc.values()), launches=n)
    check(f"{case} {op}: bit-exact, {xm} XMEM accesses between launches", ok and xm == 0 and not watch["other"])


assert set(WORDS) == set(LJ["kernels"]), "asm/ and launches.json disagree"
a, b = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 6)
coins = vec("kem_coins", np.uint8).tobytes(); ecoins = vec("kem_ecoins", np.uint8).tobytes()
cases = [("main", coins[:32], coins[32:], ecoins, vec("kem_pk", np.uint8).tobytes(), vec("kem_sk", np.uint8).tobytes(),
          vec("kem_ct", np.uint8).tobytes(), vec("kem_ss", np.uint8).tobytes())]
sc = vec("seed_coins", np.uint8, (6, 64)); se = vec("seed_ecoins", np.uint8, (6, 32))
spk = vec("seed_pk", np.uint8, (6, 1184)); ssk = vec("seed_sk", np.uint8, (6, 2400))
sct = vec("seed_ct", np.uint8, (6, 1088)); sss = vec("seed_ss", np.uint8, (6, 32))
cases += [(f"seed{v}", sc[v, :32].tobytes(), sc[v, 32:].tobytes(), se[v].tobytes(), spk[v].tobytes(), ssk[v].tobytes(),
           sct[v].tobytes(), sss[v].tobytes()) for v in range(a, b)]
print(f"{len(WORDS)} kernels from asm/, {len(cases)} input sets", flush=True)
for case, d, z, m_, pk, sk, ct, ss in cases:
    kem, w = fresh("keygen"); ek, dk, st, cyc, n = kem.keygen(d, z)
    done(case, "keygen", ek == pk and dk == sk and st == 1, w, cyc, n)
    kem, w = fresh("encaps"); K, c, st, cyc, n = kem.encaps(pk, m_)
    done(case, "encaps", K == ss and c == ct and st == 1, w, cyc, n)
    kem, w = fresh("decaps"); K, st, cyc, n = kem.decaps(sk, ct)
    done(case, "decaps", K == ss and st == 1, w, cyc, n)
sk = vec("kem_sk", np.uint8).tobytes(); ss = vec("kem_ss", np.uint8).tobytes()
bad = vec("kem_bad_ct", np.uint8, (3, 1088)); bss = vec("kem_bad_ss", np.uint8, (3, 32))
for v, nm in enumerate(("flipped bit", "c1 = 0", "all-zero ct")):
    kem, w = fresh("reject"); K, st, cyc, n = kem.decaps(sk, bad[v].tobytes())
    done(f"rejection ({nm})", "decaps", K == bss[v].tobytes() and K != ss and st == 1, w, cyc, n)
for op in ("keygen", "encaps", "decaps"):
    tr = {r["trace"] for r in runs.values() if r["op"] == op}
    check(f"{op}: launch trace identical across {sum(r['op'] == op for r in runs.values())} inputs", len(tr) == 1)
print("ALL OK:", all(res.values()))
json.dump(dict(results=res, runs=runs, kernels=len(WORDS)), open(ROOT / "results" / "asm_e2e.json", "w"), indent=1)
sys.exit(0 if all(res.values()) else 1)
