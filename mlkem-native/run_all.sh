#!/bin/bash
# ML-KEM-768 on the LEO-NG IPU (NX8): rebuild the reference vectors from mlkem-native (host C) and run every
# check of the implementation. About 10-15 minutes, CPU only. The output is also written to results/run_all.log.
set -eo pipefail
cd "$(dirname "$0")"
mkdir -p build results vectors
if [ -z "$RUN_ALL_INNER" ]; then
  RUN_ALL_INNER=1 "$0" "$@" 2>&1 | tee results/run_all.log
  exit
fi
PY=${PY:-$HOME/kernelgen-venv/bin/python}
REF=${REF:-ref/mlkem-native}
for g in gen_vectors gen_vectors512; do
  gcc -O2 -std=c99 -I "$REF/mlkem" -I "$REF/mlkem/src" refgen/$g.c -o build/$g
done
./build/gen_vectors vectors && ./build/gen_vectors512 vectors
cd tests
echo "######## kernels (NX8) against mlkem-native vectors and hashlib"
echo "== test_keccak (Keccak-f[1600]: 24 one-round programs, and one looped program)"; nice -n 19 $PY test_keccak.py nx8 4
echo "== test_nx8_polyops (polynomial kernels)"; nice -n 19 $PY test_nx8_polyops.py
echo "== test_nx8_io (bytes/bits/fields, sponge on bit rows, CBD, SampleNTT)"; nice -n 19 $PY test_nx8_io.py
echo "######## ML-KEM-768 entirely on the IPU, host rule enforced (StrictSession)"
nice -n 19 $PY test_kem_ipu.py seeds 0 3 > ../build/kem_ipu_seeds0_3.log 2>&1 &
A=$!
nice -n 19 $PY test_kem_ipu.py seeds 3 6 > ../build/kem_ipu_seeds3_6.log 2>&1 &
B=$!
echo "== test_kem_ipu main"; nice -n 19 $PY test_kem_ipu.py main
wait $A; wait $B
echo "== test_kem_ipu seeds 0..2"; cat ../build/kem_ipu_seeds0_3.log
echo "== test_kem_ipu seeds 3..5"; cat ../build/kem_ipu_seeds3_6.log
echo "######## end to end from the committed assembly (asm/*.asm), host watched from outside"
echo "== test_asm_e2e"; nice -n 19 $PY test_asm_e2e.py
$PY - <<'PYEOF'
import json, sys
r = {}
for f in ("nx8_polyops", "nx8_io", "kem_ipu_main", "kem_ipu_seeds0_3", "kem_ipu_seeds3_6", "asm_e2e"):
    r.update(json.load(open(f"../results/{f}.json"))["results"])
bad = [k for k, v in r.items() if not v]
print("NX8 summary:", "ALL BIT-EXACT" if not bad else f"FAILED: {bad}")
sys.exit(1 if bad else 0)
PYEOF
