ML-KEM-768 on the LEO-NG IPU (NX8), unprotected: the reference is mlkem-native

KeyGen, Encaps and Decaps run entirely on the IPU in NX8 under the host rule (the host writes constants and
input bytes before the first kernel launch and reads output bytes after the last), bit-exact against mlkem-native.

Layout
  ipu/mlkem768_ipu.py      the KEM: keygen, encaps, decaps as fixed sequences of kernel launches
  ipu/kernels/nx8io.py     byte/bit I/O, Keccak sponge, SampleNTT, CBD, input checks
  ipu/kernels/nx8poly.py   NTT, inverse NTT, basemul, reductions, compress/decompress
  ipu/kernels/nx8lib.py    NX8 kernel builder (limb splitting, bounds)
  ipu/kernels/keccak.py    Keccak-f[1600]
  ipu/sched.py             hardware loops and loop-register allocation
  ipu/emu.py               emulator session; StrictSession enforces the host rule
  ipu/dump_asm.py          writes asm/ from the kernel builders
  ipu/mlkem_consts.py      q, zetas, reference NTT
  refgen/                  reference vectors from mlkem-native (C, host compiler)
  vectors/                 the reference vectors used by the tests
  tests/                   test_keccak, test_nx8_polyops, test_nx8_io (kernels), test_kem_ipu (the KEM),
                           test_asm_e2e (the KEM run from the committed asm/ files, host watched from outside)
  results/                 test results, cycles (kem_ipu_*.json), mlkem-native CI benchmarks, run_all.log
  asm/                     every kernel's IPU assembly exactly as run (<kernel>.asm, 160 kernels) and
                           launches.json: launch order of keygen/encaps/decaps, each kernel's words and CRs
  report/                  ML-KEM_on_LEO-NG_IPU.pptx and make_deck.py, which builds it from results/

Requirements
  ../emu   - the LEO-NG IPU emulator, a git submodule at the repository root (see ../README.txt)
  ref/mlkem-native - mlkem-native, commit 948be12 (git clone https://github.com/pq-code-package/mlkem-native
             ref/mlkem-native && git -C ref/mlkem-native checkout 948be12); needed to regenerate the vectors
  Python 3.12+ with numpy; gcc

Run
  ./run_all.sh     (10-15 minutes, CPU only; ends with "NX8 summary: ALL BIT-EXACT"; writes results/run_all.log)
  python ipu/dump_asm.py   (rewrites asm/; checks each file assembles to the words the KEM runs)
  <python with python-pptx> report/make_deck.py      (rebuilds the deck from results/)
