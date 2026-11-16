from common import *
import keccak as KC
mode = sys.argv[1] if len(sys.argv) > 1 else "fp32"
ntest = int(sys.argv[2]) if len(sys.argv) > 2 else 2
st_in = vec("keccak_in", np.uint64, (16, 25)); st_out = vec("keccak_out", np.uint64, (16, 25))
S = emu.Session(mode)
kernels = [KC.keccak_round(1000 + 0 * r, 0, r, mode) for r in range(24)]   # same scratch: constants differ only in the iota row
# give each round its own scratch so per-round constants persist
kernels = [KC.keccak_round(300 + 128 * r, 0, r, mode) for r in range(24)]
w = emu.session_write_nx8 if mode == "nx8" else (lambda s, row, v: s.write_rows_f32(row, v))
rd = emu.session_read_nx8 if mode == "nx8" else (lambda s, row, n: s.read_rows_f32(row, n))
allok = True
for t in range(ntest):
    w(S, 0, KC.state_to_rows(st_in[t]))
    cyc = 0
    for r in range(24):
        cyc += S.run_kernel(kernels[r], f"round{r}").cycles
    got = KC.rows_to_state(rd(S, 0, 15))
    ok = got == [int(v) for v in st_out[t]]
    allok &= ok
    print(f"test {t}: Keccak-f[1600] bit-exact={ok}  cycles={cyc}")
print(mode, "all ok:", allok, " words/round program:", len(kernels[0]["words"]))
# the same permutation as ONE program: a 24-iteration hardware loop over the round body
kl = KC.keccak_round(300, 0, 0, mode, loop=24)
okl = True
for t in range(ntest):
    w(S, 0, KC.state_to_rows(st_in[t]))
    st = S.run_kernel(kl, "keccak-f looped")
    okl &= KC.rows_to_state(rd(S, 0, 15)) == [int(v) for v in st_out[t]]
print(mode, f"looped Keccak-f[1600]: bit-exact={okl} cycles={st.cycles} program words={len(kl['words'])}"
      f" (body {kl.get('body_words', '?')} x 24 + prologue)")
sys.exit(0 if allok and okl else 1)
