#define _POSIX_C_SOURCE 200809L
/* Reference vectors from mlkem-native (portable C backend, ML-KEM-768), for the IPU kernels.
 * Built with the system compiler on the host CPU only. Writes raw little-endian files into argv[1]. */
#define MLK_CONFIG_PARAMETER_SET 768
#define MLK_CONFIG_NAMESPACE_PREFIX mlkref
#include "src/common.h"
#include "src/compress.c"
#include "src/debug.c"
#include "src/indcpa.c"
#include "src/kem.c"
#include "src/poly.c"
#include "src/poly_k.c"
#include "src/sampling.c"
#include "src/verify.c"
#include "src/fips202/fips202.c"
#include "src/fips202/fips202x4.c"
#include "src/fips202/keccakf1600.c"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng_s = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void)
{
  rng_s ^= rng_s << 13;
  rng_s ^= rng_s >> 7;
  rng_s ^= rng_s << 17;
  return rng_s;
}
static int16_t rnd_range(int lo, int hi) /* [lo, hi) */
{
  return (int16_t)(lo + (int)(rnd() % (uint64_t)(hi - lo)));
}

static const char *outdir;
static void dump(const char *name, const void *p, size_t n)
{
  char path[512];
  FILE *f;
  snprintf(path, sizeof path, "%s/%s.bin", outdir, name);
  f = fopen(path, "wb");
  if (!f || fwrite(p, 1, n, f) != n)
  {
    perror(path);
    exit(1);
  }
  fclose(f);
}

#define NT 16

int main(int argc, char **argv)
{
  static mlk_poly in[NT], out[NT];
  static mlk_polyvec va[NT], vb[NT];
  static mlk_polyvec_mulcache vc[NT];
  int t, i, k;
  if (argc < 2)
  {
    fprintf(stderr, "usage: %s outdir\n", argv[0]);
    return 1;
  }
  outdir = argv[1];

  /* forward NTT: |x| < q */
  for (t = 0; t < NT; t++)
    for (i = 0; i < MLKEM_N; i++) in[t].coeffs[i] = rnd_range(-MLKEM_Q + 1, MLKEM_Q);
  memcpy(out, in, sizeof in);
  for (t = 0; t < NT; t++) mlk_poly_ntt(&out[t]);
  dump("ntt_in", in, sizeof in);
  dump("ntt_out", out, sizeof out);

  /* inverse NTT (to Montgomery): |x| < q */
  for (t = 0; t < NT; t++)
    for (i = 0; i < MLKEM_N; i++) in[t].coeffs[i] = rnd_range(-MLKEM_Q + 1, MLKEM_Q);
  memcpy(out, in, sizeof in);
  for (t = 0; t < NT; t++) mlk_poly_invntt_tomont(&out[t]);
  dump("intt_in", in, sizeof in);
  dump("intt_out", out, sizeof out);

  /* basemul_acc (k=3): a in [0,q), b in (-q,q) */
  for (t = 0; t < NT; t++)
    for (k = 0; k < MLKEM_K; k++)
      for (i = 0; i < MLKEM_N; i++)
      {
        va[t].vec[k].coeffs[i] = rnd_range(0, MLKEM_Q);
        vb[t].vec[k].coeffs[i] = rnd_range(-MLKEM_Q + 1, MLKEM_Q);
      }
  for (t = 0; t < NT; t++)
  {
    mlk_polyvec_mulcache_compute(&vc[t], &vb[t]);
    mlk_polyvec_basemul_acc_montgomery_cached(&out[t], &va[t], &vb[t], &vc[t]);
  }
  dump("bm_a", va, sizeof va);
  dump("bm_b", vb, sizeof vb);
  dump("bm_out", out, sizeof out);

  dump("mc_out", vc, sizeof vc);

  /* poly_reduce: any int16 -> [0,q) ; poly_tomont: -> (-q,q) */
  for (t = 0; t < NT; t++)
    for (i = 0; i < MLKEM_N; i++) in[t].coeffs[i] = (int16_t)rnd();
  memcpy(out, in, sizeof in);
  for (t = 0; t < NT; t++) mlk_poly_reduce(&out[t]);
  dump("red_in", in, sizeof in);
  dump("red_out", out, sizeof out);
  memcpy(out, in, sizeof in);
  for (t = 0; t < NT; t++) mlk_poly_tomont(&out[t]);
  dump("tomont_out", out, sizeof out);

  /* compress / decompress (scalar), inputs canonical */
  {
    static int16_t cin[NT * MLKEM_N];
    static uint16_t c10[NT * MLKEM_N], c4[NT * MLKEM_N], c1[NT * MLKEM_N];
    static int16_t d10[NT * MLKEM_N], d4[NT * MLKEM_N];
    for (i = 0; i < NT * MLKEM_N; i++)
    {
      cin[i] = (int16_t)(i < MLKEM_Q ? i : rnd_range(0, MLKEM_Q));
      c10[i] = mlk_scalar_compress_d10(cin[i]);
      c4[i] = mlk_scalar_compress_d4(cin[i]);
      c1[i] = mlk_scalar_compress_d1(cin[i]);
      d10[i] = mlk_scalar_decompress_d10((uint16_t)(i % 1024));
      d4[i] = mlk_scalar_decompress_d4((uint8_t)(i % 16));
    }
    dump("cmp_in", cin, sizeof cin);
    dump("cmp_d10", c10, sizeof c10);
    dump("cmp_d4", c4, sizeof c4);
    dump("cmp_d1", c1, sizeof c1);
    dump("dec_d10", d10, sizeof d10);
    dump("dec_d4", d4, sizeof d4);
  }

  /* CBD eta=2 */
  {
    static uint8_t buf[NT][2 * MLKEM_N / 4];
    for (t = 0; t < NT; t++)
    {
      for (i = 0; i < 2 * MLKEM_N / 4; i++) buf[t][i] = (uint8_t)rnd();
      mlk_poly_cbd2(&out[t], buf[t]);
    }
    dump("cbd2_in", buf, sizeof buf);
    dump("cbd2_out", out, sizeof out);
  }

  /* Keccak-f[1600] */
  {
    static uint64_t st[NT][25], st2[NT][25];
    for (t = 0; t < NT; t++)
      for (i = 0; i < 25; i++) st[t][i] = rnd();
    memcpy(st2, st, sizeof st);
    for (t = 0; t < NT; t++) mlk_keccakf1600_permute(st2[t]);
    dump("keccak_in", st, sizeof st);
    dump("keccak_out", st2, sizeof st2);
  }

  /* full KEM, deterministic */
  {
    uint8_t coins[64], ecoins[32];
    static uint8_t pk[MLKEM_INDCCA_PUBLICKEYBYTES], sk[MLKEM_INDCCA_SECRETKEYBYTES];
    static uint8_t ct[MLKEM_INDCCA_CIPHERTEXTBYTES];
    uint8_t ss1[32], ss2[32];
    for (i = 0; i < 64; i++) coins[i] = (uint8_t)rnd();
    for (i = 0; i < 32; i++) ecoins[i] = (uint8_t)rnd();
    if (mlk_kem_keypair_derand(pk, sk, coins, NULL)) return 2;
    if (mlk_kem_enc_derand(ct, ss1, pk, ecoins, NULL)) return 3;
    if (mlk_kem_dec(ss2, ct, sk, NULL)) return 4;
    if (memcmp(ss1, ss2, 32)) return 5;
    dump("kem_coins", coins, 64);
    dump("kem_ecoins", ecoins, 32);
    dump("kem_pk", pk, sizeof pk);
    dump("kem_sk", sk, sizeof sk);
    dump("kem_ct", ct, sizeof ct);
    dump("kem_ss", ss1, 32);
    /* implicit rejection on crafted ciphertexts: one flipped bit, c1 = 0, all-zero ct */
    {
      static uint8_t bad[3][MLKEM_INDCCA_CIPHERTEXTBYTES];
      uint8_t ssb[3][32];
      int v;
      memcpy(bad[0], ct, sizeof ct);
      bad[0][5] ^= 1;
      memcpy(bad[1], ct, sizeof ct);
      memset(bad[1], 0, MLKEM_POLYVECCOMPRESSEDBYTES_DU);
      memset(bad[2], 0, sizeof ct);
      for (v = 0; v < 3; v++)
        if (mlk_kem_dec(ssb[v], bad[v], sk, NULL)) return 6;
      dump("kem_bad_ct", bad, sizeof bad);
      dump("kem_bad_ss", ssb, sizeof ssb);
    }
  }

  /* Barrett (centred representative) over every int16 */
  {
    static int16_t bin[65536], bout[65536];
    for (i = 0; i < 65536; i++)
    {
      bin[i] = (int16_t)(i - 32768);
      bout[i] = mlk_barrett_reduce(bin[i]);
    }
    dump("barrett_in", bin, sizeof bin);
    dump("barrett_out", bout, sizeof bout);
  }

  /* poly_add / poly_sub: |x|, |y| < q */
  {
    static mlk_poly xa[NT], xb[NT], xs[NT], xd[NT];
    for (t = 0; t < NT; t++)
      for (i = 0; i < MLKEM_N; i++)
      {
        xa[t].coeffs[i] = rnd_range(-MLKEM_Q + 1, MLKEM_Q);
        xb[t].coeffs[i] = rnd_range(-MLKEM_Q + 1, MLKEM_Q);
      }
    memcpy(xs, xa, sizeof xa);
    memcpy(xd, xa, sizeof xa);
    for (t = 0; t < NT; t++)
    {
      mlk_poly_add(&xs[t], &xb[t]);
      mlk_poly_sub(&xd[t], &xb[t]);
    }
    dump("add_a", xa, sizeof xa);
    dump("add_b", xb, sizeof xb);
    dump("add_out", xs, sizeof xs);
    dump("sub_out", xd, sizeof xd);
  }

  /* poly_frommsg (decompress d=1) */
  {
    static uint8_t msg[NT][MLKEM_INDCPA_MSGBYTES];
    static mlk_poly fm[NT];
    for (t = 0; t < NT; t++)
    {
      for (i = 0; i < MLKEM_INDCPA_MSGBYTES; i++) msg[t][i] = (uint8_t)rnd();
      mlk_poly_frommsg(&fm[t], msg[t]);
    }
    dump("msg_in", msg, sizeof msg);
    dump("frommsg_out", fm, sizeof fm);
  }

  /* compress / decompress d = 5, 11 (ML-KEM-1024 parameters; scalar functions) */
  {
    static uint16_t c5[NT * MLKEM_N], c11[NT * MLKEM_N];
    static int16_t d5[NT * MLKEM_N], d11[NT * MLKEM_N], ci[NT * MLKEM_N];
    for (i = 0; i < NT * MLKEM_N; i++)
    {
      ci[i] = (int16_t)(i < MLKEM_Q ? i : rnd_range(0, MLKEM_Q));
      c5[i] = mlk_scalar_compress_d5(ci[i]);
      c11[i] = mlk_scalar_compress_d11(ci[i]);
      d5[i] = mlk_scalar_decompress_d5((uint8_t)(i % 32));
      d11[i] = mlk_scalar_decompress_d11((uint16_t)(i % 2048));
    }
    dump("cmp5_in", ci, sizeof ci);
    dump("cmp_d5", c5, sizeof c5);
    dump("cmp_d11", c11, sizeof c11);
    dump("dec_d5", d5, sizeof d5);
    dump("dec_d11", d11, sizeof d11);
  }

  /* fresh random seeds: keygen / encaps / decaps outputs, plus the input checks' error returns */
  {
#define NSEED 6
    static uint8_t sc[NSEED][64], se[NSEED][32];
    static uint8_t spk[NSEED][MLKEM_INDCCA_PUBLICKEYBYTES], ssk[NSEED][MLKEM_INDCCA_SECRETKEYBYTES];
    static uint8_t sct[NSEED][MLKEM_INDCCA_CIPHERTEXTBYTES], sss[NSEED][32], sss2[NSEED][32];
    static uint8_t badpk[MLKEM_INDCCA_PUBLICKEYBYTES], badsk[MLKEM_INDCCA_SECRETKEYBYTES];
    uint8_t tct[MLKEM_INDCCA_CIPHERTEXTBYTES], tss[32];
    int32_t rc[2];
    int v;
    for (v = 0; v < NSEED; v++)
    {
      for (i = 0; i < 64; i++) sc[v][i] = (uint8_t)rnd();
      for (i = 0; i < 32; i++) se[v][i] = (uint8_t)rnd();
      if (mlk_kem_keypair_derand(spk[v], ssk[v], sc[v], NULL)) return 7;
      if (mlk_kem_enc_derand(sct[v], sss[v], spk[v], se[v], NULL)) return 8;
      if (mlk_kem_dec(sss2[v], sct[v], ssk[v], NULL)) return 9;
      if (memcmp(sss[v], sss2[v], 32)) return 10;
    }
    /* invalid ek: first coefficient set to q (bytes 0..1 = 0x01, 0x0D low nibble) */
    memcpy(badpk, spk[0], sizeof badpk);
    badpk[0] = (uint8_t)(MLKEM_Q & 0xFF);
    badpk[1] = (uint8_t)((badpk[1] & 0xF0) | (MLKEM_Q >> 8));
    rc[0] = mlk_kem_enc_derand(tct, tss, badpk, se[0], NULL);
    /* invalid dk: one bit of h flipped */
    memcpy(badsk, ssk[0], sizeof badsk);
    badsk[MLKEM_INDCCA_SECRETKEYBYTES - 64] ^= 1;
    rc[1] = mlk_kem_dec(tss, sct[0], badsk, NULL);
    dump("seed_coins", sc, sizeof sc);
    dump("seed_ecoins", se, sizeof se);
    dump("seed_pk", spk, sizeof spk);
    dump("seed_sk", ssk, sizeof ssk);
    dump("seed_ct", sct, sizeof sct);
    dump("seed_ss", sss, sizeof sss);
    dump("bad_pk", badpk, sizeof badpk);
    dump("bad_sk", badsk, sizeof badsk);
    dump("bad_rc", rc, sizeof rc);
  }

  return 0;
}

/* only the *_derand APIs are used; this satisfies the link and must never run */
int randombytes(uint8_t *out, size_t outlen);
int randombytes(uint8_t *out, size_t outlen)
{
  (void)out;
  (void)outlen;
  abort();
}
