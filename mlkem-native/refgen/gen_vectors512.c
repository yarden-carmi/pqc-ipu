#define _POSIX_C_SOURCE 200809L
/* CBD eta=3 reference vectors: mlkem-native built for ML-KEM-512 (eta1 = 3). Host compiler only. */
#define MLK_CONFIG_PARAMETER_SET 512
#define MLK_CONFIG_NAMESPACE_PREFIX mlkref512
#include "src/common.h"
#include "src/sampling.c"
#include "src/fips202/fips202.c"
#include "src/fips202/fips202x4.c"
#include "src/fips202/keccakf1600.c"

#include <stdio.h>
#include <stdlib.h>

static uint64_t rng_s = 0x2545F4914F6CDD1Dull;
static uint64_t rnd(void)
{
  rng_s ^= rng_s << 13;
  rng_s ^= rng_s >> 7;
  rng_s ^= rng_s << 17;
  return rng_s;
}

static void dump(const char *dir, const char *name, const void *p, size_t n)
{
  char path[512];
  FILE *f;
  snprintf(path, sizeof path, "%s/%s.bin", dir, name);
  f = fopen(path, "wb");
  if (!f || fwrite(p, 1, n, f) != n)
  {
    perror(path);
    exit(1);
  }
  fclose(f);
}

int main(int argc, char **argv)
{
  static uint8_t buf[16][3 * MLKEM_N / 4];
  static mlk_poly out[16];
  int t, i;
  if (argc < 2) return 1;
  for (t = 0; t < 16; t++)
  {
    for (i = 0; i < 3 * MLKEM_N / 4; i++) buf[t][i] = (uint8_t)rnd();
    mlk_poly_cbd3(&out[t], buf[t]);
  }
  dump(argv[1], "cbd3_in", buf, sizeof buf);
  dump(argv[1], "cbd3_out", out, sizeof out);
  return 0;
}

int randombytes(uint8_t *out, size_t outlen);
int randombytes(uint8_t *out, size_t outlen)
{
  (void)out;
  (void)outlen;
  abort();
}
