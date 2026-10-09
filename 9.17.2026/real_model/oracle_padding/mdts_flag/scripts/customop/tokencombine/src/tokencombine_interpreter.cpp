// TokenCombine reference implementation for the interpreter backend (host CPU). FP16 is converted by hand because the host
// compiler (g++ 11) has no _Float16 on x86.
#include "CustomOpInterpreterInterface.h"

#include <stdint.h>
#include <string.h>

static float halfToFloat(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {   // subnormal: normalize
      int32_t e = -1;
      uint32_t m = man;
      do { m <<= 1; e++; } while (!(m & 0x400));
      bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3ff) << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000 | (man << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

static uint16_t floatToHalf(float f) {   // round to nearest even
  uint32_t x;
  memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000;
  int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
  uint32_t man = x & 0x7fffff;
  if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00 | (man ? 0x200 : 0));
  if (exp >= 31) return (uint16_t)(sign | 0x7c00);
  if (exp <= 0) {
    if (exp < -10) return (uint16_t)sign;
    man |= 0x800000;
    const uint32_t shift = (uint32_t)(14 - exp);
    uint32_t h = man >> shift;
    const uint32_t rem = man & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1))) h++;
    return (uint16_t)(sign | h);
  }
  uint32_t h = ((uint32_t)exp << 10) | (man >> 13);
  const uint32_t rem = man & 0x1fff;
  if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
  return (uint16_t)(sign | h);
}

extern "C" {
void TokenCombineInterpreter(CustomOpContext *ctx) {
  const CustomOpIOTensor &data = ctx->inputs[0], &idx = ctx->inputs[1];
  CustomOpIOTensor &out = ctx->outputs[0];
  const int32_t k = getIntParam(ctx->params[0]);
  const int32_t B = out.sizes[0], T = out.sizes[1], D = out.sizes[2], N = data.sizes[1], M = idx.sizes[1];
  const uint16_t *dp = (const uint16_t *)data.data;
  const int32_t *ip = (const int32_t *)idx.data;
  uint16_t *op = (uint16_t *)out.data;
  for (int32_t b = 0; b < B; b++)
    for (int32_t t = 0; t < T; t++)
      for (int32_t d = 0; d < D; d++) {
        float acc = 0.0f;
        for (int32_t j = 0; j < k; j++) {
          const int32_t r = ip[(int64_t)b * M + (int64_t)t * k + j];
          if (r >= 0 && r < N) acc += halfToFloat(dp[((int64_t)b * N + r) * D + d]);
        }
        op[((int64_t)b * T + t) * D + d] = floatToHalf(acc);
      }
}
}
