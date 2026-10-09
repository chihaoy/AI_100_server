// TokenCombine device kernel: correctness first, plain C++ (no HVX intrinsics). Each of the four hardware threads of a core
// takes every fourth token of the core's output tile; a token's k rows are accumulated in FP32, 64 columns at a time.
#include "CustomOpAICInterface.h"
#include "stddef.h"

extern "C" {
void TokenCombineAIC(const CustomOpContext *ctx, const int32_t threadId) {
  const CustomOpIOTensor &data = ctx->inputs[0], &idx = ctx->inputs[1];
  CustomOpIOTensor &out = ctx->outputs[0];
  const int32_t k = getIntParam(ctx->params[0]);
  const int32_t B = out.sizes[0], T = out.sizes[1], D = out.sizes[2], N = data.sizes[1], DD = data.sizes[2], M = idx.sizes[1];
  const float16_ty *dp = (const float16_ty *)data.data;
  const int32_t *ip = (const int32_t *)idx.data;
  float16_ty *op = (float16_ty *)out.data;
  for (int32_t b = 0; b < B; b++)
    for (int32_t t = threadId; t < T; t += 4) {
      const int32_t *row = ip + (ptrdiff_t)b * M + (ptrdiff_t)t * k;
      float16_ty *o = op + ((ptrdiff_t)b * T + t) * D;
      for (int32_t d0 = 0; d0 < D; d0 += 64) {
        const int32_t w = D - d0 < 64 ? D - d0 : 64;
        float acc[64];
        for (int32_t d = 0; d < w; d++) acc[d] = 0.0f;
        for (int32_t j = 0; j < k; j++) {
          const int32_t r = row[j];
          if (r < 0 || r >= N) continue;
          const float16_ty *src = dp + ((ptrdiff_t)b * N + r) * DD + d0;
          for (int32_t d = 0; d < w; d++) acc[d] += (float)src[d];
        }
        for (int32_t d = 0; d < w; d++) o[d0 + d] = (float16_ty)acc[d];
      }
    }
}
}
