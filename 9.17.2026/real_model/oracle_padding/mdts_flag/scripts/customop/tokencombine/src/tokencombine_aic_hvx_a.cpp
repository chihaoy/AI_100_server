// TokenCombine device kernel, HVX: 64 columns per vector, each token's k rows widened to FP32 (vcvt sf<-hf), summed with IEEE
// single-precision vector adds and narrowed once (vcvt hf<-sf, operand order "hi, lo"; the headers do not document the lane order of
// the widening pair, so both orders are built and the small four-SoC test picks the one that matches the reference).
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
  const HVX_Vector zero = Q6_V_vsplat_R(0);
  for (int32_t b = 0; b < B; b++)
    for (int32_t t = threadId; t < T; t += 4) {
      const int32_t *row = ip + (ptrdiff_t)b * M + (ptrdiff_t)t * k;
      float16_ty *o = op + ((ptrdiff_t)b * T + t) * D;
      int32_t d0 = 0;
      for (; d0 + 64 <= D; d0 += 64) {
        HVX_Vector lo = zero, hi = zero;
        for (int32_t j = 0; j < k; j++) {
          const int32_t r = row[j];
          if (r < 0 || r >= N) continue;
          const HVX_VectorPair w = Q6_Wsf_vcvt_Vhf(*(const HVX_UVector *)(dp + ((ptrdiff_t)b * N + r) * DD + d0));
          lo = Q6_Vsf_vadd_VsfVsf(lo, Q6_V_lo_W(w));
          hi = Q6_Vsf_vadd_VsfVsf(hi, Q6_V_hi_W(w));
        }
        *(HVX_UVector *)(o + d0) = Q6_Vhf_vcvt_VsfVsf(hi, lo);
      }
      for (; d0 < D; d0++) {   // columns past the last full vector
        float acc = 0.0f;
        for (int32_t j = 0; j < k; j++) {
          const int32_t r = row[j];
          if (r >= 0 && r < N) acc += (float)dp[((ptrdiff_t)b * N + r) * DD + d0];
        }
        o[d0] = (float16_ty)acc;
      }
    }
}
}
