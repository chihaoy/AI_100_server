// TokenCombine utility functions: verification, kernel selection, shape inference and output-driven tiling.
#include "CustomOpFunctions.h"
#include "CustomOpInterpreterInterface.h"
#include "CustomOpTileConfigHelpers.h"
#include "CustomOpTypes.h"

#include <string.h>

extern "C" {
bool customOpVerify(const CustomOpPropertiesHandle *const opProp) {
  if (opProp->paramsSize != 1 || opProp->inputsSize != 2 || opProp->outputsSize != 1)
    return false;
  const CustomOpParam &k = opProp->params[0];
  if (strcmp(k.name, "k") || k.size != 0 || getIntParam(k) < 1)
    return false;
  const CustomOpIOTensor &data = opProp->inputs[0], &idx = opProp->inputs[1], &out = opProp->outputs[0];
  if (data.rank != 3 || idx.rank != 2 || out.rank != 3 || data.dtype != DTFloat16 || out.dtype != DTFloat16)
    return false;
  if (idx.dtype != DTIInt32 && idx.dtype != DTQInt32)
    return false;
  const int32_t kk = getIntParam(k);
  return idx.untiledSizes[0] == data.untiledSizes[0] && idx.untiledSizes[1] % kk == 0 && out.untiledSizes[0] == data.untiledSizes[0] &&
         out.untiledSizes[1] == idx.untiledSizes[1] / kk && out.untiledSizes[2] == data.untiledSizes[2];
}

const char *customOpSelectImpl(const CustomOpPropertiesHandle *const opProp, const CustomOpKernelInfo *const kernelInfos,
                               const int32_t numKernels, const char *backend) {
  if (strcmp(backend, "AIC") == 0)
    return "TokenCombineAIC";
  if (strcmp(backend, "Interpreter") == 0)
    return "TokenCombineInterpreter";
  return nullptr;
}

bool customOpInferShape(CustomOpPropertiesHandle *const opProp) {
  const CustomOpIOTensor &data = opProp->inputs[0], &idx = opProp->inputs[1];
  CustomOpIOTensor &out = opProp->outputs[0];
  const int32_t k = getIntParam(opProp->params[0]);
  out.rank = 3;
  out.dtype = data.dtype;
  out.sizes[0] = data.sizes[0];
  out.sizes[1] = idx.sizes[1] / k;
  out.sizes[2] = data.sizes[2];
  return true;
}

// The output may be tiled along devices, tokens and columns; tiles are concatenated.
bool customOpSetProperties(CustomOpPropertiesHandle *const opProp) {
  setTileConfig(opProp, "out", {0, 1, 2});
  return true;
}

// An output tile [b0:b1, t0:t1, d0:d1] reads data[b0:b1, all rows, d0:d1] and idx[b0:b1, t0*k:t1*k].
bool customOpMapTiles(CustomOpPropertiesHandle *const opProp) {
  const CustomOpIOTensor &out = opProp->outputs[0];
  const int32_t k = getIntParam(opProp->params[0]);
  const std::vector<int32_t> s = tileStartIndices(out), z = tileSizes(out);
  const int32_t rows = opProp->inputs[0].untiledSizes[1];
  createInputTile(opProp, 0, {s[0], 0, s[2]}, {s[0] + z[0], rows, s[2] + z[2]});
  createInputTile(opProp, 1, {s[0], s[1] * k}, {s[0] + z[0], (s[1] + z[1]) * k});
  return true;
}

void customOpDeallocateMemory(CustomOpPropertiesHandle *const opProp) {
  CustomOpTileConfigHelpers::destroyTileConfigsAndMergeConfigs(opProp);
}
}
