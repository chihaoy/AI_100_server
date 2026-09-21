// One resident QPC, one input, any number of outputs. Measures synchronous host
// latency (set data + enqueue + wait), excluding model setup and warm-up.
// Build and invocation are managed by moe_multilayer_bench.py.
#include "QAicApi.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#define CK(call) do { auto status = (call); if (status != QS_SUCCESS) { \
  fprintf(stderr, "%s failed: %d\n", #call, int(status)); return 1; } } while (0)
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: %s QPC_DIR INPUT_F16 OUT_DIR ITERATIONS WARMUP\n", argv[0]);
    return 2;
  }
  const std::string qpcdir = argv[1], outdir = argv[3];
  const int iterations = std::stoi(argv[4]), warmup = std::stoi(argv[5]);
  if (iterations < 1 || warmup < 0 || std::filesystem::exists(outdir)) {
    fprintf(stderr, "Require positive iterations, nonnegative warmup, new output directory\n");
    return 2;
  }
  std::ifstream input(argv[2], std::ios::binary);
  if (!input) { fprintf(stderr, "Cannot open input\n"); return 1; }
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(input)), {});
  std::filesystem::create_directories(outdir);
  auto setup = Clock::now();
  QID devices[] = {0, 1, 2, 3};
  QAicContext *context = nullptr;
  QAicQueue *queue = nullptr;
  QAicQpcObj *qpc = nullptr;
  QAicProgram *program = nullptr;
  QAicExecObj *exec = nullptr;
  CK(qaicCreateContext(&context, nullptr, 4, devices, nullptr, nullptr, nullptr, nullptr));
  CK(qaicCreateQueue(context, &queue, nullptr, devices[0]));
  CK(qaicOpenQpcFile(&qpc, (qpcdir + "/programqpc.bin").c_str()));
  QAicProgramProperties_t properties;
  CK(qaicProgramPropertiesInitDefault(&properties));
  properties.devMapping = "0:1:2:3";
  CK(qaicCreateProgram(context, &program, &properties, devices[0], "moe_multilayer", qpc));
  CK(qaicLoadProgram(program));
  CK(qaicRunActivationCmd(program, QAIC_PROGRAM_CMD_ACTIVATE_FULL));
  QData descriptor{};
  CK(qaicProgramGetIoDescriptor(program, &descriptor));
  QAicExecObjProperties_t ep = QAIC_EXECOBJ_PROPERTIES_DEFAULT;
  CK(qaicCreateExecObj(context, &exec, &ep, program, &descriptor, nullptr, nullptr));
  QAicIoBufferInfo_t *info = nullptr;
  CK(qaicProgramGetIoBufferInfo(program, &info));
  std::vector<QBuffer> buffers(info->numBufferMappings, QBuffer{});
  int inputs = 0;
  for (uint32_t i = 0; i < info->numBufferMappings; ++i) {
    const auto &mapping = info->bufferMappings[i];
    if (mapping.index >= buffers.size()) return 1;
    auto &buffer = buffers[mapping.index];
    buffer.size = mapping.size;
    buffer.buf = static_cast<uint8_t *>(aligned_alloc(4096, (buffer.size + 4095) / 4096 * 4096));
    if (!buffer.buf) return 1;
    buffer.type = QBUFFER_TYPE_HEAP;
    memset(buffer.buf, 0, buffer.size);
    if (mapping.ioType == BUFFER_IO_TYPE_INPUT) {
      ++inputs;
      if (data.size() != buffer.size) {
        fprintf(stderr, "Input size mismatch: file %zu, buffer %zu\n", data.size(), buffer.size);
        return 1;
      }
      memcpy(buffer.buf, data.data(), data.size());
    }
    printf("buffer %u: %s size=%u dtype=%d io=%d\n", mapping.index,
           mapping.bufferName, mapping.size, int(mapping.dataType), int(mapping.ioType));
  }
  if (inputs != 1) { fprintf(stderr, "Expected exactly one input\n"); return 1; }
  const double setup_ms = elapsed(setup);
  std::vector<double> times;
  for (int i = -warmup; i < iterations; ++i) {
    auto start = Clock::now();
    CK(qaicExecObjSetData(exec, buffers.size(), buffers.data()));
    CK(qaicEnqueueExecObj(queue, exec, nullptr));
    CK(qaicExecObjWaitForCompletion(exec, 60000000));
    if (i >= 0) times.push_back(elapsed(start));
  }
  for (uint32_t i = 0; i < info->numBufferMappings; ++i) {
    const auto &mapping = info->bufferMappings[i];
    if (mapping.ioType != BUFFER_IO_TYPE_OUTPUT) continue;
    std::string name = mapping.bufferName;
    std::replace(name.begin(), name.end(), '/', '_');
    std::ofstream output(outdir + "/" + name + ".bin", std::ios::binary);
    const auto &buffer = buffers[mapping.index];
    output.write(reinterpret_cast<const char *>(buffer.buf), buffer.size);
    if (!output) return 1;
  }
  std::ofstream timing(outdir + "/timing.json");
  timing << "{\"setup_ms\":" << setup_ms << ",\"warmup\":" << warmup << ",\"samples_ms\":[";
  for (size_t i = 0; i < times.size(); ++i) timing << (i ? "," : "") << times[i];
  timing << "]}\n";
  timing.close();
  std::sort(times.begin(), times.end());
  printf("HOST_LATENCY_MS median=%.4f p10=%.4f p90=%.4f setup=%.2f\n",
         times[times.size()/2], times[times.size()/10], times[times.size()*9/10], setup_ms);
  CK(qaicReleaseExecObj(exec));
  CK(qaicRunActivationCmd(program, QAIC_PROGRAM_CMD_DEACTIVATE_FULL));
  CK(qaicUnloadProgram(program));
  CK(qaicReleaseProgram(program));
  CK(qaicCloseQpc(qpc));
  CK(qaicReleaseQueue(queue));
  CK(qaicReleaseContext(context));
  for (auto &buffer : buffers) free(buffer.buf);
  return 0;
}
