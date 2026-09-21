// Full-model, fixed-shape prefill benchmark; capacities are chosen at compile time.
// QPC IO is input_ids/position_ids int64, last-token logits float32, and optional
// per-layer routing counts int32 for validating static capacity plans.
#include "QAicApi.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
static void check(QStatus status, const char *name) {
  if (status != QS_SUCCESS)
    throw std::runtime_error(std::string(name) + ": " + std::to_string(status));
}
#define CK(call) check(call, #call)

static std::vector<uint8_t> read_file(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("Cannot read " + path);
  return {(std::istreambuf_iterator<char>(file)), {}};
}
struct Runtime {
  QAicContext *context = nullptr;
  QAicQueue *queue = nullptr;
  QAicQpcObj *qpc = nullptr;
  QAicProgram *program = nullptr;
  QAicExecObj *exec = nullptr;
  bool loaded = false, active = false;
  std::vector<QBuffer> buffers;
  ~Runtime() {
    if (exec) qaicReleaseExecObj(exec);
    if (active) qaicRunActivationCmd(program, QAIC_PROGRAM_CMD_DEACTIVATE_FULL);
    if (loaded) qaicUnloadProgram(program);
    if (program) qaicReleaseProgram(program);
    if (qpc) qaicCloseQpc(qpc);
    if (queue) qaicReleaseQueue(queue);
    if (context) qaicReleaseContext(context);
    for (auto &buffer : buffers) free(buffer.buf);
  }
};

int main(int argc, char **argv) try {
  if (argc < 8 || argc > 10)
    throw std::runtime_error("usage: host QPC_DIR IDS_I64 POS_I64 OUT ITERATIONS ROUNDS WARMUP [--counts [CAPACITIES_I32]]");
  const bool with_counts = argc >= 9;
  if (with_counts && std::string(argv[8]) != "--counts") throw std::runtime_error("Unknown option");
  std::vector<uint8_t> capacity_bytes;
  if (argc == 10) {
    capacity_bytes = read_file(argv[9]);
    if (capacity_bytes.size() != 48 * 2 * sizeof(int32_t)) throw std::runtime_error("Invalid capacity file size");
    const auto *caps = reinterpret_cast<const int32_t *>(capacity_bytes.data());
    if (!std::all_of(caps, caps + 96, [](int32_t c) { return c >= 1 && c <= 128; }))
      throw std::runtime_error("Invalid compiled capacity");
  }
  const std::string qpc_dir = argv[1], out = argv[4];
  const int iterations = std::stoi(argv[5]), rounds = std::stoi(argv[6]), warmup = std::stoi(argv[7]);
  if (iterations < 1 || rounds < 1 || warmup < 0 || std::filesystem::exists(out))
    throw std::runtime_error("Require positive iterations/rounds, nonnegative warmup and a new output directory");
  auto ids = read_file(argv[2]), positions = read_file(argv[3]);
  if (ids.empty() || ids.size() != positions.size() || ids.size() % sizeof(int64_t))
    throw std::runtime_error("Invalid token/position input sizes");
  std::filesystem::create_directories(out);
  std::ofstream resources(out + "/resources.csv"), setup(out + "/setup.csv"), samples(out + "/samples.csv");
  resources << "stage,device,dram_free_kib,nsp_free\n";
  setup << std::setprecision(12) << "phase,ms\n";
  samples << std::setprecision(12) << "round,iteration,host_ms\n";
  auto snapshot = [&](const std::string &stage) {
    for (QID device = 0; device < 4; ++device) {
      QResourceInfo info{};
      CK(qaicGetResourceInfo(device, &info));
      resources << stage << ',' << device << ',' << info.dramFree << ',' << unsigned(info.nspFree) << '\n';
      if (stage == "before" && info.nspFree != 16)
        throw std::runtime_error("A selected card is already in use");
    }
    resources.flush();
  };
  snapshot("before");
  {
    Runtime runtime;
    auto total_setup = Clock::now(), start = total_setup;
    QID devices[] = {0, 1, 2, 3};
    CK(qaicCreateContext(&runtime.context, nullptr, 4, devices, nullptr, nullptr, nullptr, nullptr));
    CK(qaicCreateQueue(runtime.context, &runtime.queue, nullptr, 0));
    CK(qaicOpenQpcFile(&runtime.qpc, (qpc_dir + "/programqpc.bin").c_str()));
    setup << "context_queue_qpc," << elapsed(start) << '\n';
    QAicProgramProperties_t properties;
    CK(qaicProgramPropertiesInitDefault(&properties));
    properties.devMapping = "0:1:2:3";
    start = Clock::now();
    CK(qaicCreateProgram(runtime.context, &runtime.program, &properties, 0, "qwen3_baseline", runtime.qpc));
    setup << "create_program," << elapsed(start) << '\n';
    start = Clock::now();
    CK(qaicLoadProgram(runtime.program));
    runtime.loaded = true;
    setup << "load_program," << elapsed(start) << '\n';
    snapshot("loaded");
    start = Clock::now();
    CK(qaicRunActivationCmd(runtime.program, QAIC_PROGRAM_CMD_ACTIVATE_FULL));
    runtime.active = true;
    setup << "activate_program," << elapsed(start) << '\n';
    snapshot("active");
    QData descriptor{};
    CK(qaicProgramGetIoDescriptor(runtime.program, &descriptor));
    QAicExecObjProperties_t exec_properties = QAIC_EXECOBJ_PROPERTIES_DEFAULT;
    CK(qaicCreateExecObj(runtime.context, &runtime.exec, &exec_properties, runtime.program, &descriptor, nullptr, nullptr));
    QAicIoBufferInfo_t *io = nullptr;
    CK(qaicProgramGetIoBufferInfo(runtime.program, &io));
    if (io->numBufferMappings != (with_counts ? 4u : 3u)) throw std::runtime_error("Unexpected IO count");
    runtime.buffers.resize(io->numBufferMappings);
    int output_index = -1, count_index = -1, input_count = 0;
    std::vector<bool> seen(io->numBufferMappings);
    for (uint32_t k = 0; k < io->numBufferMappings; ++k) {
      const auto &mapping = io->bufferMappings[k];
      auto index = mapping.index;
      if (index >= runtime.buffers.size() || seen[index]) throw std::runtime_error("Invalid IO mapping");
      seen[index] = true;
      auto &buffer = runtime.buffers[index];
      buffer.size = mapping.size;
      buffer.type = QBUFFER_TYPE_HEAP;
      buffer.buf = static_cast<uint8_t *>(aligned_alloc(4096, (buffer.size + 4095) / 4096 * 4096));
      if (!buffer.buf) throw std::bad_alloc();
      std::memset(buffer.buf, 0, buffer.size);
      std::string name = mapping.bufferName;
      name = name.substr(name.find_last_of('/') + 1);
      if (mapping.ioType == BUFFER_IO_TYPE_INPUT) {
        if (mapping.dataType != BUFFER_DATA_TYPE_INT64I) throw std::runtime_error("Expected int64 input");
        const auto &value = name == "input_ids" ? ids : positions;
        if ((name != "input_ids" && name != "position_ids") || value.size() != buffer.size)
          throw std::runtime_error("Unexpected input name/size: " + name);
        std::memcpy(buffer.buf, value.data(), value.size());
        ++input_count;
      } else if (mapping.ioType == BUFFER_IO_TYPE_OUTPUT) {
        if (with_counts && name == "routing_counts") {
          if (buffer.size != 48 * 128 * sizeof(int32_t)) throw std::runtime_error("Unexpected counts size");
          count_index = index;
          continue;
        }
        // This SDK's v1 mapping reports internal FP16 even when custom IO exposes
        // FP32. The Python runner checks the serialized IO descriptor's type.
        if (name != "logits" || buffer.size != 151936 * sizeof(float))
          throw std::runtime_error("Expected Qwen3 last-token float32 logits");
        output_index = index;
      }
      std::cout << "IO " << name << " bytes=" << buffer.size << '\n';
    }
    if (input_count != 2 || output_index < 0 || (with_counts && count_index < 0))
      throw std::runtime_error("Unexpected IO layout");
    setup << "total_setup," << elapsed(total_setup) << '\n';
    setup.flush();
    std::vector<uint8_t> reference;
    std::vector<uint8_t> reference_counts;
    auto &output = runtime.buffers[output_index];
    for (int round = 0; round < rounds; ++round) {
      for (int iteration = -warmup; iteration < iterations; ++iteration) {
        start = Clock::now();
        CK(qaicExecObjSetData(runtime.exec, runtime.buffers.size(), runtime.buffers.data()));
        CK(qaicEnqueueExecObj(runtime.queue, runtime.exec, nullptr));
        CK(qaicExecObjWaitForCompletion(runtime.exec, 60000000));
        const double duration = elapsed(start);
        auto values = reinterpret_cast<const float *>(output.buf);
        if (!std::all_of(values, values + output.size / sizeof(float), [](float x) { return std::isfinite(x); }))
          throw std::runtime_error("Nonfinite logits");
        if (reference.empty()) reference.assign(output.buf, output.buf + output.size);
        else if (std::memcmp(reference.data(), output.buf, output.size))
          throw std::runtime_error("Repeated prefill changed the logits");
        if (with_counts) {
          auto &buffer = runtime.buffers[count_index];
          const auto *counts = reinterpret_cast<const int32_t *>(buffer.buf);
          for (size_t layer = 0; layer < 48; ++layer) {
            int total = 0;
            for (size_t expert = 0; expert < 128; ++expert) {
              auto count = counts[layer * 128 + expert];
              if (count < 0 || count > 128) throw std::runtime_error("Invalid routing count");
              if (!capacity_bytes.empty()) {
                const auto *caps = reinterpret_cast<const int32_t *>(capacity_bytes.data());
                if (count > caps[layer * 2 + expert / 64])
                  throw std::runtime_error("Capacity overflow at layer " + std::to_string(layer) +
                                           ", bank position " + std::to_string(expert));
              }
              total += count;
            }
            if (total != 128 * 8) throw std::runtime_error("Invalid routing assignment total");
          }
          if (reference_counts.empty()) reference_counts.assign(buffer.buf, buffer.buf + buffer.size);
          else if (std::memcmp(reference_counts.data(), buffer.buf, buffer.size))
            throw std::runtime_error("Repeated prefill changed the routing counts");
        }
        if (iteration >= 0) samples << round << ',' << iteration << ',' << duration << '\n';
      }
      samples.flush();
      snapshot("round" + std::to_string(round));
      std::cout << "ROUND_OK " << round << std::endl;
    }
    std::ofstream logits(out + "/logits_f32.bin", std::ios::binary);
    logits.write(reinterpret_cast<const char *>(reference.data()), reference.size());
    logits.close();
    if (!logits) throw std::runtime_error("Failed to write logits");
    if (with_counts) {
      std::ofstream counts(out + "/counts_i32.bin", std::ios::binary);
      counts.write(reinterpret_cast<const char *>(reference_counts.data()), reference_counts.size());
      counts.close();
      if (!counts) throw std::runtime_error("Failed to write counts");
    }
  }
  snapshot("released");
  resources.close(); setup.close(); samples.close();
  if (!resources || !setup || !samples) throw std::runtime_error("Failed to write benchmark data");
  std::cout << "BASELINE_CHECKS_PASSED samples=" << iterations * rounds << std::endl;
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << std::endl;
  return 1;
}
