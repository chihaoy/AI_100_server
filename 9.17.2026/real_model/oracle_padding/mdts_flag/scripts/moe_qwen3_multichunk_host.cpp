// Multi-chunk prefill on one loaded program with the KV cache retained on the device (experiment 4a, E40).
// Chunk k feeds tokens [kT, (k+1)T) at positions kT..(k+1)T-1, so it attends to the cache written by chunks 0..k-1.
// Capacity profiles: a program may carry int32 inputs (cap_rows1, cap_rows2, ..., filled with 0..len-1, and cap_tag, zeros)
// whose lengths select a compiled specialization; each profile lists those lengths and the capacity of each of the 128
// lanes in routing_counts order. routing_counts may cover fewer than 48 layers (two-layer cuts).
// After every attempt the routing counts are checked against the profile's lane capacities (overflow detection); an
// overflowing attempt is rejected and the chunk is rerun with the fallback profile, which rewrites the chunk's KV
// entries at the same positions. The whole sequence is repeated REPEATS times; accepted logits must repeat exactly.
// Plan file (whitespace separated):
//   PROFILES <P>
//   <name> <n_inputs> [<input name> <length>]... <128 lane capacities>      (P lines)
//   PRIMARY <index> FALLBACK <index or -1>
//   FORCE <chunk> <profile>                                                  (optional, first attempt of that chunk)
//   CAPTURE                                (optional: every chunk is an independent prefill at positions 0..T-1, the
//                                           accepted routing counts of all chunks go to counts_all.bin, no logits files)
// Usage: multichunk_host QPC_DIR IDS_I64 OUT T REPEATS PLAN
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
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point start) { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
static void check(QStatus status, const char *name) {
  if (status != QS_SUCCESS) throw std::runtime_error(std::string(name) + ": " + std::to_string(status));
}
#define CK(call) check(call, #call)
static std::vector<uint8_t> read_file(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("Cannot read " + path);
  return {(std::istreambuf_iterator<char>(file)), {}};
}
struct Profile { std::string name; std::map<std::string, uint32_t> inputs; std::vector<int> caps; };
struct Runtime {
  QAicContext *context = nullptr; QAicQueue *queue = nullptr; QAicQpcObj *qpc = nullptr; QAicProgram *program = nullptr;
  QAicExecObj *exec = nullptr; bool loaded = false, active = false; std::vector<QBuffer> buffers;
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
  if (argc != 7) throw std::runtime_error("usage: multichunk_host QPC_DIR IDS_I64 OUT T REPEATS PLAN");
  const std::string qpc_dir = argv[1], out = argv[3];
  const auto ids_bytes = read_file(argv[2]);
  const int T = std::stoi(argv[4]), repeats = std::stoi(argv[5]);
  const auto *all_ids = reinterpret_cast<const int64_t *>(ids_bytes.data());
  if (ids_bytes.size() % (8 * T)) throw std::runtime_error("Token file is not a whole number of chunks");
  const int K = int(ids_bytes.size() / (8 * T));
  std::vector<Profile> profiles; int primary = -1, fallback = -1; std::map<int, int> force; bool capture = false;
  {
    std::ifstream plan(argv[6]); std::string word; int P = 0;
    if (!(plan >> word >> P) || word != "PROFILES" || P < 1) throw std::runtime_error("Bad plan header");
    for (int p = 0; p < P; ++p) {
      Profile pr; int n = 0; plan >> pr.name >> n;
      for (int i = 0; i < n; ++i) { std::string nm; uint32_t len; plan >> nm >> len; pr.inputs[nm] = len; }
      pr.caps.resize(128); for (auto &c : pr.caps) plan >> c;
      if (!plan) throw std::runtime_error("Bad profile line");
      profiles.push_back(pr);
    }
    if (!(plan >> word >> primary) || word != "PRIMARY" || !(plan >> word >> fallback) || word != "FALLBACK") throw std::runtime_error("Bad PRIMARY/FALLBACK");
    while (plan >> word) {
      if (word == "CAPTURE") { capture = true; continue; }
      int c, p; if (word != "FORCE" || !(plan >> c >> p)) throw std::runtime_error("Bad FORCE"); force[c] = p;
    }
    if (primary < 0 || primary >= P || fallback >= P) throw std::runtime_error("Bad profile index");
  }
  if (repeats < 1 || std::filesystem::exists(out)) throw std::runtime_error("Require positive repeats and a new output directory");
  std::filesystem::create_directories(out);
  std::ofstream setup(out + "/setup.csv"), attempts(out + "/attempts.csv");
  setup << std::setprecision(12) << "phase,ms\n";
  attempts << std::setprecision(12) << "repeat,chunk,attempt,profile,host_ms,accepted,overflow_lanes,first_overflow_layer,max_excess\n";
  for (QID device = 0; device < 4; ++device) {
    QResourceInfo info{}; CK(qaicGetResourceInfo(device, &info));
    if (info.nspFree != 16) throw std::runtime_error("A selected card is already in use");
  }
  Runtime rt;
  auto t_setup = Clock::now(), start = t_setup;
  QID devices[] = {0, 1, 2, 3};
  CK(qaicCreateContext(&rt.context, nullptr, 4, devices, nullptr, nullptr, nullptr, nullptr));
  CK(qaicCreateQueue(rt.context, &rt.queue, nullptr, 0));
  CK(qaicOpenQpcFile(&rt.qpc, (qpc_dir + "/programqpc.bin").c_str()));
  QAicProgramProperties_t properties; CK(qaicProgramPropertiesInitDefault(&properties)); properties.devMapping = "0:1:2:3";
  CK(qaicCreateProgram(rt.context, &rt.program, &properties, 0, "qwen3_multichunk", rt.qpc));
  start = Clock::now(); CK(qaicLoadProgram(rt.program)); rt.loaded = true; setup << "load_program," << elapsed(start) << '\n';
  start = Clock::now(); CK(qaicRunActivationCmd(rt.program, QAIC_PROGRAM_CMD_ACTIVATE_FULL)); rt.active = true; setup << "activate_program," << elapsed(start) << '\n';
  QData descriptor{}; CK(qaicProgramGetIoDescriptor(rt.program, &descriptor));
  QAicExecObjProperties_t exec_properties = QAIC_EXECOBJ_PROPERTIES_DEFAULT;
  CK(qaicCreateExecObj(rt.context, &rt.exec, &exec_properties, rt.program, &descriptor, nullptr, nullptr));
  QAicIoBufferInfo_t *io = nullptr; CK(qaicProgramGetIoBufferInfo(rt.program, &io));
  const uint32_t nb = io->numBufferMappings; rt.buffers.resize(nb);
  std::vector<QBufferDimensions> dims(nb); std::vector<std::vector<uint32_t>> shapes(nb); std::vector<std::string> names(nb);
  int ids_i = -1, pos_i = -1, logits_i = -1, counts_i = -1;
  for (uint32_t k = 0; k < nb; ++k) {
    const auto &m = io->bufferMappings[k]; uint32_t i = m.index;
    if (i >= nb) throw std::runtime_error("Invalid IO mapping");
    std::string name = m.bufferName; name = name.substr(name.find_last_of('/') + 1); names[i] = name;
    size_t cap_bytes = m.size;
    if (name.rfind("cap_", 0) == 0) { uint32_t mx = 0; for (auto &p : profiles) mx = std::max(mx, p.inputs.at(name)); cap_bytes = 4ull * mx; }
    auto &b = rt.buffers[i]; b.size = m.size; b.type = QBUFFER_TYPE_HEAP;
    b.buf = static_cast<uint8_t *>(aligned_alloc(4096, (cap_bytes + 4095) / 4096 * 4096)); if (!b.buf) throw std::bad_alloc();
    std::memset(b.buf, 0, cap_bytes);
    if (name == "input_ids") { ids_i = i; shapes[i] = {1, uint32_t(T)}; dims[i].sizeOfElem = 8; }
    else if (name == "position_ids") { pos_i = i; shapes[i] = {1, uint32_t(T)}; dims[i].sizeOfElem = 8; }
    else if (name == "logits") { logits_i = i; shapes[i] = {1, 1, 151936}; dims[i].sizeOfElem = 4; }
    else if (name == "routing_counts") { counts_i = i; shapes[i] = {uint32_t(m.size / 512), 128}; dims[i].sizeOfElem = 4; }
    else if (name.rfind("cap_", 0) == 0) { shapes[i] = {0}; dims[i].sizeOfElem = 4; }
    else throw std::runtime_error("Unexpected IO " + name);
    dims[i].count = shapes[i].size(); dims[i].dims = shapes[i].data();
    std::cout << "IO " << name << " bytes=" << m.size << '\n';
  }
  if (ids_i < 0 || pos_i < 0 || logits_i < 0 || counts_i < 0) throw std::runtime_error("Need input_ids, position_ids, logits and routing_counts");
  const int layers = int(rt.buffers[counts_i].size / 512);
  if (rt.buffers[logits_i].size != 151936 * 4 || rt.buffers[counts_i].size != size_t(layers) * 512 || layers < 1 || layers > 48) throw std::runtime_error("Unexpected output sizes");
  setup << "total_setup," << elapsed(t_setup) << '\n'; setup.flush();
  auto select = [&](const Profile &p) {
    for (uint32_t i = 0; i < nb; ++i) if (names[i].rfind("cap_", 0) == 0) {   // cap_rows<t>: 0..len-1; cap_tag: zeros
      uint32_t len = p.inputs.at(names[i]); shapes[i][0] = len; rt.buffers[i].size = 4 * len;
      auto *v = reinterpret_cast<int32_t *>(rt.buffers[i].buf); for (uint32_t r = 0; r < len; ++r) v[r] = names[i] == "cap_tag" ? 0 : int32_t(r);
    }
  };
  std::vector<std::vector<uint8_t>> accepted(capture ? 0 : K);
  std::ofstream counts_all;
  if (capture) counts_all.open(out + "/counts_all.bin", std::ios::binary);
  for (int rep = 0; rep < repeats; ++rep) {
    for (int c = 0; c < K; ++c) {
      std::memcpy(rt.buffers[ids_i].buf, all_ids + size_t(c) * T, 8 * T);
      auto *pos = reinterpret_cast<int64_t *>(rt.buffers[pos_i].buf); for (int t = 0; t < T; ++t) pos[t] = (capture ? 0 : int64_t(c) * T) + t;
      int prof = force.count(c) ? force[c] : primary;
      for (int attempt = 0;; ++attempt) {
        const Profile &p = profiles[prof]; select(p);
        start = Clock::now();
        CK(qaicExecObjSetDataExt(rt.exec, nb, rt.buffers.data(), dims.data()));
        CK(qaicEnqueueExecObj(rt.queue, rt.exec, nullptr));
        CK(qaicExecObjWaitForCompletion(rt.exec, 60000000));
        const double ms = elapsed(start);
        const auto *counts = reinterpret_cast<const int32_t *>(rt.buffers[counts_i].buf);
        int over = 0, first = -1, excess = 0;
        for (int L = 0; L < layers; ++L) {
          int total = 0;
          for (int e = 0; e < 128; ++e) { int n = counts[L * 128 + e]; total += n; if (n > p.caps[e]) { ++over; excess = std::max(excess, n - p.caps[e]); if (first < 0) first = L; } }
          if (total != 8 * T) throw std::runtime_error("Invalid routing assignment total at layer " + std::to_string(L));
        }
        const bool ok = over == 0 || fallback < 0;
        attempts << rep << ',' << c << ',' << attempt << ',' << p.name << ',' << ms << ',' << ok << ',' << over << ',' << first << ',' << excess << '\n';
        if (rep == 0 && !capture) {
          std::ofstream f(out + "/counts_chunk" + std::to_string(c) + "_attempt" + std::to_string(attempt) + ".bin", std::ios::binary);
          f.write(reinterpret_cast<const char *>(counts), layers * 512);
          std::ofstream g(out + "/logits_chunk" + std::to_string(c) + "_attempt" + std::to_string(attempt) + ".bin", std::ios::binary);
          g.write(reinterpret_cast<const char *>(rt.buffers[logits_i].buf), 151936 * 4);
        }
        if (ok && capture) { if (rep == 0) counts_all.write(reinterpret_cast<const char *>(counts), layers * 512); break; }
        if (ok) {
          const auto *lb = rt.buffers[logits_i].buf; std::vector<uint8_t> v(lb, lb + 151936 * 4);
          for (size_t q = 0; q < 151936; ++q) if (!std::isfinite(reinterpret_cast<const float *>(lb)[q])) throw std::runtime_error("Nonfinite logits");
          if (rep == 0) accepted[c] = v; else if (v != accepted[c]) throw std::runtime_error("Repeated sequence changed the logits of chunk " + std::to_string(c));
          break;
        }
        if (attempt > 0 || prof == fallback) throw std::runtime_error("Fallback profile overflowed at chunk " + std::to_string(c));
        prof = fallback;
      }
    }
    attempts.flush(); std::cout << "REPEAT_OK " << rep << std::endl;
  }
  for (int c = 0; c < int(accepted.size()); ++c) { std::ofstream f(out + "/logits_chunk" + std::to_string(c) + ".bin", std::ios::binary); f.write(reinterpret_cast<const char *>(accepted[c].data()), accepted[c].size()); }
  std::cout << "MULTICHUNK_CHECKS_PASSED chunks=" << K << " repeats=" << repeats << std::endl;
  return 0;
} catch (const std::exception &error) { std::cerr << error.what() << std::endl; return 1; }
