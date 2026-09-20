// End-to-end path-2 prefill driver (Qwen3-30B-A3B, T=128): 48 layers, each layer = hot QPC on cards 0,1 (residual + hot
// MoE partial) and cold QPC on cards 2,3 (cold MoE partial), launched together from one host thread; the host adds the two
// outputs (F16C) and feeds the sum to the next layer.  Loads and activates all 96 programs up front.
//   g++ -O2 -std=c++17 -mavx2 -mf16c -I/opt/qti-aic/dev/inc tools/moe_e2e_host.cpp -o /tmp/moe_e2e_host -L/opt/qti-aic/dev/lib/x86_64 -lQAic -Wl,-rpath,/opt/qti-aic/dev/lib/x86_64 -lpthread -ldl
//   /tmp/moe_e2e_host chain <plan.txt: 48 lines "hot_qpc_dir cold_qpc_dir"> <h0.bin fp16 [T,H]> <out_dir> [N=10] [--dump-layers]
//   /tmp/moe_e2e_host naive <qpc_dir> <input_ids_i64.bin> <position_ids_i64.bin> <out_dir> [N=10]
#include "QAicApi.h"
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

#define CK(call) do { QStatus s_ = (call); if (s_ != QS_SUCCESS) { fprintf(stderr, "%s failed: %d\n", #call, (int)s_); exit(1); } } while (0)
static const size_t T = 128, H = 2048, NEL = T * H;
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
static void report(const char *name, std::vector<double> t) { std::sort(t.begin(), t.end()); size_t n = t.size(); printf("  %-46s median %8.2f ms  p10 %8.2f  p90 %8.2f  min %8.2f\n", name, t[n / 2], t[n / 10], t[9 * n / 10], t[0]); }

struct Pair {                       // one context + one queue per card pair, shared by all its programs
  QAicContext *ctx = nullptr; QAicQueue *q = nullptr; std::vector<QID> devs; std::string mapping;
  void open(std::vector<QID> d, const std::string &m) { devs = d; mapping = m; CK(qaicCreateContext(&ctx, nullptr, devs.size(), devs.data(), nullptr, nullptr, nullptr, nullptr)); if (getenv("E2E_DEBUG")) qaicSetLogLevel(ctx, QL_DEBUG); CK(qaicCreateQueue(ctx, &q, nullptr, devs[0])); }
};
struct Prog {
  Pair *pr; QAicQpcObj *qpc = nullptr; QAicProgram *prog = nullptr; QAicExecObj *eo = nullptr;
  uint32_t nbuf = 0; std::vector<QBuffer> bufs; std::vector<int> ins; int out = -1;
  bool lazy = false;
  void activate() { CK(qaicRunActivationCmd(prog, QAIC_PROGRAM_CMD_ACTIVATE_FULL)); }
  void deactivate() { CK(qaicRunActivationCmd(prog, QAIC_PROGRAM_CMD_DEACTIVATE_FULL)); }
  void open(Pair *p, const std::string &dir, const char *name, bool lazy_ = false) {
    lazy = lazy_;
    pr = p; CK(qaicOpenQpcFile(&qpc, (dir + "/programqpc.bin").c_str()));
    QAicProgramProperties_t pp; CK(qaicProgramPropertiesInitDefault(&pp)); if (!pr->mapping.empty()) pp.devMapping = pr->mapping.c_str();
    CK(qaicCreateProgram(pr->ctx, &prog, &pp, pr->devs[0], name, qpc));
    QStatus ls = qaicLoadProgram(prog); if (ls != QS_SUCCESS) { fprintf(stderr, "%s: qaicLoadProgram failed: %d\n", name, (int)ls); exit(1); }
    if (!lazy) { QStatus as = qaicRunActivationCmd(prog, QAIC_PROGRAM_CMD_ACTIVATE_FULL); if (as != QS_SUCCESS) { fprintf(stderr, "%s: activate failed: %d\n", name, (int)as); exit(1); } }
    QData iod{}; CK(qaicProgramGetIoDescriptor(prog, &iod)); QAicExecObjProperties_t ep = QAIC_EXECOBJ_PROPERTIES_DEFAULT;
    CK(qaicCreateExecObj(pr->ctx, &eo, &ep, prog, &iod, nullptr, nullptr));
    QAicIoBufferInfo_t *info = nullptr; CK(qaicProgramGetIoBufferInfo(prog, &info)); nbuf = info->numBufferMappings; bufs.assign(nbuf, QBuffer{});
    for (uint32_t i = 0; i < nbuf; i++) { auto &m = info->bufferMappings[i]; QBuffer &b = bufs[m.index]; b.size = m.size; b.buf = (uint8_t *)aligned_alloc(4096, (m.size + 4095) / 4096 * 4096); b.type = QBUFFER_TYPE_HEAP;
      if (m.ioType == BUFFER_IO_TYPE_INPUT) ins.push_back(m.index); else if (m.ioType == BUFFER_IO_TYPE_OUTPUT) out = m.index; }
    std::sort(ins.begin(), ins.end());
  }
  // --swap mode: the QPC file is opened once (kept in host memory); per layer the program is created+loaded+activated
  // (weights DMA'd to the cards) before use and deactivated+unloaded+released after, because a card cannot hold two
  // 16-core programs at once ("Device 0 does not satisfy the resource requirements", qaicCreateProgram).
  std::string name_; bool up_ = false;
  void open_qpc(Pair *p, const std::string &dir, const char *name) { pr = p; name_ = name; CK(qaicOpenQpcFile(&qpc, (dir + "/programqpc.bin").c_str())); }
  static inline double t_execobj = 0, t_relexec = 0, t_create = 0, t_load = 0, t_activate = 0, t_deact = 0, t_unload = 0, t_release = 0; static inline int n_swaps = 0;
  void up() {
    QAicProgramProperties_t pp; CK(qaicProgramPropertiesInitDefault(&pp)); if (!pr->mapping.empty()) pp.devMapping = pr->mapping.c_str();
    auto a = clk::now(); QStatus cs = qaicCreateProgram(pr->ctx, &prog, &pp, pr->devs[0], name_.c_str(), qpc); if (cs != QS_SUCCESS) { fprintf(stderr, "%s: create failed %d\n", name_.c_str(), (int)cs); exit(1); }
    auto b = clk::now(); QStatus ls = qaicLoadProgram(prog); if (ls != QS_SUCCESS) { fprintf(stderr, "%s: load failed %d\n", name_.c_str(), (int)ls); exit(1); }
    auto c = clk::now(); CK(qaicRunActivationCmd(prog, QAIC_PROGRAM_CMD_ACTIVATE_FULL)); auto d = clk::now();
    t_create += ms(a, b); t_load += ms(b, c); t_activate += ms(c, d); n_swaps++;
    QData iod{}; CK(qaicProgramGetIoDescriptor(prog, &iod)); QAicExecObjProperties_t ep = QAIC_EXECOBJ_PROPERTIES_DEFAULT;
    auto e0 = clk::now(); CK(qaicCreateExecObj(pr->ctx, &eo, &ep, prog, &iod, nullptr, nullptr)); t_execobj += ms(e0, clk::now());
    if (bufs.empty()) { QAicIoBufferInfo_t *info = nullptr; CK(qaicProgramGetIoBufferInfo(prog, &info)); nbuf = info->numBufferMappings; bufs.assign(nbuf, QBuffer{});
      for (uint32_t i = 0; i < nbuf; i++) { auto &m = info->bufferMappings[i]; QBuffer &b = bufs[m.index]; b.size = m.size; b.buf = (uint8_t *)aligned_alloc(4096, (m.size + 4095) / 4096 * 4096); b.type = QBUFFER_TYPE_HEAP;
        if (m.ioType == BUFFER_IO_TYPE_INPUT) ins.push_back(m.index); else if (m.ioType == BUFFER_IO_TYPE_OUTPUT) out = m.index; } std::sort(ins.begin(), ins.end()); }
    up_ = true;
  }
  void down() { auto r0 = clk::now(); CK(qaicReleaseExecObj(eo)); t_relexec += ms(r0, clk::now()); auto a = clk::now(); CK(qaicRunActivationCmd(prog, QAIC_PROGRAM_CMD_DEACTIVATE_FULL)); auto b = clk::now(); CK(qaicUnloadProgram(prog)); auto c = clk::now(); CK(qaicReleaseProgram(prog)); auto d = clk::now();
    t_deact += ms(a, b); t_unload += ms(b, c); t_release += ms(c, d); eo = nullptr; prog = nullptr; up_ = false; }
  void setdata() { CK(qaicExecObjSetData(eo, nbuf, bufs.data())); }
  void launch() { CK(qaicEnqueueExecObj(pr->q, eo, nullptr)); }
  void wait() { CK(qaicExecObjWaitForCompletion(eo, 60000000)); }
};
static void f32_to_f16(uint16_t *dst, const float *src, size_t n) { for (size_t i = 0; i < n; i += 8) _mm_storeu_si128((__m128i *)(dst + i), _mm256_cvtps_ph(_mm256_loadu_ps(src + i), 0)); }
static void add_f16_to_f32(const uint16_t *a, const uint16_t *b, float *y, size_t n) { for (size_t i = 0; i < n; i += 8) _mm256_storeu_ps(y + i, _mm256_add_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(a + i))), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(b + i))))); }
static std::vector<uint8_t> readfile(const std::string &f) { std::ifstream s(f, std::ios::binary); std::vector<uint8_t> v((std::istreambuf_iterator<char>(s)), {}); if (v.empty()) { fprintf(stderr, "cannot read %s\n", f.c_str()); exit(1); } return v; }

int main(int argc, char **argv) {
  if (argc < 5) { fprintf(stderr, "usage: see header\n"); return 1; }
  std::string mode = argv[1];
  if (mode == "naive") {
    std::string qdir = argv[2], out = argv[5]; int N = argc > 6 ? atoi(argv[6]) : 10;
    Pair p; p.open({0, 1, 2, 3}, "0:1:2:3"); Prog r; r.open(&p, qdir, "naive");
    auto ids = readfile(argv[3]), pos = readfile(argv[4]);
    fprintf(stderr, "naive: %u buffers, inputs %zu, out idx %d size %zu\n", r.nbuf, r.ins.size(), r.out, r.bufs[r.out].size);
    memcpy(r.bufs[r.ins[0]].buf, ids.data(), std::min(ids.size(), r.bufs[r.ins[0]].size)); memcpy(r.bufs[r.ins[1]].buf, pos.data(), std::min(pos.size(), r.bufs[r.ins[1]].size)); r.setdata();
    std::vector<double> t; for (int it = 0; it < N + 2; it++) { auto a = clk::now(); r.setdata(); r.launch(); r.wait(); auto b = clk::now(); if (it >= 2) t.push_back(ms(a, b)); }
    printf("naive 4-card single QPC, whole prefill (T=128), N=%d:\n", N); report("host wall clock per prefill", t);
    std::ofstream o(out + "/logits_naive.bin", std::ios::binary); o.write((const char *)r.bufs[r.out].buf, r.bufs[r.out].size); printf("  logits written (%zu bytes)\n", r.bufs[r.out].size);
    return 0;
  }
  // chain mode
  std::string plan = argv[2], h0f = argv[3], out = argv[4]; int N = argc > 5 ? atoi(argv[5]) : 10; bool dump = false, lazy = false;
  bool swap = false;
  for (int k = 6; k < argc; k++) { std::string o = argv[k]; if (o == "--dump-layers") dump = true; if (o == "--lazy") lazy = true; if (o == "--swap") swap = true; }   // --swap: load/unload programs per layer
  std::vector<std::pair<std::string, std::string>> layers; { std::ifstream f(plan); std::string a, b; while (f >> a >> b) layers.emplace_back(a, b); }
  size_t L = layers.size(); printf("chain: %zu layers, loading %zu programs ...\n", L, 2 * L); fflush(stdout);
  Pair hotp, coldp; hotp.open({0, 1}, "0:1"); coldp.open({2, 3}, "2:3");
  std::vector<Prog> hot(L), cold(L); auto tl0 = clk::now();
  std::vector<std::string> names(2 * L); for (size_t l = 0; l < L; l++) { names[2 * l] = "hot" + std::to_string(l); names[2 * l + 1] = "cold" + std::to_string(l); }
  for (size_t l = 0; l < L; l++) { if (swap) { hot[l].open_qpc(&hotp, layers[l].first, names[2 * l].c_str()); cold[l].open_qpc(&coldp, layers[l].second, names[2 * l + 1].c_str()); continue; }
    hot[l].open(&hotp, layers[l].first, names[2 * l].c_str(), lazy); cold[l].open(&coldp, layers[l].second, names[2 * l + 1].c_str(), lazy);
    if (hot[l].bufs[hot[l].out].size != NEL * 2 || cold[l].bufs[cold[l].out].size != NEL * 2) { fprintf(stderr, "layer %zu: unexpected output size\n", l); return 1; }
    if (l % 8 == 7) { printf("  loaded+activated %zu/%zu layer pairs (%.1f s)\n", l + 1, L, ms(tl0, clk::now()) / 1e3); fflush(stdout); } }
  if (swap) printf("swap mode: %zu QPC files opened in host memory in %.1f s; programs are loaded/unloaded per layer\n", 2 * L, ms(tl0, clk::now()) / 1e3);
  else printf("all %zu programs resident and activated in %.1f s\n", 2 * L, ms(tl0, clk::now()) / 1e3);
  auto h0 = readfile(h0f); if (h0.size() != NEL * 2) { fprintf(stderr, "h0 must be fp16 [T,H]\n"); return 1; }
  std::vector<float> h(NEL); for (size_t i = 0; i < NEL; i += 8) _mm256_storeu_ps(h.data() + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(h0.data() + i * 2))));
  std::vector<float> hin = h; std::vector<double> t_total; std::vector<std::vector<double>> t_layer(L), t_act(L), t_run(L);
  int warm = getenv("E2E_WARM") ? atoi(getenv("E2E_WARM")) : 2;
  for (int it = 0; it < N + warm; it++) {
    h = hin; auto a = clk::now();
    for (size_t l = 0; l < L; l++) {
      auto la = clk::now();
      if (lazy) { hot[l].activate(); cold[l].activate(); }
      if (swap) { hot[l].up(); cold[l].up(); }
      auto lb = clk::now();
      f32_to_f16((uint16_t *)hot[l].bufs[hot[l].ins[0]].buf, h.data(), NEL); f32_to_f16((uint16_t *)cold[l].bufs[cold[l].ins[0]].buf, h.data(), NEL);
      hot[l].setdata(); cold[l].setdata(); hot[l].launch(); cold[l].launch(); hot[l].wait(); cold[l].wait();
      add_f16_to_f32((const uint16_t *)hot[l].bufs[hot[l].out].buf, (const uint16_t *)cold[l].bufs[cold[l].out].buf, h.data(), NEL);
      auto lc = clk::now();
      if (lazy) { hot[l].deactivate(); cold[l].deactivate(); }
      if (swap) { hot[l].down(); cold[l].down(); }
      if (it >= warm) { t_layer[l].push_back(ms(la, clk::now())); t_act[l].push_back(ms(la, lb) + ms(lc, clk::now())); t_run[l].push_back(ms(lb, lc)); }
      if (dump && it == N + warm - 1) { std::ofstream o(out + "/h_out_L" + std::to_string(l) + ".bin", std::ios::binary); o.write((const char *)h.data(), NEL * 4); }
    }
    auto b = clk::now(); if (it >= warm) t_total.push_back(ms(a, b));
  }
  printf("path-2 chain, %zu layers, host adds between layers, N=%d:\n", L, N); report("host wall clock per prefill (48 layers)", t_total);
  double sum_med = 0; for (size_t l = 0; l < L; l++) { auto v = t_layer[l]; std::sort(v.begin(), v.end()); sum_med += v[v.size() / 2]; }
  if (swap && Prog::n_swaps) printf("  swap breakdown per program (avg over %d): create %.0f ms, load %.0f ms, activate %.0f ms, execobj create %.0f ms, execobj release %.0f ms, deactivate %.0f ms, unload %.0f ms, release %.0f ms\n", Prog::n_swaps, Prog::t_create / Prog::n_swaps, Prog::t_load / Prog::n_swaps, Prog::t_activate / Prog::n_swaps, Prog::t_execobj / Prog::n_swaps, Prog::t_relexec / Prog::n_swaps, Prog::t_deact / Prog::n_swaps, Prog::t_unload / Prog::n_swaps, Prog::t_release / Prog::n_swaps);
  if (lazy || swap) { double sa = 0, sr = 0; for (size_t l = 0; l < L; l++) { auto v = t_act[l]; std::sort(v.begin(), v.end()); sa += v[v.size() / 2]; auto w = t_run[l]; std::sort(w.begin(), w.end()); sr += w[w.size() / 2]; } printf("  %s: program swap (create+load+activate / deactivate+unload+release) per layer sum %.2f ms;  run (write x, launch, wait, add) sum %.2f ms\n", swap ? "swap mode" : "lazy mode", sa, sr); }
  printf("  sum of per-layer medians %.2f ms;  per layer (median ms):", sum_med); for (size_t l = 0; l < L; l++) { auto v = t_layer[l]; std::sort(v.begin(), v.end()); printf(" %.2f", v[v.size() / 2]); } printf("\n");
  std::ofstream o(out + "/h_final_f32.bin", std::ios::binary); o.write((const char *)h.data(), NEL * 4); printf("  final residual stream written\n");
  return 0;
}
