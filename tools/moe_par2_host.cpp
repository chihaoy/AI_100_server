// Path 2 of naive_vs_64_16 in C++: two QPCs (hot lane group on cards 0,1; cold lane group on cards 2,3), launched from one
// host thread with the async C API, outputs added on the host with F16C (fp16 -> fp32) vectorised code.
// Measures host wall clock per inference: each QPC solo, serial, concurrent (+ add).  Zero-copy IO buffers, set up once.
//   g++ -O2 -std=c++17 -mavx2 -mf16c -I/opt/qti-aic/dev/inc tools/moe_par2_host.cpp -o /tmp/moe_par2_host \
//       -L/opt/qti-aic/dev/lib/x86_64 -lQAic -Wl,-rpath,/opt/qti-aic/dev/lib/x86_64 -lpthread -ldl
//   /tmp/moe_par2_host <hot_qpc_dir> <cold_qpc_dir> <x.bin> [N] [ref_fp16.bin ...]
#include "QAicApi.h"
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

#define CK(call) do { QStatus s_ = (call); if (s_ != QS_SUCCESS) { fprintf(stderr, "%s failed: %d\n", #call, (int)s_); exit(1); } } while (0)
static const size_t T = 128, H = 2048, NEL = T * H, NBYTES = NEL * 2;

struct Prog {
  QAicContext *ctx = nullptr; QAicQpcObj *qpc = nullptr; QAicProgram *prog = nullptr; QAicQueue *q = nullptr; QAicExecObj *eo = nullptr;
  uint32_t nbuf = 0; QBuffer *bufs = nullptr; std::vector<QBuffer> own; int in = -1, out = -1;
  void open(const std::vector<QID> &devs, const std::string &mapping, const std::string &dir, const char *name) {
    CK(qaicCreateContext(&ctx, nullptr, devs.size(), devs.data(), nullptr, nullptr, nullptr, nullptr));
    if (getenv("PAR2_DEBUG")) qaicSetLogLevel(ctx, QL_DEBUG);
    CK(qaicOpenQpcFile(&qpc, (dir + "/programqpc.bin").c_str()));
    QAicProgramProperties_t pp; CK(qaicProgramPropertiesInitDefault(&pp)); if (!mapping.empty()) pp.devMapping = mapping.c_str();
    CK(qaicCreateProgram(ctx, &prog, &pp, devs[0], name, qpc));
    CK(qaicLoadProgram(prog)); CK(qaicRunActivationCmd(prog, QAIC_PROGRAM_CMD_ACTIVATE_FULL));
    CK(qaicCreateQueue(ctx, &q, nullptr, devs[0]));
    QData iod{}; CK(qaicProgramGetIoDescriptor(prog, &iod));   // execObj needs the program's serialised IO descriptor
    // multi-device programs do not support runtime-allocated (DmaBuf) IO buffers ("DmaBuf mode unsupported for
    // multi-qaic program"), so allocate heap QBuffers ourselves and hand them over with setData (as the Python binding does)
    QAicExecObjProperties_t ep = QAIC_EXECOBJ_PROPERTIES_DEFAULT;
    CK(qaicCreateExecObj(ctx, &eo, &ep, prog, &iod, nullptr, nullptr));
    QAicIoBufferInfo_t *info = nullptr; CK(qaicProgramGetIoBufferInfo(prog, &info));
    nbuf = info->numBufferMappings; own.assign(nbuf, QBuffer{}); bufs = own.data();
    for (uint32_t i = 0; i < nbuf; i++) {
      auto &m = info->bufferMappings[i];
      QBuffer &qb = own[m.index]; qb.size = m.size; qb.buf = (uint8_t *)aligned_alloc(4096, (m.size + 4095) / 4096 * 4096); qb.type = QBUFFER_TYPE_HEAP; qb.handle = 0; qb.offset = 0;
      if (m.ioType == BUFFER_IO_TYPE_INPUT) in = m.index; else if (m.ioType == BUFFER_IO_TYPE_OUTPUT) out = m.index;
    }
    fprintf(stderr, "%s: %u buffers, in=%d (%zu B) out=%d (%zu B)\n", name, nbuf, in, bufs[in].size, out, bufs[out].size);
  }
  void set_input(const void *x) { memcpy(bufs[in].buf, x, NBYTES); CK(qaicExecObjSetData(eo, nbuf, bufs)); }
  void launch() { CK(qaicEnqueueExecObj(q, eo, nullptr)); }
  void wait() { CK(qaicExecObjWaitForCompletion(eo, 10000000)); }
  const uint16_t *y() const { return (const uint16_t *)bufs[out].buf; }
  void close() {   // free the cards so another program can load on them
    qaicReleaseExecObj(eo); qaicReleaseQueue(q); qaicRunActivationCmd(prog, QAIC_PROGRAM_CMD_DEACTIVATE_FULL); qaicUnloadProgram(prog);
    qaicReleaseProgram(prog); qaicCloseQpc(qpc); qaicReleaseContext(ctx); for (auto &b : own) free(b.buf); own.clear();
  }
};

// y32 = a(fp16) + b(fp16), F16C, 262144 elements
static void add_f16(const uint16_t *a, const uint16_t *b, float *y) {
  for (size_t i = 0; i < NEL; i += 8) {
    __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(a + i)));
    __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(b + i)));
    _mm256_storeu_ps(y + i, _mm256_add_ps(va, vb));
  }
}
static std::vector<uint16_t> load16(const std::string &f) { std::vector<uint16_t> v(NEL); std::ifstream s(f, std::ios::binary); s.read((char *)v.data(), NBYTES); if (!s) { fprintf(stderr, "cannot read %s\n", f.c_str()); exit(1); } return v; }
static float h2f(uint16_t h) { return _cvtsh_ss(h); }
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
static void report(const char *name, std::vector<double> t) { std::sort(t.begin(), t.end()); size_t n = t.size(); printf("  %-52s median %6.2f ms  p10 %6.2f  p90 %6.2f  min %6.2f\n", name, t[n / 2], t[n / 10], t[9 * n / 10], t[0]); }

int main(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s hot_qpc_dir cold_qpc_dir x.bin [N] [ref.bin ...]\n", argv[0]); return 1; }
  int N = argc > 4 ? atoi(argv[4]) : 50, warm = 5;
  auto x = load16(argv[3]);
  for (int k = 5; k < argc; k++) {           // --ref1=<qpc_dir>: single-card QPC on card 0 (e.g. an identity graph) = pure launch + PCIe floor
    std::string arg = argv[k]; if (arg.rfind("--ref1=", 0) != 0) continue;
    Prog r; r.open({0}, "", arg.substr(7), "ref1"); r.set_input(x.data()); std::vector<double> t;
    for (int it = 0; it < N + warm; it++) { auto a = clk::now(); r.launch(); r.wait(); auto b = clk::now(); if (it >= warm) t.push_back(ms(a, b)); }
    printf("single-card QPC, same loop:\n"); report(("1-card QPC " + arg.substr(7)).c_str(), t); r.close();
  }
  for (int k = 5; k < argc; k++) {           // --ref4=<qpc_dir>: a 4-card single QPC (naive / 2-partition) timed with the same loop
    std::string arg = argv[k]; if (arg.rfind("--ref4=", 0) != 0) continue;
    Prog r; r.open({0, 1, 2, 3}, "0:1:2:3", arg.substr(7), "ref4"); r.set_input(x.data()); std::vector<double> t;
    for (int it = 0; it < N + warm; it++) { auto a = clk::now(); r.launch(); r.wait(); auto b = clk::now(); if (it >= warm) t.push_back(ms(a, b)); }
    r.close();
    printf("4-card single QPC, same loop:\n"); report(("4-card QPC " + arg.substr(7).substr(arg.substr(7).rfind('/', arg.size() - 9) + 1)).c_str(), t);
  }
  Prog hot, cold;
  hot.open({0, 1}, "0:1", argv[1], "hot"); cold.open({2, 3}, "2:3", argv[2], "cold");
  hot.set_input(x.data()); cold.set_input(x.data());     // x is constant: input buffers written once, reused
  std::vector<float> y(NEL);
  std::vector<double> t_hot, t_cold, t_serial, t_conc, t_conc_add, t_add;
  for (int it = 0; it < N + warm; it++) { auto a = clk::now(); hot.launch(); hot.wait(); auto b = clk::now(); if (it >= warm) t_hot.push_back(ms(a, b)); }
  for (int it = 0; it < N + warm; it++) { auto a = clk::now(); cold.launch(); cold.wait(); auto b = clk::now(); if (it >= warm) t_cold.push_back(ms(a, b)); }
  for (int it = 0; it < N + warm; it++) { auto a = clk::now(); hot.launch(); hot.wait(); cold.launch(); cold.wait(); add_f16(hot.y(), cold.y(), y.data()); auto b = clk::now(); if (it >= warm) t_serial.push_back(ms(a, b)); }
  for (int it = 0; it < N + warm; it++) { auto a = clk::now(); hot.launch(); cold.launch(); hot.wait(); cold.wait(); auto b = clk::now(); add_f16(hot.y(), cold.y(), y.data()); auto c = clk::now(); if (it >= warm) { t_conc.push_back(ms(a, b)); t_conc_add.push_back(ms(a, c)); t_add.push_back(ms(b, c)); } }
  // chained form: every iteration writes a fresh x (fp32 -> fp16, F16C) into BOTH input buffers and calls setData again,
  // as a real layer-by-layer loop would (the next layer's x is this layer's sum); the sum is fed back as the next x
  std::vector<double> t_chain; std::vector<float> xf(NEL); for (size_t i = 0; i < NEL; i++) xf[i] = h2f(x[i]);
  auto store_f16 = [&](uint16_t *dst, const float *src) { for (size_t i = 0; i < NEL; i += 8) _mm_storeu_si128((__m128i *)(dst + i), _mm256_cvtps_ph(_mm256_loadu_ps(src + i), 0)); };
  for (int it = 0; it < N + warm; it++) {
    auto a = clk::now();
    store_f16((uint16_t *)hot.bufs[hot.in].buf, xf.data()); store_f16((uint16_t *)cold.bufs[cold.in].buf, xf.data());
    CK(qaicExecObjSetData(hot.eo, hot.nbuf, hot.bufs)); CK(qaicExecObjSetData(cold.eo, cold.nbuf, cold.bufs));
    hot.launch(); cold.launch(); hot.wait(); cold.wait(); add_f16(hot.y(), cold.y(), y.data());
    auto b = clk::now(); if (it >= warm) t_chain.push_back(ms(a, b));
    for (size_t i = 0; i < NEL; i++) xf[i] = 0.5f * y[i];   // pretend the next layer's input is this layer's sum (scaled to stay in range)
  }
  hot.set_input(x.data()); cold.set_input(x.data());
  printf("C++ host loop, one thread, async enqueue, zero-copy buffers, F16C add; N=%d after %d warm-up:\n", N, warm);
  report("chained: write new x into both inputs + setData + run both + add", t_chain);
  report("hot solo (enqueue -> completion)", t_hot); report("cold solo", t_cold); report("hot -> cold serial + add", t_serial);
  report("hot || cold concurrent (enqueue both, wait both)", t_conc); report("  + host add fp16->fp32 (F16C)", t_conc_add); report("host add alone", t_add);
  for (int k = 5; k < argc; k++) if (std::string(argv[k]).rfind("--dump=", 0) == 0) { std::ofstream o(argv[k] + 7, std::ios::binary); o.write((const char *)y.data(), NEL * 4); }   // y as fp32 [T,H]
  float maxy = 0; for (size_t i = 0; i < NEL; i++) maxy = std::max(maxy, std::fabs(y[i]));
  for (int k = 5; k < argc; k++) {           // reference outputs: fp16 [T,H] files; a '+' joins two files to be summed
    std::string arg = argv[k]; if (arg.rfind("--", 0) == 0) continue; std::vector<float> r(NEL, 0.f); size_t p = 0;
    while (p <= arg.size()) { size_t q = arg.find('+', p); if (q == std::string::npos) q = arg.size(); auto v = load16(arg.substr(p, q - p)); for (size_t i = 0; i < NEL; i++) r[i] += h2f(v[i]); p = q + 1; }
    int bad = 0, close = 0; float md = 0;
    for (size_t t = 0; t < T; t++) { float d = 0; for (size_t j = 0; j < H; j++) d = std::max(d, std::fabs(y[t * H + j] - r[t * H + j])); md = std::max(md, d); if (d > 0.01f * maxy) bad++; if (d < 1e-3f) close++; }
    printf("  vs %s: max|diff| %.4f  tokens<1e-3 %d/128  tokens>1%% of max|y|=%.3f: %d/128\n", arg.c_str(), md, close, maxy, bad);
  }
  return 0;
}
